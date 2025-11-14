# run_vla_control.py

import numpy as np

from core.env_wrapper import PandaEnv
from core.vla_agent import OpenVLAAgent
from core.diagnostics_n_logging import init_control_logger
from core.ik_solver import solve_IK
from core.control_utils import apply_action_to_pose, create_pid
from core.config import load_cfg


def main():
    # Load YAML + CLI config (includes --prompt, --cfg, --unnorm_key)
    cfg, _ = load_cfg()

    # --- setup ---
    panda = PandaEnv(xml_path=cfg.xml_path, camera_name=cfg.camera_name)
    env = panda.env  # keep original env for low-level access if needed

    agent = OpenVLAAgent(cfg)
    pid = create_pid(env, cfg.pid)

    log_f, log_w, log_path = init_control_logger(
        n_actions=7,
        n_joints=env.n_ctrl,
    )
    print("Logging control data to:", log_path)

    # start with current joint config as target (same as original)
    desired_q = env.get_q(joint_idxs=env.ctrl_joint_idxs)
    action = np.zeros(7, dtype=np.float32)   # default until first VLA call

    steps_per_policy = cfg.sim_hz // cfg.policy_hz

    while env.tick < cfg.max_tick:
        # --- high-level policy ---
        if env.tick % steps_per_policy == 0:
            image = panda.get_image()
            action = agent.act(image)

            # keep original debug print
            print(env.tick, action)

            # 4) convert action -> EE target pose + gripper
            p_trgt, R_trgt, gripper_q = apply_action_to_pose(
                env,
                action,
                cfg.steps,
                body_name="panda_eef",
            )

            # 5) IK: EE pose -> arm joint angles (NO reset)
            q_arm_curr = env.get_q(joint_idxs=env.rev_joint_idxs)
            q_arm_trgt = solve_IK(
                env,
                max_tick=cfg.max_tick,
                p_trgt=p_trgt,
                R_trgt=R_trgt,
                body_name="panda_eef",
                curr_q=q_arm_curr,
                is_render=False,
                VERBOSE=False,
                reset_env=False,       # For online usage (same as original)
            )

            # 6) build full joint target (arm + gripper)
            desired_q = np.concatenate([q_arm_trgt, gripper_q])

        # --- low-level PID control every step ---
        pid.update(x_trgt=desired_q)
        pid.update(
            t_curr=env.get_sim_time(),
            x_curr=env.get_q(joint_idxs=env.ctrl_joint_idxs),
            VERBOSE=False,
        )

        tau_pid = pid.out()
        tau_grav = env.data.qfrc_bias[env.ctrl_joint_idxs]   # gravity + Coriolis etc.
        torque = tau_pid + tau_grav

        q_curr_all = env.get_q(joint_idxs=env.ctrl_joint_idxs)

        # ----- LOGGING ROW -----
        row = []
        row.append(env.tick)
        row.append(env.get_sim_time())

        # Action is a 7D array
        row.extend(list(action))       # act_0..act_6

        # Current joints
        row.extend(list(q_curr_all))   # q_0..q_8

        # Desired joints
        row.extend(list(desired_q))    # q_des_0..q_des_8

        # Torques
        row.extend(list(torque))       # tau_0..tau_8

        log_w.writerow(row)
        # ------------------------

        panda.step(torque)

        if (env.tick % 3) == 0:
            panda.render()

    log_f.close()
    print("Done, logs saved to:", log_path)

    env.close_viewer()
    print("Done")


if __name__ == "__main__":
    main()
