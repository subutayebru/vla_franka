# run_vla_control.py

# --- Apple Silicon (MPS) runtime setup ---
# Must run before torch / mujoco are imported (transitively via the modules
# below). See docs/MACOS_PORT.md.
import os

# Fall back to CPU for any op OpenVLA uses that MPS doesn't implement yet,
# instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# EGL is the Linux/Docker offscreen backend; on macOS the default (CGL) is
# correct. Clear it if inherited from the environment.
if os.environ.get("MUJOCO_GL") == "egl":
    del os.environ["MUJOCO_GL"]
# NOTE: run this with plain `python`, not `mjpython`. We render the cameras
# off-screen and display them in a matplotlib "live view" window on the main
# thread (scene overview + the wrist cam OpenVLA sees), so we need matplotlib's
# default interactive macOS backend here — do NOT force Agg.

import numpy as np

from core.env_wrapper import PandaEnv
from core.vla_agent import OpenVLAAgent
from core.diagnostics_n_logging import init_control_logger, action_report, ActionMonitor
from core.ik_solver import solve_IK
from core.control_utils import apply_action_to_pose, create_pid
from core.config import load_cfg


def main():
    # Load YAML + CLI config (includes --prompt, --cfg, --unnorm_key)
    cfg, _ = load_cfg()

    # --- setup ---
    panda = PandaEnv(xml_path=cfg.xml_path, camera_name=cfg.camera_name,
                     view_camera=cfg.view_camera)
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

    # Behavior monitor: target body for the EE→target distance readout.
    # "grasp the yellow cube" => obj_box_07 (rgba 1 0.9 0). Change if the
    # instruction targets a different object.
    target_body = "obj_box_07"
    prev_ee_pos = None
    prev_dist = None

    # Live action sanity-check graph (separate window): distance + dx/dy/dz/grip.
    action_monitor = ActionMonitor(target_name=target_body)

    while env.tick < cfg.max_tick:
        # --- high-level policy ---
        if env.tick % steps_per_policy == 0:
            image = panda.get_image()
            action = agent.act(image)

            # behavior monitor: readable action + EE→target progress
            ee_pos = env.get_p_body("panda_eef").copy()
            target_pos = env.get_p_body(target_body).copy()
            report, prev_dist = action_report(
                env.tick, action, ee_pos, target_pos,
                target_name=target_body,
                prev_ee_pos=prev_ee_pos, prev_dist=prev_dist,
            )
            print(report, flush=True)
            action_monitor.update(env.tick, prev_dist, action)
            prev_ee_pos = ee_pos

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

    action_monitor.close()
    panda.close()
    print("Done")


if __name__ == "__main__":
    main()
