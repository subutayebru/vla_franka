import numpy as np

from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
import torch
from torchvision.transforms import ToPILImage
from PIL import Image

from src.mujoco_parser import MuJoCoParserClass
from src.PID import PID_ControllerClass
from scipy.spatial.transform import Rotation as R
from core.ik_solver import solve_IK

import os
from core.diagnostics_n_logging import make_control_logger, init_hud, update_hud 
os.environ["MUJOCO_GL"] = "egl"  # set before importing mujoco
import mujoco
import numpy as np

CAMERA_FOR_VLA = 'panda_eye_in_hand' #'panda_eye_in_hand'           # use a named camera in XML, e.g. "camera_top"; None uses free cam

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b", #"openvla/openvla-v01-7b"
    trust_remote_code=True,
)

vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", #"openvla/openvla-v01-7b"
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)


def set_gripper(desired_q, option="open"):
    if option == "open":
        desired_q[7], desired_q[8] = np.pi, -np.pi
    elif option == "close":
        desired_q[7], desired_q[8] = 0.0, 0.0
    return desired_q

def apply_action_to_pose(env, action, body_name='panda_eef'):
    dx, dy, dz, droll, dpitch, dyaw, dgrip = action

    # Maybe scale these if needed
    pos_scale = 0.5#0.05   # meters per step, tune
    rot_scale = 0.8#0.2    # radians per step, tune

    d_pos = np.array([dx, dy, dz]) * pos_scale
    d_rpy = np.array([droll, dpitch, dyaw]) * rot_scale

    # Current EE pose
    p_curr = env.data.body(body_name).xpos.copy()
    R_curr = env.data.body(body_name).xmat.reshape(3, 3).copy()

    # New target pose
    p_trgt = p_curr + d_pos
    R_delta = R.from_euler('xyz', d_rpy).as_matrix()
    R_trgt = R_curr @ R_delta

    # Gripper target joints from dgrip
    if dgrip > 0:    # you can tune threshold
        gripper_q = np.array([0.0, 0.0])          # close
    else:
        gripper_q = np.array([np.pi, -np.pi])     # open

    return p_trgt, R_trgt, gripper_q

def main():
    # ---- MuJoCo env ----
    xml_path = '/home/es_admin/vla-franka/Simple-MuJoCo-PickNPlace/asset/panda/franka_panda_w_objs.xml'
    env = MuJoCoParserClass(name='Panda', rel_xml_path=xml_path, VERBOSE=False)
    env.forward()

    env.init_viewer(viewer_title="VLA control", viewer_width=1600, viewer_height=900,
                    viewer_hide_menus=False)
    env.update_viewer(cam_id=0)
    env.reset()

    # print("rev_joint_idxs:", env.rev_joint_idxs)
    # print("ctrl_joint_idxs:", env.ctrl_joint_idxs)
    # print("n_ctrl:", env.n_ctrl)

    # One-time sanity: desired_q length should match number of controlled joints
    q_curr = env.get_q(joint_idxs=env.ctrl_joint_idxs)
    #print("Initial q_curr shape:", q_curr.shape)

    # Renderer using env.model/env.data
    renderer = mujoco.Renderer(env.model, 480, 640)
    
    # ---- HUD init ----
    renderer.update_scene(env.data, CAMERA_FOR_VLA) #if only renderer.update_scene(env.data)   it shows the whole environemnt
    initial_rgb = renderer.render()
    fig, ax, im = init_hud(initial_rgb)
    
    # ---- PID ----
    PID = PID_ControllerClass(
        name='PID', dim=env.n_ctrl,
        k_p=800.0,
        k_i=20.0,
        k_d=100.0,
        out_min=env.ctrl_ranges[env.ctrl_joint_idxs, 0],
        out_max=env.ctrl_ranges[env.ctrl_joint_idxs, 1],
        ANTIWU=True
    )
    PID.reset()

    # ---- Control logger ----
    log_f, log_w, log_path = make_control_logger(
        n_actions=7,
        n_joints=env.n_ctrl,   # 9 joints (7 arm + 2 gripper)
    )
    print("Logging control data to:", log_path)

    # ---- VLA setup ----
    instruction = "pick up the yellow cube from the top"
    prompt = "In:What action should the robot take to {instruction}?\nOut:"

    max_tick = 100000
    policy_hz = 5                        # call policy 5 Hz
    sim_hz = 500                         # assume ~500 sim steps/sec
    steps_per_policy = sim_hz // policy_hz

    # start with current joint config as target
    desired_q = env.get_q(joint_idxs=env.ctrl_joint_idxs) #???????????? SAME AS Q_CURR

    
    while env.tick < max_tick:

        # --- Call VLA every few steps ---
        if env.tick % steps_per_policy == 0:
            # 1) render current observation
            renderer.update_scene(env.data, CAMERA_FOR_VLA)
            rgb = renderer.render()
            image = Image.fromarray(rgb)

            # ---> HUD update <---
            hud_title = f"tick={env.tick}"
            update_hud(ax, im, rgb, hud_title)
            # --------------------


            # 2) prepare inputs for VLA
            inputs = processor(prompt, image)
            vision_mod = vla.vision_backbone
            vision_param = next(vision_mod.parameters())
            img_dtype = vision_param.dtype
            img_device = vision_param.device
            inputs["pixel_values"] = inputs["pixel_values"].to(
                device=img_device, dtype=img_dtype
            )

            #3) get action
            with torch.inference_mode():
                action = vla.predict_action(
                    **inputs,
                    unnorm_key= 'bridge_orig', #nyu_franka_play_dataset_converted_externally_to_rlds',
                    do_sample=False,
                )

            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            print(env.tick,  action)

            #action = np.array([0.20, 0.20, -0.10,  0.0, 0.0, 0.0,   -1.0]) #dummy action for debugging Ik solver and PID
            #action = np.array([0.0, 0.0, 0.0,  0.0, 0.0, 0.0,   1.0]) #dummy action for debugging Ik solver and PID

            # 4) convert action -> EE target pose + gripper
            p_trgt, R_trgt, gripper_q = apply_action_to_pose(env, action, body_name='panda_eef')

            p_curr_dbg = env.data.body('panda_eef').xpos.copy()
            # print(f"[{env.tick}] VLA action:", action)
            # print(f"    EE pos before IK: {p_curr_dbg}")
            # print(f"    EE pos target   : {p_trgt}")

            # 5) IK: EE pose -> arm joint angles (NO reset)
            q_arm_curr = env.get_q(joint_idxs=env.rev_joint_idxs)
            q_arm_trgt = solve_IK(
                env,
                max_tick=max_tick,
                p_trgt=p_trgt,
                R_trgt=R_trgt,
                body_name='panda_eef',
                curr_q=q_arm_curr,
                is_render=False,
                VERBOSE=False,
                reset_env=False,       # For online usage
            )

            # 6) build full joint target (arm + gripper)
            desired_q = np.concatenate([q_arm_trgt, gripper_q])

        # --- Low-level PID control every step ---
        PID.update(x_trgt=desired_q)
        PID.update(
            t_curr=env.get_sim_time(),
            x_curr=env.get_q(joint_idxs=env.ctrl_joint_idxs),
            VERBOSE=False
        )

        tau_pid = PID.out()
        tau_grav = env.data.qfrc_bias[env.ctrl_joint_idxs]   # gravity + Coriolis etc.

        torque = tau_pid + tau_grav

        q_curr = env.get_q(joint_idxs=env.ctrl_joint_idxs)
        
        
        tracking_err = np.linalg.norm(desired_q - q_curr)
        q_curr_all = env.get_q(joint_idxs=env.ctrl_joint_idxs)
        q_curr_arm = q_curr_all[:7]
        q_curr_grip = q_curr_all[7:]

        des_arm = desired_q[:7]
        des_grip = desired_q[7:]

        err_arm = np.linalg.norm(des_arm - q_curr_arm)
        err_grip = np.linalg.norm(des_grip - q_curr_grip)

        #print(f"[{env.tick}] arm_err={err_arm:.4f}, grip_err={err_grip:.4f}, total={np.linalg.norm(desired_q - q_curr_all):.4f}")
        #print('current position of the arm', env.rev_joint_idxs , q_curr)
        
        # ----- LOGGING ROW -----
        # If action is only computed every few steps, we still log the last action
        row = []
        row.append(env.tick)
        row.append(env.get_sim_time())

        # Action is a 7D array
        row.extend(list(action))          # act_0..act_6

        # Current joints
        row.extend(list(q_curr_all))      # q_0..q_8

        # Desired joints
        row.extend(list(desired_q))       # q_des_0..q_des_8

        # Torques
        row.extend(list(torque))          # tau_0..tau_8

        log_w.writerow(row)
        # ------------------------

        env.step(ctrl=torque, ctrl_idxs=env.ctrl_joint_idxs)

        if (env.tick % 3) == 0:
            env.render()
    
    log_f.close()
    print("Done, logs saved to:", log_path)
        
            
    env.close_viewer()
    print("Done")


if __name__ == "__main__":
    main()
