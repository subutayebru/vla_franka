# vla_pnp.py
import os
# Keep GUI visible: do NOT force EGL for this script
os.environ.pop("MUJOCO_GL", None)             # or: os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import numpy as np
from PIL import Image

from src.mujoco_parser import MuJoCoParserClass
from src.PID import PID_ControllerClass
from ik_single import get_q_from_ik_single
from vla_openvla import OpenVLA

def get_site_id(model, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid == -1:
        raise ValueError(f"Site not found: {name}")
    return sid

# === ADD set_gripper HERE (taken from your pnp.py) ============================
def set_gripper(desired_q: np.ndarray, option: str = "open") -> np.ndarray:
    """
    Franka fingers assumed at indices 7 and 8 in qpos.
    open  -> (pi, -pi)
    close -> (0.0, 0.0)
    """
    desired_q = desired_q.copy()
    if option == "open":
        desired_q[7], desired_q[8] = np.pi, -np.pi
    elif option == "close":
        desired_q[7], desired_q[8] = 0.0, 0.0
    return desired_q
# ============================================================================

XML_ABS = "/home/es_admin/vla-franka/Simple-MuJoCo-PickNPlace/asset/panda/franka_panda_w_objs.xml"
EE_SITE = "grip_site"   # <-- CHANGE to your end-effector name 

def build_renderer(model):
    # Offscreen buffer for RGB frames to feed VLA (no extra window)
    return mujoco.Renderer(model, height=480, width=640)
_last_a7 = None
def smooth(a7, alpha=0.5):
    global _last_a7
    if _last_a7 is None:
        _last_a7 = a7.astype(np.float32)
        return _last_a7
    _last_a7 = alpha * a7.astype(np.float32) + (1.0 - alpha) * _last_a7
    return _last_a7
def vla_to_desired_q(env, renderer, vla, instruction, step_scale=0.03):
    """
    1) Grab RGB offscreen while GUI is open.
    2) VLA predicts 7D delta (xyz + rpy + grip) in [-1,1].
    3) Convert to target EE pose, run single-pose IK -> desired_q (full qpos).
    4) Set gripper in desired_q.
    """
    # 1) RGB frame
    renderer.update_scene(env.data)
    rgb = renderer.render()  # HxWx3 uint8

    # 2) VLA inference
    a7 = vla.predict(rgb, instruction)  # [dx,dy,dz, droll,dpitch,dyaw, grip] ~ [-1,1]
    a7 = smooth(a7, alpha=0.5)
    dx, dy, dz, grip = a7[0], a7[1], a7[2], a7[-1]
    dx, dy, dz = dx * 0.005, dy * 0.005, dz * 0.005
    #dx, dy, dz = dx * step_scale, dy * step_scale, dz * step_scale

    # 3) current EE pose
    sid = get_site_id(env.model, "grip_site")          # site for position
    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "panda_eef")  # body for orientation

    p_curr = env.data.site_xpos[sid].copy()
    q_curr = env.data.xquat[bid].copy() 

    target_pos  = p_curr + np.array([dx, dy, dz], dtype=np.float32)
    target_quat = q_curr  # keep orientation for now
    
    if np.linalg.norm([dx, dy, dz]) < 1e-5:
        return env.data.qpos.copy()

    desired_q = get_q_from_ik_single(
        env, target_pos, target_quat,
        q_init=env.data.qpos.copy(),
        site_name=EE_SITE,
        arm_joint_names=None  # or provide your exact 7 arm joint names
    )

    # 4) gripper
    desired_q = set_gripper(desired_q, option=("close" if grip > 0 else "open"))
    return desired_q

def main():
    # Separate model/data for the offscreen renderer (same XML)
    model = mujoco.MjModel.from_xml_path(XML_ABS)
    data  = mujoco.MjData(model)
    renderer = build_renderer(model)

    # Your GUI env wrapper
    env = MuJoCoParserClass(name="Panda", rel_xml_path="asset/panda/franka_panda_w_objs.xml", VERBOSE=False)
    env.forward()
    env.init_viewer(viewer_title="PNP + VLA", viewer_width=1600, viewer_height=900, viewer_hide_menus=False)
    env.update_viewer(cam_id=0)
    env.reset()

    # PID controller (same tuning you used)
    PID = PID_ControllerClass(
        name="PID", dim=env.n_ctrl,
        k_p=800.0, k_i=20.0, k_d=100.0,
        out_min=env.ctrl_ranges[env.ctrl_joint_idxs, 0],
        out_max=env.ctrl_ranges[env.ctrl_joint_idxs, 1],
        ANTIWU=True
    )
    PID.reset()

    # VLA & instruction
    vla = OpenVLA(device="cuda")  # replace with your real OpenVLA loader when ready
    instruction = "Pick up the cube and place it on the platform"

    desired_q = env.data.qpos.copy()
    max_tick = 200000

    while env.tick < max_tick:
        # Refresh target every ~2 sim ticks (~50 Hz if sim ~100 Hz)
        if env.tick % 2 == 0:
            desired_q = vla_to_desired_q(env, renderer, vla, instruction, step_scale=0.03)
        
        # Slice target to controllable joints
        q_trgt_ctrl = desired_q[env.ctrl_joint_idxs]                    # shape (env.n_ctrl,)
        q_curr_ctrl = env.get_q(joint_idxs=env.ctrl_joint_idxs)         # shape (env.n_ctrl,)

        # PID -> torques
        PID.update(
            x_trgt=q_trgt_ctrl,
            x_curr=q_curr_ctrl,
            t_curr=env.get_sim_time(),
            VERBOSE=False
        )
        torque = PID.out()

        # Advance physics
        env.step(ctrl=torque, ctrl_idxs=env.ctrl_joint_idxs)

        # Draw GUI
        if (env.tick % 3) == 0:
            env.render()

    env.close_viewer()
    print("Done.")

if __name__ == "__main__":
    main()
