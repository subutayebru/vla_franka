#core/controller.py
import numpy as np, mujoco
from core.ik_utils import clamp_workspace, set_gripper, ik_servo_step

def vla_to_desired_q(env, renderer, vla_agent, instruction, ee_sid, eef_body_id, cam_id, ws, step_scale_xyz, step_scale_rpy):
    renderer.update_scene(env.data, camera=cam_id)
    rgb = renderer.render()

    a7 = vla_agent.act(rgb, instruction)
    fine_mode = getattr(env, "_fine_mode", False)

    dxyz = a7[:3] * step_scale_xyz
    droll, dpitch, dyaw = a7[3]*step_scale_rpy, a7[4]*step_scale_rpy, a7[5]*step_scale_rpy
    grip = a7[6]

    p_curr = env.data.site_xpos[ee_sid].copy()
    R_ee   = env.data.site_xmat[ee_sid].reshape(3,3).copy()
    q_curr = env.data.xquat[eef_body_id].copy()

    # EE frame → world; note sign on z if needed
    dpos_world = R_ee @ np.array([dxyz[0], dxyz[1], -dxyz[2]], dtype=np.float32)
    if fine_mode:
        dpos_world[2] = -abs(dpos_world[2])

    target_pos  = clamp_workspace(p_curr + dpos_world, ws)
    target_quat = q_curr

    q_des = ik_servo_step(env, target_pos, target_quat, rate=0.2, max_step=0.05)
    q_des = set_gripper(q_des, "close" if grip > 0 else "open")
    return q_des, a7, rgb

def fallback_step(env, target_pos, eef_body_id, ws):
    target_pos = clamp_workspace(target_pos, ws)
    target_quat = env.data.xquat[eef_body_id].copy()
    return ik_servo_step(env, target_pos, target_quat, rate=0.25, max_step=0.05)
