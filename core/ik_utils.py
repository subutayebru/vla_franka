#ik_utils.py
import numpy as np
from ik_single import get_q_from_ik_single

def clamp_workspace(p, ws):
    import numpy as np
    return np.array([np.clip(p[0], *ws[0]), np.clip(p[1], *ws[1]), np.clip(p[2], *ws[2])], dtype=np.float32)

def set_gripper(qpos, option="open"):
    out=qpos.copy()
    if option=="open": out[7], out[8] = np.pi, -np.pi
    else:              out[7], out[8] = 0.0, 0.0
    return out

def ik_servo_step(env, target_pos, target_quat, rate=0.25, max_step=0.05):
    q_now = env.data.qpos.copy()
    q_ik  = get_q_from_ik_single(env, target_pos, target_quat, q_init=q_now, site_name="grip_site", arm_joint_names=None)
    if (q_ik is None) or (not np.isfinite(q_ik).all()): return q_now
    q_des = (1.0-rate)*q_now + rate*q_ik
    dq = q_des[env.ctrl_joint_idxs] - q_now[env.ctrl_joint_idxs]
    peak = float(np.max(np.abs(dq))) if dq.size else 0.0
    if peak > max_step:
        q_des[env.ctrl_joint_idxs] = q_now[env.ctrl_joint_idxs] + dq*(max_step/peak)
    return q_des
