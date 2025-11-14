# core/control_utils.py

import numpy as np
from scipy.spatial.transform import Rotation as R
from src.PID import PID_ControllerClass


def set_gripper(desired_q, option="open"):
    if option == "open":
        desired_q[7], desired_q[8] = np.pi, -np.pi
    elif option == "close":
        desired_q[7], desired_q[8] = 0.0, 0.0
    return desired_q


def apply_action_to_pose(env, action, step_cfg, body_name="panda_eef"):
    """
    Map a 7D action to EE pose + gripper targets.

    step_cfg: cfg.steps (StepsCfg) with
      - step_scale_xyz
      - step_scale_rpy
    """
    dx, dy, dz, droll, dpitch, dyaw, dgrip = action

    d_pos = np.array([dx, dy, dz]) * step_cfg.step_scale_xyz
    d_rpy = np.array([droll, dpitch, dyaw]) * step_cfg.step_scale_rpy

    # Current EE pose
    p_curr = env.data.body(body_name).xpos.copy()
    R_curr = env.data.body(body_name).xmat.reshape(3, 3).copy()

    # New target pose
    p_trgt = p_curr + d_pos
    R_delta = R.from_euler("xyz", d_rpy).as_matrix()
    R_trgt = R_curr @ R_delta

    # Gripper target joints from dgrip
    if dgrip > 0:
        gripper_q = np.array([0.0, 0.0])      # close
    else:
        gripper_q = np.array([np.pi, -np.pi]) # open

    return p_trgt, R_trgt, gripper_q


def create_pid(env, pid_cfg):
    """
    pid_cfg: cfg.pid with fields kp, ki, kd
    """
    pid = PID_ControllerClass(
        name="PID",
        dim=env.n_ctrl,
        k_p=pid_cfg.kp,
        k_i=pid_cfg.ki,
        k_d=pid_cfg.kd,
        out_min=env.ctrl_ranges[env.ctrl_joint_idxs, 0],
        out_max=env.ctrl_ranges[env.ctrl_joint_idxs, 1],
        ANTIWU=True,
    )
    pid.reset()
    return pid
