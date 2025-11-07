# ik_single.py
import numpy as np
import mujoco

def get_site_id(model, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid == -1:
        raise ValueError(f"Site not found: {name}")
    return sid

def get_joint_id(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid == -1:
        raise ValueError(f"Joint not found: {name}")
    return jid

def get_q_from_ik_single(
    env,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    q_init: np.ndarray = None,
    site_name: str = "ee_site",
    arm_joint_names: list = None,
    pos_tol: float = 1e-3,
    ori_tol: float = 1e-2,
    max_iters: int = 200,
    damping: float = 1e-3,
    step_scale: float = 1.0,
    max_step_rad: float = 0.1,
):
    """
    Solve IK for a single target pose (pos+quat) of a site, returning a full qpos vector.
    """
    m = env.model
    d = env.data

    site_id = get_site_id(m, site_name)

    # choose arm joints (exclude fingers)
    if arm_joint_names is not None:
        arm_jids = [get_joint_id(m, nm) for nm in arm_joint_names]
    else:
        ctrl_jids = list(np.array(env.ctrl_joint_idxs, dtype=int))
        arm_jids = ctrl_jids[:7] if len(ctrl_jids) >= 7 else ctrl_jids

    # map arm joints -> dof indices
    arm_dofs = []
    for jid in arm_jids:
        adr = m.jnt_dofadr[jid]
        dofnum = m.jnt_dofnum[jid]
        arm_dofs.extend(range(adr, adr + dofnum))
    arm_dofs = np.array(arm_dofs, dtype=int)

    # init qpos
    q_work = d.qpos.copy() if q_init is None else q_init.copy()

    def clip_to_limits(qv):
        for jid in arm_jids:
            lo, hi = m.jnt_range[jid]
            if lo < hi:
                qv[jid] = np.clip(qv[jid], lo, hi)
        return qv

    def quat_mul(q1, q2):
        w1,x1,y1,z1 = q1
        w2,x2,y2,z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=np.float64)

    def quat_conj(q):
        q = np.array(q, dtype=np.float64)
        q[1:] *= -1.0
        return q

    jacp = np.zeros((3, m.nv), dtype=np.float64)
    jacr = np.zeros((3, m.nv), dtype=np.float64)

    for _ in range(max_iters):
        d.qpos[:] = q_work
        mujoco.mj_forward(m, d)

        p_curr = d.site_xpos[site_id].copy()
        q_curr = d.site_xquat[site_id].copy()

        err_p = target_pos - p_curr
        q_err = quat_mul(target_quat, quat_conj(q_curr))
        if q_err[0] < 0:  # shortest path
            q_err = -q_err
        err_r = 2.0 * q_err[1:4]  # small-angle approx

        if np.linalg.norm(err_p) < pos_tol and np.linalg.norm(err_r) < ori_tol:
            break

        mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
        Jp = jacp[:, arm_dofs]
        Jr = jacr[:, arm_dofs]
        J = np.vstack([Jp, Jr])            # (6, nd)
        e = np.hstack([err_p, err_r])      # (6,)

        JT = J.T
        H = JT @ J + (damping**2) * np.eye(J.shape[1])
        g = JT @ e
        try:
            dq_arm = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dq_arm = np.linalg.lstsq(H, g, rcond=None)[0]

        dq_arm = np.clip(step_scale * dq_arm, -max_step_rad, +max_step_rad)

        q_new = q_work.copy()
        for i, jid in enumerate(arm_jids):
            q_new[jid] += dq_arm[i]
        q_new = clip_to_limits(q_new)
        q_work = q_new

    return q_work.copy()
