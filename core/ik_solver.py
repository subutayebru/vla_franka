#core.ik_solver.py
import numpy as np
import mujoco

def r2w(R):
    """
        R to \omega
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.math.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()


def trim_scale(x,th):
    """
        Trim scale
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x


def get_J_body(model,data,body_name,rev_joint_idxs=None):
    J_p = np.zeros((3,model.nv)) # nv: nDoF
    J_R = np.zeros((3,model.nv))
    mujoco.mj_jacBody(model,data,J_p,J_R,data.body(body_name).id)
    if rev_joint_idxs is not None:
        J_p = J_p[:,rev_joint_idxs]
        J_R = J_R[:,rev_joint_idxs]
    J_full = np.array(np.vstack([J_p,J_R]))
    return J_p,J_R,J_full


def solve_IK(env, max_tick, p_trgt, R_trgt, body_name,
             curr_q=None, is_render=False, VERBOSE=False, reset_env=True):
    """
    Numerical IK solver for a single target pose.

    Args:
        env: MuJoCoParserClass environment (must have model, data, rev_joint_idxs, forward(), render(), reset()).
        max_tick: Used here as max number of IK iterations (int).
        p_trgt: (3,) target position of the body in world frame.
        R_trgt: (3,3) target rotation matrix of the body in world frame.
        body_name: name of the MuJoCo body to control (e.g. 'panda_eef').
        curr_q: optional initial joint configuration (for env.rev_joint_idxs). If None, uses current env qpos.
        is_render: if True, renders every few iterations during IK.
        VERBOSE: if True, prints error norm per iteration.
        reset_env: 
            - True  -> call env.reset() at the end (useful for offline IK planning, like get_q_from_ik).
            - False -> leave env state at the IK solution (useful for online control with VLA).

    Returns:
        q: (n_rev_joints,) numpy array of joint positions for env.rev_joint_idxs.
    """
    # Initial joint configuration
    if curr_q is None:
        q = env.data.qpos[env.rev_joint_idxs].copy()
    else:
        q = np.array(curr_q, dtype=float).copy()

    # Stopping criterion on pose error
    err_eps = 1e-2

    # Iterative IK
    for it in range(max_tick):
        # Jacobian at current configuration
        J_p, J_R, J_full = get_J_body(env.model, env.data, body_name, rev_joint_idxs=env.rev_joint_idxs)

        # Current pose
        p_curr = env.data.body(body_name).xpos
        R_curr = env.data.body(body_name).xmat.reshape(3, 3)

        # Position error
        p_err = p_trgt - p_curr

        # Orientation error
        R_err = np.linalg.solve(R_curr, R_trgt)     # R_curr^{-1} * R_trgt
        w_err = R_curr @ r2w(R_err)                # 3D rotation error vector

        # Full 6D error
        err = np.concatenate((p_err, w_err))
        err_norm = np.linalg.norm(err)

        # if VERBOSE:
        #     print(f"[IK] iter {it:03d}  err_norm: {err_norm:.6f}")

        # if it % 10 == 0:
        #     print(f"    [IK] iter {it} err_norm: {err_norm:.6f}")

        # Check convergence
        if err_norm < err_eps:
            if VERBOSE:
                print("[IK] Converged")
            break

        # Damped least-squares IK step
        J = J_full  # (6, n_joints)
        eps = 1e-1
        A = J.T @ J + eps * np.eye(J.shape[1])
        b = J.T @ err
        dq = np.linalg.solve(A, b)

        # Limit step size (to avoid crazy jumps)
        dq = trim_scale(dq, th=5.0 * np.pi / 180.0)  # 5 deg in radians

        # Update joints and FK
        q = q + dq
        env.data.qpos[env.rev_joint_idxs] = q
        env.forward()
        
        p_after_ik = env.data.body('panda_eef').xpos.copy()
        # print(f"    EE pos after IK : {p_after_ik}")
        # print(f"    IK position error: {np.linalg.norm(p_trgt - p_after_ik):.6f}")

        # Optional rendering for debugging
        if is_render and (it % 5 == 0):
            env.render()

    # For offline planning (e.g., get_q_from_ik) we want to restore the initial env state
    if reset_env:
        env.reset()

    return q
