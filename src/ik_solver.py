import numpy as np
import mujoco


def r2w(R):
    """
    Convert a rotation matrix R to its corresponding axis-angle vector w
    (SO(3) -> so(3) logarithm map, approximated).

    Args:
        R: (3,3) rotation matrix.

    Returns:
        w: (3,) rotation vector.
    """
    el = np.array([
        [R[2, 1] - R[1, 2]],
        [R[0, 2] - R[2, 0]],
        [R[1, 0] - R[0, 1]]
    ])
    norm_el = np.linalg.norm(el)

    if norm_el > 1e-10:
        # Standard formula: w = theta * axis, theta from trace(R)
        w = np.arctan2(norm_el, np.trace(R) - 1) / norm_el * el
    elif R[0, 0] > 0 and R[1, 1] > 0 and R[2, 2] > 0:
        # Identity or very close
        w = np.array([[0, 0, 0]]).T
    else:
        # 180-degree rotation edge case
        w = np.math.pi / 2 * np.array([[R[0, 0] + 1],
                                       [R[1, 1] + 1],
                                       [R[2, 2] + 1]])
    return w.flatten()


def trim_scale(x, th):
    """
    Trim (clip) the vector x so that its maximum absolute element is <= th,
    while keeping direction.

    Args:
        x: (n,) vector
        th: scalar threshold

    Returns:
        x_trimmed: (n,) vector
    """
    x = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x * th / x_abs_max
    return x


def get_J_body(model, data, body_name, rev_joint_idxs=None):
    """
    Compute the body Jacobian for a given body.

    Args:
        model: mujoco.MjModel
        data: mujoco.MjData
        body_name: string name of the body
        rev_joint_idxs: optional list/array of joint indices to select (e.g. revolute joints)

    Returns:
        J_p: (3, nv or n_sel) linear part of Jacobian
        J_R: (3, nv or n_sel) angular part of Jacobian
        J_full: (6, nv or n_sel) stacked [J_p; J_R]
    """
    J_p = np.zeros((3, model.nv))  # nv: # of DoFs
    J_R = np.zeros((3, model.nv))

    mujoco.mj_jacBody(model, data, J_p, J_R, data.body(body_name).id)

    if rev_joint_idxs is not None:
        J_p = J_p[:, rev_joint_idxs]
        J_R = J_R[:, rev_joint_idxs]

    J_full = np.vstack([J_p, J_R])
    return J_p, J_R, np.array(J_full)


def solve_IK(env, max_tick, p_trgt, R_trgt, body_name,
             curr_q=None, is_render=False, VERBOSE=False, reset_env=True):
    """
    Numerical IK solver for a single target pose using damped least squares.

    Args:
        env: MuJoCoParserClass environment. Must have:
             - model, data
             - rev_joint_idxs (indices of arm joints in qpos)
             - forward(), render(), reset()
        max_tick: maximum number of IK iterations (int)
        p_trgt: (3,) target position of the body in world frame
        R_trgt: (3,3) target rotation matrix of the body in world frame
        body_name: name of the body to control (e.g. 'panda_eef')
        curr_q: optional initial joint configuration for env.rev_joint_idxs.
                If None, uses current env.data.qpos[env.rev_joint_idxs].
        is_render: if True, renders every few iterations during IK
        VERBOSE: if True, prints error norms
        reset_env:
            - True  -> calls env.reset() at the end (useful for offline planning like get_q_from_ik)
            - False -> leaves env at the IK solution (useful for online control, e.g. VLA loop)

    Returns:
        q: (n_rev_joints,) numpy array of joint positions for env.rev_joint_idxs.
    """
    # Initial configuration
    if curr_q is None:
        q = env.data.qpos[env.rev_joint_idxs].copy()
    else:
        q = np.array(curr_q, dtype=float).copy()

    # Just to be safe, sync FK with current q
    env.data.qpos[env.rev_joint_idxs] = q
    env.forward()

    err_eps = 1e-2

    for it in range(max_tick):
        # Jacobian for this body at current state
        J_p, J_R, J_full = get_J_body(env.model, env.data, body_name,
                                      rev_joint_idxs=env.rev_joint_idxs)

        # Current pose
        p_curr = env.data.body(body_name).xpos
        R_curr = env.data.body(body_name).xmat.reshape(3, 3)

        # Position error
        p_err = (p_trgt - p_curr)

        # Orientation error
        R_err = np.linalg.solve(R_curr, R_trgt)  # R_curr^{-1} * R_trgt
        w_err = R_curr @ r2w(R_err)              # rotation error vector in world frame

        # Full 6D error
        err = np.concatenate((p_err, w_err))
        err_norm = np.linalg.norm(err)

        if VERBOSE:
            print(f"[IK] iter {it:03d}  err_norm: {err_norm:.6f}")

        # Convergence check
        if err_norm < err_eps:
            if VERBOSE:
                print("[IK] Converged.")
            break

        # Damped least squares step
        J = J_full
        eps = 1e-1
        A = J.T @ J + eps * np.eye(J.shape[1])
        b = J.T @ err
        dq = np.linalg.solve(A, b)

        # Limit step size to avoid large jumps
        dq = trim_scale(dq, th=5.0 * np.pi / 180.0)  # 5 degrees in radians

        # Update joint positions and FK
        q = q + dq
        env.data.qpos[env.rev_joint_idxs] = q
        env.forward()

        # Optional debug rendering
        if is_render and (it % 5 == 0):
            env.render()

    # For offline planning, restore env to initial state
    if reset_env:
        env.reset()

    return q
