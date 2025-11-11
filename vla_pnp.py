# vla_pnp.py
import os, csv, time
os.environ.pop("MUJOCO_GL", None)  # keep GUI visible

import enum
import mujoco
import numpy as np
import matplotlib.pyplot as plt

from src.mujoco_parser import MuJoCoParserClass
from src.PID import PID_ControllerClass
from ik_single import get_q_from_ik_single
from vla_openvla import OpenVLA

# ======= toggles / knobs =======
ABLATE_VLA    = False   # True -> replace VLA with zeros (ablation)
PHASE_AGNOSTIC = False  # True -> single instruction, no FSM (for ablation/demo)
SHOW_VLA_FEED = True    # live window (Matplotlib) of what the VLA sees
OVERLAY_HUD   = True    # draw phase/action/pose text on that image
LOG_CSV       = f"/home/es_admin/vla-franka/Simple-MuJoCo-PickNPlace/vla_logs/vla_logs_{int(time.time())}.csv"

# ======= model element names (from your dump) =======
EE_SITE  = "grip_site"
EEF_BODY = "panda_eef"
OBJ_BODY = "obj_box_06"
PLT_BODY = "object table"

# ======= camera =======
CAMERA_NAME = "panda_eye_in_hand"  # or: 'panda_robotview', 'panda_eye_in_hand', 'standing_cam'

# ======= workspace =======
WORKSPACE = {"x": (0.25, 0.75), "y": (-0.35, 0.35), "z": (0.02, 0.55)}

def clamp_workspace(p):
    return np.array([
        np.clip(p[0], *WORKSPACE["x"]),
        np.clip(p[1], *WORKSPACE["y"]),
        np.clip(p[2], *WORKSPACE["z"]),
    ], dtype=np.float32)

# ======= gripper helper =======
def set_gripper(qpos: np.ndarray, option="open") -> np.ndarray:
    out = qpos.copy()
    if option == "open":
        out[7], out[8] = np.pi, -np.pi
    else:  # "close"
        out[7], out[8] = 0.0, 0.0
    return out

# ======= smoothing & small helpers =======
_last_a7 = None
def smooth(a7, alpha=0.5):
    global _last_a7
    a7 = a7.astype(np.float32).reshape(-1)
    if _last_a7 is None:
        _last_a7 = a7
        return _last_a7
    _last_a7 = alpha * a7 + (1.0 - alpha) * _last_a7
    return _last_a7

def deadband3(v, thresh=1e-4):
    # zero very tiny 3D motions (before scaling to world)
    return v if (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) >= thresh else np.zeros_like(v)

def smooth_phase(a7, alpha_move=0.5, alpha_fine=0.8, fine=False):
    # more smoothing during fine alignment near targets
    return smooth(a7, alpha=(alpha_fine if fine else alpha_move))

def blend_toward_planar_goal(p_ee, p_goal, vla_xy, max_push=0.02, radius=0.15):
    """
    Blend the VLA planar (world XY) step toward the goal as we get close.
    - p_ee:   current EE world position (3,)
    - p_goal: target world position (3,) but only XY used
    - vla_xy: current world XY step (2,) (meters per control step)
    """
    e = (p_goal[:2] - p_ee[:2])
    dist = np.linalg.norm(e)
    if dist < 1e-8:
        return vla_xy
    w = np.clip((radius - dist) / radius, 0.0, 1.0)
    push = (e / dist) * (w * max_push)  # up to max_push m/step when near
    return vla_xy + push


def mask_cube_magenta(env, renderer, cam_id, body_id, rgba=(1.0,0.0,1.0,1.0)):
    m = env.model; d = env.data
    geom_ids = np.where(m.geom_bodyid == body_id)[0]
    if geom_ids.size == 0:
        return None, None
    orig = m.geom_rgba.copy()
    for gid in geom_ids:
        m.geom_rgba[gid,:] = rgba
    renderer.update_scene(d, camera=cam_id)
    rgb = renderer.render()
    # revert
    for gid in geom_ids:
        m.geom_rgba[gid,:] = orig[gid,:]
    return rgb, orig

def flip_h(rgb):
    return rgb[:, ::-1].copy()

def action_from(vla, proc_rgb, instr):
    # single call wrapper just for probes
    return vla.predict(proc_rgb, instr).astype(np.float32)



# ======= classical IK servo (fallback) =======
def ik_servo_to(env, target_pos, target_quat, rate=0.25):
    """
    Take a soft step toward target pose via IK, with guards + joint rate limit.
    'rate' is a [0..1] blend toward IK solution.
    """
    q_now = env.data.qpos.copy()
    q_ik = get_q_from_ik_single(env, target_pos, target_quat, q_init=q_now,
                                site_name=EE_SITE, arm_joint_names=None)
    if (q_ik is None) or (not np.isfinite(q_ik).all()):
        return q_now
    # soften and rate-limit on controlled joints
    q_des = (1.0 - rate) * q_now + rate * q_ik
    max_step = 0.05
    dq = q_des[env.ctrl_joint_idxs] - q_now[env.ctrl_joint_idxs]
    peak = float(np.max(np.abs(dq))) if dq.size else 0.0
    if peak > max_step:
        q_des[env.ctrl_joint_idxs] = q_now[env.ctrl_joint_idxs] + dq * (max_step / peak)
    return q_des

# ======= VLA mapping (returns q_des, a7, rgb for logging/vis) =======
def vla_to_desired_q(env, renderer, vla, instruction,
                     ee_sid, eef_body_id, cam_id,
                     step_scale_xyz=0.01, step_scale_rpy=0.05):
    """
    VLA action (EE-frame deltas) -> world target -> IK -> desired qpos.
    Includes phase-aware smoothing, planar goal blend, world-vertical Z in fine mode,
    and a joint-rate limiter.
    Returns: q_des (qpos), a7 (smoothed), rgb (HxWx3 uint8)
    """
    # 1) RGB (from the chosen camera)
    renderer.update_scene(env.data, camera=cam_id)
    rgb = renderer.render()

    # 2) VLA inference (or ablation)
    if ABLATE_VLA:
        a7_raw = np.array([0,0,0, 0,0,0, -1], dtype=np.float32)
    else:
        a7_raw = vla.predict(rgb, instruction)
        if a7_raw.shape[0] < 7 or not np.isfinite(a7_raw).all():
            a7_raw = np.zeros(7, dtype=np.float32)

    fine_mode = getattr(env, "_fine_mode", False)
    a7 = smooth_phase(a7_raw, fine=fine_mode)

    # scale to meters/radians per step
    dxyz = np.array([a7[0], a7[1], a7[2]], dtype=np.float32) * step_scale_xyz
    dxyz = deadband3(dxyz, thresh=5e-6)  # ~2mm^2 norm deadband after scaling
    dx_ee, dy_ee, dz_ee = dxyz.tolist()
    droll, dpitch, dyaw = a7[3]*step_scale_rpy, a7[4]*step_scale_rpy, a7[5]*step_scale_rpy
    grip = a7[6]

    # Hold pose if nothing to do (but still update gripper)
    if (dx_ee*dx_ee + dy_ee*dy_ee + dz_ee*dz_ee) < 1e-10 and abs(dyaw) < 1e-3:
        q_hold = env.data.qpos.copy()
        q_hold = set_gripper(q_hold, "close" if grip > 0 else "open")
        return q_hold, a7, rgb

    # 3) Current EE pose (world)
    p_curr = env.data.site_xpos[ee_sid].copy()
    R_ee   = env.data.site_xmat[ee_sid].reshape(3,3).copy()
    q_curr = env.data.xquat[eef_body_id].copy()

    # 3.5) Optional planar blend toward object/platform near the goal
    goal_xy = None
    if getattr(env, "_mode", "") == "APPROACH_OBJ":
        goal_xy = env.data.xpos[getattr(env, "_obj_body_id")][:2]
    elif getattr(env, "_mode", "") == "APPROACH_PLT":
        goal_xy = env.data.xpos[getattr(env, "_plt_body_id")][:2]

    if goal_xy is not None:
        # convert EE-frame xy step to world, blend, then project back to EE frame
        ee_xy_local = np.array([dx_ee, dy_ee, 0.0], dtype=np.float32)
        step_xy_world = (R_ee @ ee_xy_local)[:2]
        goal3 = np.array([goal_xy[0], goal_xy[1], p_curr[2]], dtype=np.float32)
        step_xy_world = blend_toward_planar_goal(p_curr, goal3, step_xy_world,
                                                 max_push=0.02, radius=0.15)
        ee_xy_world3 = np.array([step_xy_world[0], step_xy_world[1], 0.0], dtype=np.float32)
        ee_xy_local2 = R_ee.T @ ee_xy_world3
        dx_ee, dy_ee = float(ee_xy_local2[0]), float(ee_xy_local2[1])

    # 4) EE-frame delta -> world delta
    # NOTE: If "down goes up" in your setup, flip the sign on dz_ee below.
    dpos_world = R_ee @ np.array([dx_ee, dy_ee, -dz_ee], dtype=np.float32)

    # During fine alignment, make Z strictly vertical (world) to avoid spirals
    if fine_mode:
        dpos_world[2] = -abs(dpos_world[2])  # only go down in fine phases

    target_pos  = clamp_workspace(p_curr + dpos_world)
    target_quat = q_curr  # keep orientation constant (add yaw later if desired)

    # 5) IK with guards + joint-rate limit
    q_now = env.data.qpos.copy()
    q_ik = get_q_from_ik_single(env, target_pos, target_quat,
                                q_init=q_now, site_name=EE_SITE, arm_joint_names=None)
    if (q_ik is None) or (not np.isfinite(q_ik).all()) \
       or np.any(np.abs(q_ik[env.ctrl_joint_idxs] - q_now[env.ctrl_joint_idxs]) > 0.25):
        q_des = q_now  # hold if IK bad or too big a jump
    else:
        beta = 0.2  # soften toward IK
        q_des = (1.0 - beta) * q_now + beta * q_ik

    # joint-rate limiting (safety)
    max_step = 0.05  # rad per control step on controlled joints
    qd_now = q_now[env.ctrl_joint_idxs]
    qd_des = q_des[env.ctrl_joint_idxs]
    dq = qd_des - qd_now
    peak = float(np.max(np.abs(dq))) if dq.size else 0.0
    if peak > max_step:
        qd_des = qd_now + dq * (max_step / peak)
        q_des[env.ctrl_joint_idxs] = qd_des

    # 6) Gripper from model
    q_des = set_gripper(q_des, "close" if grip > 0 else "open")
    return q_des, a7, rgb

# ======= FSM =======
class Phase(enum.Enum):
    SEARCH=0; APPROACH=1; GRASP=2; LIFT=3; MOVE_TO_PLACE=4; RELEASE=5; RETREAT=6
    RUN=7  # used only when PHASE_AGNOSTIC=True

def main():
    # Scene
    env = MuJoCoParserClass(
        name="Panda",
        rel_xml_path="asset/panda/franka_panda_w_objs.xml",
        VERBOSE=False
    )
    env.forward()
    env.init_viewer(viewer_title="PNP + VLA", viewer_width=1600, viewer_height=900, viewer_hide_menus=False)
    print("=== Cameras ===")
    for i in range(env.model.ncam):
        print(i, mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, i))

    # lock GUI + VLA to the same camera
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    assert cam_id != -1, f"Camera '{CAMERA_NAME}' not found; try 'standing_cam' or 'panda_eye_in'."
    env.update_viewer(cam_id=cam_id)
    env.reset()

    # Renderer (same model/data as GUI)
    renderer = mujoco.Renderer(env.model, height=480, width=640)

    # IDs once
    ee_sid      = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    eef_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, EEF_BODY)
    obj_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, OBJ_BODY)
    plt_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, PLT_BODY)
    assert ee_sid!=-1 and eef_body_id!=-1 and obj_body_id!=-1 and plt_body_id!=-1, "Check names"

    # expose ids to vla_to_desired_q for goal blending
    env._obj_body_id = obj_body_id
    env._plt_body_id = plt_body_id
    env._mode = ""        # "APPROACH_OBJ" / "APPROACH_PLT" / ""
    env._fine_mode = False

    # PID (tame gains for moving targets)
    PID = PID_ControllerClass(
        name="PID", dim=env.n_ctrl,
        k_p=300.0, k_i=5.0, k_d=40.0,
        out_min=env.ctrl_ranges[env.ctrl_joint_idxs, 0],
        out_max=env.ctrl_ranges[env.ctrl_joint_idxs, 1],
        ANTIWU=True
    )
    PID.reset()

    # VLA
    vla = OpenVLA(device="cuda", debug=True)

    # FSM setup
    if PHASE_AGNOSTIC:
        phase = Phase.RUN
        timeouts = {}
    else:
        phase = Phase.SEARCH
        timeouts = {
            Phase.SEARCH:800, Phase.APPROACH:1200, Phase.GRASP:300,
            Phase.LIFT:800, Phase.MOVE_TO_PLACE:1200, Phase.RELEASE:300, Phase.RETREAT:600
        }

    t_enter = env.tick
    z_grasp, z_lift, z_place = 0.03, 0.20, 0.05

    desired_q = env.data.qpos.copy()
    max_tick = 200000

    # --- progress watchdog for fallback servo ---
    last_prog_check_tick = env.tick
    last_d_obj = None
    stalled_cycles = 0
    USE_FALLBACK = False

    # ===== logging =====
    log_f = open(LOG_CSV, "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["tick","time","phase",
                  "dx","dy","dz","droll","dpitch","dyaw","grip",
                  "ee_x","ee_y","ee_z",
                  "obj_x","obj_y","obj_z",
                  "plt_x","plt_y","plt_z"])
    # ====================

    # live VLA feed window (Matplotlib)
    a7 = np.array([0,0,0,0,0,0,-1], dtype=np.float32)  # default status
    renderer.update_scene(env.data, camera=cam_id)
    vla_rgb = renderer.render()
    if SHOW_VLA_FEED:
        plt.ion()
        fig, ax = plt.subplots(num="VLA feed", figsize=(6.4, 4.8))
        imshow_handle = ax.imshow(vla_rgb)
        ax.axis("off")
        fig.canvas.draw()
        fig.canvas.flush_events()

    SINGLE_INSTRUCTION = "Pick up the red cube and place it on the black platform"

    # ========== main control loop ==========
    while env.tick < max_tick:
        sim_t = env.get_sim_time()
        p_ee  = env.data.site_xpos[ee_sid].copy()
        p_obj = env.data.xpos[obj_body_id].copy()
        p_plt = env.data.xpos[plt_body_id].copy()

        if PHASE_AGNOSTIC:
            env._mode, env._fine_mode = "", False
            instr = SINGLE_INSTRUCTION
            if env.tick % 2 == 0:
                desired_q, a7, vla_rgb = vla_to_desired_q(
                    env, renderer, vla, instr,
                    ee_sid, eef_body_id, cam_id,
                    step_scale_xyz=0.01, step_scale_rpy=0.05
                )
        else:
            # ---- FSM ----
            instr = "Pick up the red cube and place it on top of the black platform"

            if phase == Phase.SEARCH:
                instr = "Move above the red cube"
                env._mode, env._fine_mode = "APPROACH_OBJ", False
                if np.linalg.norm(p_ee[:2] - p_obj[:2]) < 0.04 and p_ee[2] > z_lift-0.02:
                    phase, t_enter = Phase.APPROACH, env.tick

            elif phase == Phase.APPROACH:
                instr = "Descend to grasp height over the red cube"
                env._mode, env._fine_mode = "APPROACH_OBJ", True

                # small XY recentre while descending if far from cube
                if np.linalg.norm(p_ee[:2] - p_obj[:2]) > 0.03:
                    dir_xy = p_obj[:2] - p_ee[:2]
                    d = np.linalg.norm(dir_xy)
                    if d > 1e-6:
                        step_xy = (dir_xy / d) * 0.005  # 5 mm in XY
                        target_pos = clamp_workspace(np.array(
                            [p_ee[0]+step_xy[0], p_ee[1]+step_xy[1], p_ee[2]], dtype=np.float32))
                        desired_q = ik_servo_to(env, target_pos, env.data.xquat[eef_body_id].copy(), rate=0.25)

                if np.linalg.norm(p_ee[:2] - p_obj[:2]) < 0.02 and p_ee[2] <= z_grasp+0.01:
                    phase, t_enter = Phase.GRASP, env.tick

            elif phase == Phase.GRASP:
                env._mode, env._fine_mode = "", False
                desired_q = set_gripper(env.data.qpos.copy(), "close")
                if env.tick - t_enter > 120:
                    f1, f2 = env.data.qpos[7], env.data.qpos[8]
                    closed = (abs(f1) < 0.02 and abs(f2) < 0.02)
                    near_obj = (np.linalg.norm(p_ee[:2] - p_obj[:2]) < 0.03) and (p_ee[2] < (z_grasp + 0.02))
                    if near_obj and not closed:
                        phase, t_enter = Phase.LIFT, env.tick
                    else:
                        desired_q = set_gripper(env.data.qpos.copy(), "open")
                        phase, t_enter = Phase.APPROACH, env.tick

            elif phase == Phase.LIFT:
                instr = "Lift up"
                env._mode, env._fine_mode = "", False
                if p_ee[2] >= z_lift:
                    phase, t_enter = Phase.MOVE_TO_PLACE, env.tick

            elif phase == Phase.MOVE_TO_PLACE:
                instr = "Move above the black platform and descend"
                env._mode, env._fine_mode = "APPROACH_PLT", True
                if np.linalg.norm(p_ee[:2] - p_plt[:2]) < 0.03 and p_ee[2] <= z_place+0.02:
                    phase, t_enter = Phase.RELEASE, env.tick

            elif phase == Phase.RELEASE:
                env._mode, env._fine_mode = "", False
                desired_q = set_gripper(env.data.qpos.copy(), "open")
                if env.tick - t_enter > timeouts[Phase.RELEASE]:
                    phase, t_enter = Phase.RETREAT, env.tick

            elif phase == Phase.RETREAT:
                instr = "Move up to a safe height"
                env._mode, env._fine_mode = "", False
                if p_ee[2] >= z_lift:
                    print("[INFO] Task complete.")
                    break

            # --- VLA or fallback control during motion phases ---
            if phase not in (Phase.GRASP, Phase.RELEASE):
                if USE_FALLBACK:
                    # classical servo waypoints
                    target_quat = env.data.xquat[eef_body_id].copy()
                    if phase == Phase.SEARCH:
                        target_pos = np.array([p_obj[0], p_obj[1], max(p_ee[2], z_lift)], dtype=np.float32)
                    elif phase == Phase.APPROACH:
                        target_pos = np.array([p_obj[0], p_obj[1], z_grasp], dtype=np.float32)
                    elif phase == Phase.LIFT:
                        target_pos = np.array([p_ee[0], p_ee[1], z_lift], dtype=np.float32)
                    elif phase == Phase.MOVE_TO_PLACE:
                        if p_ee[2] > z_lift - 0.02:
                            target_pos = np.array([p_plt[0], p_plt[1], z_lift], dtype=np.float32)
                        else:
                            target_pos = np.array([p_plt[0], p_plt[1], z_place], dtype=np.float32)
                    else:
                        target_pos = p_ee.copy()

                    target_pos = clamp_workspace(target_pos)
                    desired_q = ik_servo_to(env, target_pos, target_quat, rate=0.25)

                    # for logging/HUD continuity when in fallback
                    a7 = np.array([0,0,0,0,0,0,-1], dtype=np.float32)
                    renderer.update_scene(env.data, camera=cam_id)
                    vla_rgb = renderer.render()

                elif env.tick % 2 == 0:
                    desired_q, a7, vla_rgb = vla_to_desired_q(
                        env, renderer, vla, instr,
                        ee_sid, eef_body_id, cam_id,
                        step_scale_xyz=0.01, step_scale_rpy=0.05
                    )

        # ---- PID & step ----
        q_trgt_ctrl = desired_q[env.ctrl_joint_idxs]
        q_curr_ctrl = env.get_q(joint_idxs=env.ctrl_joint_idxs)
        PID.update(x_trgt=q_trgt_ctrl, x_curr=q_curr_ctrl, t_curr=sim_t, VERBOSE=False)
        torque = PID.out()
        env.step(ctrl=torque, ctrl_idxs=env.ctrl_joint_idxs)

        # ---- live VLA feed (Matplotlib HUD) ----
        if SHOW_VLA_FEED:
            frame = vla_rgb
            if OVERLAY_HUD:
                # draw text onto title; (Matplotlib overlays are minimal)
                phase_name = ("RUN" if PHASE_AGNOSTIC else phase.name)
                dx = float(a7[0]) if a7 is not None else 0.0
                dy = float(a7[1]) if a7 is not None else 0.0
                dz = float(a7[2]) if a7 is not None else 0.0
                grip = float(a7[6]) if a7 is not None else -1.0
                ee_z = float(p_ee[2])

                imshow_handle.set_data(frame)
                ax.set_title(
                    f"phase: {phase_name}"
                    f" | dx={dx:+.2f} dy={dy:+.2f} dz={dz:+.2f} grip={grip:+.2f}"
                    f" | ee_z={ee_z:+.3f} m"
                    f"{'  [FALLBACK]' if USE_FALLBACK else ''}",
                    fontsize=9
                )
            else:
                imshow_handle.set_data(frame)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)  # keeps UI responsive

        # ---- render GUI at lower rate ----
        if (env.tick % 3) == 0:
            env.render()

        # ------- progress check every ~200 ticks -------
        if not PHASE_AGNOSTIC and (env.tick - last_prog_check_tick >= 200):
            d_obj_now = float(np.linalg.norm(p_ee[:2] - p_obj[:2]))
            if last_d_obj is None or d_obj_now < (last_d_obj - 0.01):
                stalled_cycles = 0  # improved by >1 cm
            else:
                stalled_cycles += 1
            last_d_obj = d_obj_now
            last_prog_check_tick = env.tick

            # engage fallback after 3 stalled checks
            USE_FALLBACK = (stalled_cycles >= 3)

        # auto-exit fallback when close to goals
        if USE_FALLBACK and not PHASE_AGNOSTIC:
            if phase in (Phase.SEARCH, Phase.APPROACH) and np.linalg.norm(p_ee[:2] - p_obj[:2]) < 0.05:
                USE_FALLBACK = False; stalled_cycles = 0
            if phase == Phase.MOVE_TO_PLACE and np.linalg.norm(p_ee[:2] - p_plt[:2]) < 0.05:
                USE_FALLBACK = False; stalled_cycles = 0

        # ---- watchdog to keep FSM moving ----
        if not PHASE_AGNOSTIC and (env.tick - t_enter > timeouts.get(phase, 1500)):
            print(f"[WARN] timeout in {phase.name}, advancing")
            phase = {
                Phase.SEARCH: Phase.APPROACH,
                Phase.APPROACH: Phase.GRASP,
                Phase.LIFT: Phase.MOVE_TO_PLACE,
                Phase.MOVE_TO_PLACE: Phase.RELEASE,
            }.get(phase, Phase.RETREAT)
            t_enter = env.tick

        # ---- periodic console prints & CSV log ----
        if env.tick % 40 == 0:
            d_obj = np.linalg.norm(p_ee[:2] - p_obj[:2])
            d_plt = np.linalg.norm(p_ee[:2] - p_plt[:2])
            phase_name = ("RUN" if PHASE_AGNOSTIC else phase.name)
            print(f"[{env.tick}] phase={phase_name} z={p_ee[2]:.3f} d_obj={d_obj:.3f} d_plt={d_plt:.3f}"
                  f"{' [FALLBACK]' if USE_FALLBACK else ''}")
            
            renderer.update_scene(env.data, camera=cam_id)
            rgb_dbg = renderer.render()
            a7_dbg = vla.predict(rgb_dbg, "Move above the cube.")
            print("[VLA a7 dbg]", a7_dbg)

        phase_name = ("RUN" if PHASE_AGNOSTIC else phase.name)
        dx,dy,dz,dr,dp,dyaw,grip = (a7.tolist() if a7 is not None else [0,0,0,0,0,0,-1])
        log.writerow([
            env.tick, sim_t, phase_name,
            dx,dy,dz,dr,dp,dyaw,grip,
            p_ee[0],p_ee[1],p_ee[2],
            p_obj[0],p_obj[1],p_obj[2],
            p_plt[0],p_plt[1],p_plt[2]
        ])

    # ===== clean up =====
    log_f.close()

    if SHOW_VLA_FEED:
        try:
            plt.ioff(); plt.close(fig)
        except Exception:
            pass

    env.close_viewer()
    print(f"Done. Log written to: {LOG_CSV}")

if __name__ == "__main__":
    main()
