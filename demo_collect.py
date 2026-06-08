#!/usr/bin/env python
# demo_collect.py
#
# Collect instruction-conditioned pick-and-place demonstrations in our MuJoCo
# Franka scene, to fine-tune a VLA on (train on a CUDA GPU; collect here on Mac).
#
# Strategy (see docs / chat): BOTH cubes are present every episode and the
# instruction names the target colour ("pick up the {yellow|blue} cube and place
# it on the plate"), 50/50 across episodes, with randomized cube positions — so
# the policy must READ THE INSTRUCTION to disambiguate, not memorize one cube.
#
# A scripted expert (IK waypoints + PID) performs the task; we record, at the
# policy rate, (standing_cam image, EE-delta action, instruction). Actions use
# the SAME convention as core/control_utils.apply_action_to_pose so a model
# trained on them plugs straight into run_vla_control.py. Only successful
# episodes (named cube ends on the plate) are saved.
#
# Run with the main env venv:
#     .venv/bin/python demo_collect.py --episodes 1 --debug_video /tmp/demo_ep.mp4
#
# NOTE: this is v1 — the grasp heights/timing are tunable constants below and
# WILL need adjustment after watching the debug video.

import os
os.environ.pop("MUJOCO_GL", None)
import argparse
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

from core.env_wrapper import PandaEnv
from core.config import PIDCfg
from core.control_utils import create_pid
from core.ik_solver import solve_IK

CUBES = {
    "blue":   {"joint": "object_joint_06", "body": "obj_box_06"},
    "yellow": {"joint": "object_joint_07", "body": "obj_box_07"},
}
PLATE_BODY = "obj_box_black"          # flat black platform used as the "plate"
EEF = "panda_eef"

# Top-down grasp orientation (gripper pointing down), from get_grasp_pose_using_ik.
GRASP_R = np.array([
    [6.123e-17, 0.9848, 0.1736],
    [1.0, 0.0, 0.0],
    [0.0, 0.1736, -0.9848],
])

# --- tunable heights / timing (world Z); CUBES rest with centre ~1.13 ---
Z_HIGH = 1.40      # clearance / approach height
Z_GRASP = 1.13     # descend to cube centre to close around it
Z_PLACE = 1.18     # release height above the plate
SETTLE_STEPS = 150
WP_STEPS = 1200    # sim steps to hold each waypoint
POLICY_EVERY = 100 # record one (img, action) every N sim steps (sim 500Hz -> 5Hz)

GRIP_OPEN = np.array([np.pi, -np.pi])
GRIP_CLOSE = np.array([0.0, 0.0])

# Workspace box to randomize cube xy within (x, y ranges), kept reachable.
RAND_X = (0.55, 0.80)
RAND_Y = (-0.20, 0.20)


def _qadr(env, joint_name):
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return env.model.jnt_qposadr[jid]


def set_cube(env, joint_name, xyz):
    adr = _qadr(env, joint_name)
    env.data.qpos[adr:adr + 3] = xyz
    env.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]  # identity quat


def cube_xyz(env, body):
    return env.get_p_body(body).copy()


def rng_xy(rng):
    return np.array([rng.uniform(*RAND_X), rng.uniform(*RAND_Y)])


def ik(env, p, Rm, seed):
    return solve_IK(env, max_tick=500, p_trgt=p, R_trgt=Rm, body_name=EEF,
                    curr_q=seed, is_render=False, VERBOSE=False, reset_env=False)


def run_episode(panda, pid, rng, target, debug_writer=None):
    env = panda.env
    instruction = f"pick up the {target} cube and place it on the plate"

    # 1) reset + randomize both cubes (non-overlapping), let them settle.
    env.reset()
    pts = []
    for c in CUBES:
        while True:
            xy = rng_xy(rng)
            if all(np.linalg.norm(xy - p) > 0.12 for p in pts):
                pts.append(xy); break
        set_cube(env, CUBES[c]["joint"], [xy[0], xy[1], 1.13])
    env.forward()
    arm0 = env.get_q(joint_idxs=env.rev_joint_idxs)
    desired_q = np.concatenate([arm0, GRIP_OPEN])
    for _ in range(SETTLE_STEPS):
        _drive(panda, pid, desired_q)

    p_obj = cube_xyz(env, CUBES[target]["body"])
    p_plate = cube_xyz(env, PLATE_BODY)

    # 2) waypoints: (xyz, grip)
    wp = [
        ([p_obj[0], p_obj[1], Z_HIGH], GRIP_OPEN),     # above object
        ([p_obj[0], p_obj[1], Z_GRASP], GRIP_OPEN),    # descend
        ([p_obj[0], p_obj[1], Z_GRASP], GRIP_CLOSE),   # close
        ([p_obj[0], p_obj[1], Z_HIGH], GRIP_CLOSE),    # lift
        ([p_plate[0], p_plate[1], Z_HIGH], GRIP_CLOSE),# above plate
        ([p_plate[0], p_plate[1], Z_PLACE], GRIP_CLOSE),# descend over plate
        ([p_plate[0], p_plate[1], Z_PLACE], GRIP_OPEN),# release
        ([p_plate[0], p_plate[1], Z_HIGH], GRIP_OPEN), # retreat
    ]

    # 3) precompute IK joint targets (snapshot/restore so solving doesn't disturb sim)
    snap_q, snap_v = env.data.qpos.copy(), env.data.qvel.copy()
    seed = arm0
    targets = []
    for xyz, grip in wp:
        arm = ik(env, np.array(xyz), GRASP_R, seed)
        seed = arm
        targets.append((np.concatenate([arm, grip]), grip))
    env.data.qpos[:], env.data.qvel[:] = snap_q, snap_v
    env.forward()

    # 4) rollout, recording at policy rate
    rec = []  # list of (img(np uint8), ee_p, ee_R, grip_sign)
    tick = 0
    for desired_q, grip in targets:
        grip_sign = 1.0 if np.allclose(grip, GRIP_CLOSE) else -1.0
        for _ in range(WP_STEPS):
            if tick % POLICY_EVERY == 0:
                img = np.array(panda.get_image())
                ee_p = env.get_p_body(EEF).copy()
                ee_R = env.get_R_body(EEF).copy()
                rec.append((img, ee_p, ee_R, grip_sign))
                if debug_writer is not None:
                    debug_writer.append_data(img)
            _drive(panda, pid, desired_q)
            tick += 1

    # 5) success: target cube on the plate (xy within plate half-extent, settled height)
    pc = cube_xyz(env, CUBES[target]["body"])
    on_xy = abs(pc[0] - p_plate[0]) < 0.10 and abs(pc[1] - p_plate[1]) < 0.10
    on_z = 1.04 < pc[2] < 1.25
    success = bool(on_xy and on_z)

    # 6) convert recorded poses -> EE-delta actions (apply_action_to_pose convention)
    sx, sy = 0.5, 0.8  # step_scale_xyz, step_scale_rpy (configs/default.yaml)
    images, actions = [], []
    for i in range(len(rec) - 1):
        img, p, Rm, g = rec[i]
        p2, R2 = rec[i + 1][1], rec[i + 1][2]
        d_pos = (p2 - p) / sx
        d_R = Rm.T @ R2
        d_rpy = R.from_matrix(d_R).as_euler("xyz") / sy
        images.append(img)
        actions.append(np.concatenate([d_pos, d_rpy, [g]]).astype(np.float32))
    return success, instruction, np.array(images), np.array(actions)


def _drive(panda, pid, desired_q):
    env = panda.env
    pid.update(x_trgt=desired_q)
    pid.update(t_curr=env.get_sim_time(), x_curr=env.get_q(joint_idxs=env.ctrl_joint_idxs), VERBOSE=False)
    tau = pid.out() + env.data.qfrc_bias[env.ctrl_joint_idxs]
    panda.step(tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default="demos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug_video", default=None, help="Save an MP4 of the first episode for tuning.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    panda = PandaEnv(xml_path="asset/panda/franka_panda_w_objs.xml",
                     camera_name="standing_cam", show_view=False)
    pid = create_pid(panda.env, PIDCfg(kp=800.0, ki=20.0, kd=100.0))

    colors = ["yellow", "blue"]
    saved = succ = 0
    for ep in range(args.episodes):
        target = colors[ep % 2]
        writer = None
        if args.debug_video and ep == 0:
            import imageio
            writer = imageio.get_writer(args.debug_video, fps=20)
        ok, instr, imgs, acts = run_episode(panda, pid, rng, target, debug_writer=writer)
        if writer is not None:
            writer.close()
            print(f"debug video: {args.debug_video}")
        succ += int(ok)
        print(f"ep {ep}: target={target:6s} success={ok}  frames={len(imgs)}  instr={instr!r}", flush=True)
        if ok:
            path = os.path.join(args.out, f"ep_{ep:04d}_{target}.npz")
            np.savez_compressed(path, images=imgs, actions=acts, instruction=instr)
            saved += 1
    print(f"\nsaved {saved} successful / {args.episodes} episodes ({succ} succeeded) to {args.out}/")


if __name__ == "__main__":
    main()
