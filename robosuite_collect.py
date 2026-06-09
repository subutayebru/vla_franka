#!/usr/bin/env python
# robosuite_collect.py
#
# Collect instruction-conditioned demos for YOUR task — a Franka picking a
# YELLOW or BLUE cube — in robosuite (reliable OSC controller), to fine-tune
# OpenVLA on. Built on robosuite's two-cube `Stack` env: we recolor the cubes
# yellow/blue (they render from geom_rgba, matid=-1) and script a closed-loop
# OSC expert to pick the cube named in the instruction.
#
# Both cubes are present every episode and the instruction names the colour, so
# the policy must read the instruction to disambiguate. robosuite's placement
# sampler randomizes cube positions each reset.
#
# Run with the LIBERO venv (has robosuite):
#   .venv_libero/bin/python robosuite_collect.py --episodes 1 --debug_video /tmp/rs_expert.mp4

import os
os.environ.pop("MUJOCO_GL", None)
import argparse
import numpy as np
import robosuite as rs
from robosuite.controllers import load_controller_config

POS_SCALE = 0.05          # OSC_POSE position output_max (m per unit action)
HOVER = 0.10
GRASP_DZ = 0.005          # eef z relative to cube z at grasp
LIFT_Z = 1.05             # fixed world height to lift to
XY_TOL = 0.012
CLOSE_HOLD = 12
GRIP_CLOSE, GRIP_OPEN = 1.0, -1.0
MAX_STEPS = 150
LIFT_SUCCESS_DZ = 0.06    # cube must rise this much to count as picked

YELLOW = [1.0, 1.0, 0.0, 1.0]
BLUE = [0.05, 0.2, 1.0, 1.0]
COLOR_TO_CUBE = {"yellow": "cubeA", "blue": "cubeB"}


def recolor(env):
    m = env.sim.model
    for g, rgba in [("cubeA_g0", YELLOW), ("cubeA_g0_vis", YELLOW),
                    ("cubeB_g0", BLUE), ("cubeB_g0_vis", BLUE)]:
        try:
            m.geom_rgba[m.geom_name2id(g)] = rgba
        except Exception:
            pass


def servo(target_xyz, eef, grip):
    d = np.clip((np.asarray(target_xyz) - np.asarray(eef)) / POS_SCALE, -1, 1)
    return np.array([d[0], d[1], d[2], 0, 0, 0, grip], dtype=np.float32)


def run_episode(env, color, debug_writer=None, record=True):
    obs = env.reset()
    recolor(env)
    # reset() already rendered with the old colors; re-render so the FIRST frame
    # (and every frame) shows the recolored yellow/blue cubes.
    env.sim.forward()
    obs = env._get_observations(force_update=True)
    cube = COLOR_TO_CUBE[color]
    instruction = f"pick up the {color} cube"
    start_z = obs[f"{cube}_pos"][2]

    images, actions = [], []
    phase, grip, ctr = "hover", GRIP_OPEN, 0
    for t in range(MAX_STEPS):
        eef = obs["robot0_eef_pos"]; c = obs[f"{cube}_pos"]
        if phase == "hover":
            act = servo([c[0], c[1], c[2] + HOVER], eef, GRIP_OPEN)
            if abs(eef[0]-c[0]) < XY_TOL and abs(eef[1]-c[1]) < XY_TOL:
                phase = "descend"
        elif phase == "descend":
            act = servo([c[0], c[1], c[2] + GRASP_DZ], eef, GRIP_OPEN)
            if eef[2] < c[2] + GRASP_DZ + 0.015:
                phase, ctr = "close", 0
        elif phase == "close":
            act = servo([c[0], c[1], c[2] + GRASP_DZ], eef, GRIP_CLOSE)
            ctr += 1
            if ctr > CLOSE_HOLD:
                phase = "lift"
        else:  # lift
            act = servo([eef[0], eef[1], LIFT_Z], eef, GRIP_CLOSE)

        if record:
            images.append(obs["agentview_image"][::-1].copy())  # flip vertical (robosuite is upside-down)
            actions.append(act.copy())
        if debug_writer is not None:
            debug_writer.append_data(obs["agentview_image"][::-1])

        obs, r, done, info = env.step(act.tolist())

    lifted = obs[f"{cube}_pos"][2] - start_z
    success = bool(lifted > LIFT_SUCCESS_DZ)
    return success, instruction, np.array(images), np.array(actions, dtype=np.float32), lifted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default="rs_demos")
    ap.add_argument("--debug_video", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_controller_config(default_controller="OSC_POSE")
    env = rs.make("Stack", robots="Panda", controller_configs=cfg, has_renderer=False,
                  has_offscreen_renderer=True, use_camera_obs=True, camera_names="agentview",
                  camera_heights=256, camera_widths=256, control_freq=20)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    writer = None
    if args.debug_video:
        import imageio
        writer = imageio.get_writer(args.debug_video, fps=20)

    colors = ["yellow", "blue"]
    succ = saved = 0
    for ep in range(args.episodes):
        w = writer if ep == 0 else None
        ok, instr, imgs, acts, lifted = run_episode(env, colors[ep % 2], debug_writer=w)
        succ += int(ok)
        print(f"ep {ep}: color={colors[ep%2]:6s} success={ok} lifted={lifted:.3f} frames={len(imgs)} instr={instr!r}", flush=True)
        if ok:
            np.savez_compressed(os.path.join(args.out, f"ep{ep:04d}_{colors[ep%2]}.npz"),
                                images=imgs, actions=acts, instruction=instr)
            saved += 1
    if writer is not None:
        writer.close(); print(f"debug video: {args.debug_video}")
    env.close()
    print(f"\n{succ}/{args.episodes} success, saved {saved} to {args.out}/")


if __name__ == "__main__":
    main()
