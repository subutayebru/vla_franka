#!/usr/bin/env python
# libero_collect.py
#
# Collect instruction-conditioned demos in LIBERO with a scripted closed-loop
# OSC expert, to fine-tune OpenVLA on (train on a CUDA GPU later).
#
# Why this works where our own scene didn't: LIBERO uses robosuite's
# OperationalSpaceController, so we command end-effector deltas
# (action = clip((target - eef)/0.05, -1, 1)) and it tracks them precisely.
# libero_object has 10 tasks in the SAME scene where the instruction names which
# object to pick -> instruction-conditioned data for free.
#
# Run with the LIBERO venv:
#   .venv_libero/bin/python libero_collect.py --task_id 0 --episodes 1 \
#       --debug_video /tmp/libero_expert.mp4
#
# v1: grasp offsets/thresholds are tunable constants; expect to tune from video.

import os
os.environ.pop("MUJOCO_GL", None)
import argparse
import numpy as np

POS_SCALE = 0.05          # OSC position output_max (m per unit action)
HOVER = 0.12              # height above object to hover (m)
GRASP_DZ = 0.005          # eef z target relative to object z at grasp
LIFT_Z = 0.25             # FIXED world height to lift/carry at (not object-relative!)
RELEASE_Z = 0.18          # height over the basket to release at
XY_TOL = 0.015
Z_TOL = 0.02
CLOSE_HOLD = 15           # steps to hold the gripper closed before lifting
GRIP_CLOSE, GRIP_OPEN = 1.0, -1.0
MAX_STEPS = 400


def target_object_key(language, obs):
    # "pick up the alphabet soup and place it in the basket" -> "alphabet_soup_1"
    name = language.split("pick up the ")[1].split(" and place")[0].strip()
    key = name.replace(" ", "_") + "_1"
    assert f"{key}_pos" in obs, f"could not map '{name}' -> obs key (have: " \
        f"{[k for k in obs if k.endswith('_pos') and 'eef' not in k]})"
    return key


def servo(target_xyz, eef_xyz, grip):
    d = (np.asarray(target_xyz) - np.asarray(eef_xyz)) / POS_SCALE
    d = np.clip(d, -1.0, 1.0)
    return np.array([d[0], d[1], d[2], 0.0, 0.0, 0.0, grip], dtype=np.float32)


def run_episode(env, language, init_state=None, record=True, debug_writer=None):
    obs = env.reset()
    if init_state is not None:
        obs = env.set_init_state(init_state)
    okey = target_object_key(language, obs)

    images, actions = [], []
    phase = "hover"
    grip = GRIP_OPEN
    close_ctr = 0
    done = False

    for t in range(MAX_STEPS):
        eef = obs["robot0_eef_pos"]
        obj = obs[f"{okey}_pos"]
        basket = obs["basket_1_pos"]

        if phase == "hover":
            tgt = [obj[0], obj[1], obj[2] + HOVER]
            act = servo(tgt, eef, GRIP_OPEN)
            if abs(eef[0]-obj[0]) < XY_TOL and abs(eef[1]-obj[1]) < XY_TOL:
                phase = "descend"
        elif phase == "descend":
            tgt = [obj[0], obj[1], obj[2] + GRASP_DZ]
            act = servo(tgt, eef, GRIP_OPEN)
            if eef[2] < obj[2] + GRASP_DZ + Z_TOL:
                phase = "close"; close_ctr = 0
        elif phase == "close":
            act = servo([obj[0], obj[1], obj[2] + GRASP_DZ], eef, GRIP_CLOSE)
            close_ctr += 1
            if close_ctr > CLOSE_HOLD:
                phase = "lift"
        elif phase == "lift":
            # FIXED world height; once grasped the object moves with the gripper,
            # so never key the target off obj[2].
            tgt = [eef[0], eef[1], LIFT_Z]
            act = servo(tgt, eef, GRIP_CLOSE)
            if eef[2] > LIFT_Z - 0.03:
                phase = "to_basket"
        elif phase == "to_basket":
            tgt = [basket[0], basket[1], LIFT_Z]
            act = servo(tgt, eef, GRIP_CLOSE)
            if abs(eef[0]-basket[0]) < XY_TOL*2 and abs(eef[1]-basket[1]) < XY_TOL*2:
                phase = "release"
        else:  # release: lower a bit over basket, then open
            tgt = [basket[0], basket[1], RELEASE_Z]
            grip = GRIP_CLOSE if eef[2] > RELEASE_Z + Z_TOL else GRIP_OPEN
            act = servo(tgt, eef, grip)

        if record:
            images.append(obs["agentview_image"][::-1, ::-1].copy())
            actions.append(act.copy())
        if debug_writer is not None:
            debug_writer.append_data(obs["agentview_image"][::-1, ::-1])

        obs, r, done, info = env.step(act.tolist())
        if done:
            break

    return bool(done), np.array(images), np.array(actions, dtype=np.float32), phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated task ids to collect (overrides --task_id), e.g. 0,3,5,6,8")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default="libero_demos")
    ap.add_argument("--debug_video", default=None)
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [int(x) for x in args.tasks.split(",")] if args.tasks else [args.task_id]
    os.makedirs(args.out, exist_ok=True)

    grand_succ = grand_saved = grand_eps = 0
    for tid in task_ids:
        task = suite.get_task(tid)
        language = task.language
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(0)
        init_states = suite.get_task_init_states(tid)
        print(f"\n=== task {tid}: {language!r} (init_states={len(init_states)}) ===", flush=True)

        writer = None
        if args.debug_video and tid == task_ids[0]:
            import imageio
            writer = imageio.get_writer(args.debug_video, fps=20)

        succ = saved = 0
        for ep in range(args.episodes):
            w = writer if ep == 0 else None
            ok, imgs, acts, last_phase = run_episode(
                env, language, init_state=init_states[ep % len(init_states)], debug_writer=w)
            succ += int(ok)
            if ok:
                np.savez_compressed(
                    os.path.join(args.out, f"{args.suite}_t{tid}_ep{ep:04d}.npz"),
                    images=imgs, actions=acts, instruction=language)
                saved += 1
        if writer is not None:
            writer.close(); print(f"debug video: {args.debug_video}")
        env.close()
        print(f"task {tid}: {succ}/{args.episodes} success, saved {saved}", flush=True)
        grand_succ += succ; grand_saved += saved; grand_eps += args.episodes

    print(f"\nTOTAL: {grand_succ}/{grand_eps} success, saved {grand_saved} demos to {args.out}/")


if __name__ == "__main__":
    main()
