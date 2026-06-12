#!/usr/bin/env python
# smolvla_eval_mac.py
#
# Deploy a fine-tuned SmolVLA checkpoint on the YELLOW/BLUE cube task in robosuite,
# on Apple Silicon (MPS). Uses the LeRobot inference pipeline:
#   preprocessor(obs) -> policy.select_action -> postprocessor(action)
# The preprocessor tokenizes the `task` string and normalizes; SmolVLA resizes
# the image to 224 internally.
#
# Run with the LeRobot venv, after fine-tuning (see docs/SMOLVLA.md):
#   .venv_lerobot/bin/python smolvla_eval_mac.py \
#       --checkpoint outputs/smolvla_rs_pick/checkpoints/last/pretrained_model \
#       --color yellow --watch
#
# NOTE: validate against your first checkpoint — the observation key formatting is
# the one piece not yet run end-to-end on this Mac.

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.pop("MUJOCO_GL", None)
import argparse
import numpy as np
import torch

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--color", choices=["yellow", "blue"], default="yellow")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=150)
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()

    import robosuite as rs
    from robosuite.controllers import load_controller_config
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}  checkpoint={args.checkpoint}  color={args.color}")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)
    instruction = f"pick up the {args.color} cube"

    cfg = load_controller_config(default_controller="OSC_POSE")
    env = rs.make("Stack", robots="Panda", controller_configs=cfg, has_renderer=False,
                  has_offscreen_renderer=True, use_camera_obs=True, camera_names="agentview",
                  camera_heights=256, camera_widths=256, control_freq=20)
    cube = COLOR_TO_CUBE[args.color]

    live = None
    if args.watch:
        import matplotlib.pyplot as plt
        plt.ion(); fig, ax = plt.subplots(num="SmolVLA eval — what the policy sees"); imw = None

    succ = 0
    for ep in range(args.episodes):
        obs = env.reset(); recolor(env); env.sim.forward()
        obs = env._get_observations(force_update=True)
        start_z = obs[f"{cube}_pos"][2]
        policy.reset()
        for t in range(args.max_steps):
            frame = obs["agentview_image"][::-1]
            if live is not None:
                if imw is None:
                    imw = ax.imshow(frame); ax.axis("off")
                else:
                    imw.set_data(frame)
                ax.set_title(f"ep{ep} t={t}  '{instruction}'", fontsize=10)
                fig.canvas.draw_idle(); fig.canvas.flush_events(); plt.pause(0.001)

            # Build the lerobot observation: image as CHW float[0,1], state, task.
            img = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
            state = np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"],
                                    obs["robot0_gripper_qpos"]]).astype(np.float32)
            observation = {
                "observation.images.agentview": img.unsqueeze(0).to(device),
                "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
                "task": instruction,
            }
            observation = pre(observation)
            with torch.inference_mode():
                action = policy.select_action(observation)
            action = post(action)
            a = action.squeeze(0).float().cpu().numpy()[:7]
            obs, r, done, info = env.step(a.tolist())
        lifted = obs[f"{cube}_pos"][2] - start_z
        ok = lifted > 0.06
        succ += int(ok)
        print(f"ep {ep}: success={ok} lifted={lifted:.3f}", flush=True)

    env.close()
    print(f"\n{succ}/{args.episodes} success")


if __name__ == "__main__":
    main()
