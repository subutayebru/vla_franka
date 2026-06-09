#!/usr/bin/env python
# robosuite_eval_mac.py
#
# Deploy a fine-tuned OpenVLA checkpoint on the YELLOW/BLUE cube task in robosuite,
# on Apple Silicon (MPS). Inference counterpart of robosuite_collect.py: same
# recolored Stack env, same agentview preprocessing, fp16/eager/MPS model load.
#
# Run after fine-tuning (see docs/FINETUNE.md), with the LIBERO venv:
#   .venv_libero/bin/python robosuite_eval_mac.py \
#       --checkpoint /path/to/finetuned_ckpt --color yellow --watch
#
# NOTE: keep the gripper convention consistent with collection — our dataset
# stores the raw robosuite action (gripper +1 close / -1 open), so we feed the
# model's action straight to env.step (NO LIBERO-style gripper inversion).

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.pop("MUJOCO_GL", None)
import argparse
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig

if not hasattr(transformers.modeling_utils.PreTrainedModel, "_supports_sdpa"):
    transformers.modeling_utils.PreTrainedModel._supports_sdpa = False

LANCZOS = Image.Resampling.LANCZOS
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


def preprocess(agentview):
    # match robosuite_collect: vertical flip; then OpenVLA-style 224 + center-crop 0.9
    im = Image.fromarray(agentview[::-1]).convert("RGB").resize((224, 224), LANCZOS)
    s = 0.9 ** 0.5; nw = int(round(224 * s)); off = (224 - nw) // 2
    return im.crop((off, off, off + nw, off + nw)).resize((224, 224), LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--color", choices=["yellow", "blue"], default="yellow")
    ap.add_argument("--unnorm_key", default="rs_pick", help="RLDS dataset name used in training")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=150)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    import robosuite as rs
    from robosuite.controllers import load_controller_config

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}  checkpoint={args.checkpoint}  color={args.color}")
    mcfg = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    for a in ("attn_implementation", "_attn_implementation"):
        try:
            setattr(mcfg, a, "eager")
        except Exception:
            pass
    proc = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.checkpoint, config=mcfg, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True).to(device).eval()
    img_dtype = next(model.vision_backbone.parameters()).dtype
    instruction = f"pick up the {args.color} cube"

    def act(im):
        inputs = proc(f"In: What action should the robot take to {instruction}?\nOut:", im)
        inputs.pop("attention_mask", None)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device, dtype=img_dtype) if k == "pixel_values" else v.to(device)
        with torch.inference_mode():
            a = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
        return np.asarray(a, dtype=float).reshape(-1)

    cfg = load_controller_config(default_controller="OSC_POSE")
    env = rs.make("Stack", robots="Panda", controller_configs=cfg, has_renderer=False,
                  has_offscreen_renderer=True, use_camera_obs=True, camera_names="agentview",
                  camera_heights=256, camera_widths=256, control_freq=20)
    cube = COLOR_TO_CUBE[args.color]

    live = None
    if args.watch:
        import matplotlib.pyplot as plt
        plt.ion(); fig, ax = plt.subplots(num="robosuite eval — what OpenVLA sees")
        imw = None
    writer = None
    if args.video:
        import imageio; writer = imageio.get_writer(args.video, fps=20)

    succ = 0
    for ep in range(args.episodes):
        obs = env.reset(); recolor(env); env.sim.forward()
        obs = env._get_observations(force_update=True)
        start_z = obs[f"{cube}_pos"][2]
        for t in range(args.max_steps):
            frame = obs["agentview_image"][::-1]
            if writer is not None:
                writer.append_data(frame)
            if live is not None:
                if imw is None:
                    imw = ax.imshow(frame); ax.axis("off")
                else:
                    imw.set_data(frame)
                ax.set_title(f"ep{ep} t={t}  '{instruction}'", fontsize=10)
                fig.canvas.draw_idle(); fig.canvas.flush_events(); plt.pause(0.001)
            a = act(preprocess(obs["agentview_image"]))
            obs, r, done, info = env.step(a.tolist())
        lifted = obs[f"{cube}_pos"][2] - start_z
        ok = lifted > 0.06
        succ += int(ok)
        print(f"ep {ep}: success={ok} lifted={lifted:.3f}", flush=True)
    if writer is not None:
        writer.close(); print(f"video: {args.video}")
    env.close()
    print(f"\n{succ}/{args.episodes} success")


if __name__ == "__main__":
    main()
