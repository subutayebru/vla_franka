#!/usr/bin/env python
# run_libero_mac.py
#
# Run an OpenVLA LIBERO-finetuned checkpoint in the LIBERO MuJoCo-Franka
# simulator on Apple Silicon (MPS). Self-contained: loads the checkpoint the
# same fp16/eager/MPS way as the base-model port (no bitsandbytes, no prismatic
# package, no TensorFlow) and reimplements the official image preprocessing in
# PIL/NumPy. See docs/LIBERO_EVAL.md.
#
# IMPORTANT: run with the dedicated LIBERO venv, not the main one:
#     .venv_libero/bin/python run_libero_mac.py --task_id 0
#
# Examples:
#     # list the built-in tasks (and their prompts) in a suite
#     .venv_libero/bin/python run_libero_mac.py --list
#     # run task 0 with its built-in instruction
#     .venv_libero/bin/python run_libero_mac.py --task_id 0
#     # override the instruction (change the prompting)
#     .venv_libero/bin/python run_libero_mac.py --task_id 0 --prompt "put the black bowl on the plate"
#     # run 3 episodes and save a video
#     .venv_libero/bin/python run_libero_mac.py --task_id 0 --episodes 3 --video

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.pop("MUJOCO_GL", None)  # macOS uses CGL for offscreen; EGL is Linux-only

import argparse
import time
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig

# OpenVLA's trust_remote_code modeling predates a flag newer transformers checks.
if not hasattr(transformers.modeling_utils.PreTrainedModel, "_supports_sdpa"):
    transformers.modeling_utils.PreTrainedModel._supports_sdpa = False

LANCZOS = Image.Resampling.LANCZOS


class LiveWatch:
    """Live matplotlib window of the scene (agentview), updated each step.
    Runs on the main thread with the default interactive macOS backend, so use
    plain `python` (not mjpython). Inference is ~2.5s/step on MPS, so the window
    refreshes once per step rather than at video speed."""
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.ax = self.im = None

    def update(self, frame, title):
        if self.fig is None:
            self.plt.ion()
            self.fig, self.ax = self.plt.subplots(num="LIBERO live — what OpenVLA sees",
                                                  figsize=(6, 6))
            self.im = self.ax.imshow(frame); self.ax.axis("off")
            self.plt.show(block=False)
        else:
            self.im.set_data(frame)
        self.ax.set_title(title, fontsize=10)
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()
        self.plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            self.plt.close(self.fig)

# suite -> (default finetuned checkpoint, max env steps)
SUITES = {
    "libero_spatial": ("openvla/openvla-7b-finetuned-libero-spatial", 220),
    "libero_object":  ("openvla/openvla-7b-finetuned-libero-object", 280),
    "libero_goal":    ("openvla/openvla-7b-finetuned-libero-goal", 300),
    "libero_10":      ("openvla/openvla-7b-finetuned-libero-10", 520),
}


def parse_args():
    ap = argparse.ArgumentParser(description="Run OpenVLA LIBERO eval on Apple Silicon (MPS).")
    ap.add_argument("--suite", default="libero_spatial", choices=list(SUITES.keys()))
    ap.add_argument("--task_id", type=int, default=0, help="Task index within the suite.")
    ap.add_argument("--prompt", default=None,
                    help="Override the instruction text (default: the task's built-in language).")
    ap.add_argument("--checkpoint", default=None,
                    help="HF checkpoint (default: the finetuned checkpoint matching --suite).")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None, help="Override per-suite default.")
    ap.add_argument("--num_steps_wait", type=int, default=10, help="Settle steps before acting.")
    ap.add_argument("--center_crop", action="store_true", default=True)
    ap.add_argument("--no_center_crop", dest="center_crop", action="store_false")
    ap.add_argument("--watch", action="store_true",
                    help="Open a live window showing the scene as the model acts (updates each step).")
    ap.add_argument("--video", action="store_true", help="Save an MP4 of each episode.")
    ap.add_argument("--video_path", default="libero_rollout.mp4")
    ap.add_argument("--render", type=int, default=512, help="Sim render size (model still gets 224).")
    ap.add_argument("--list", action="store_true", help="List tasks+prompts in the suite and exit.")
    return ap.parse_args()


def main():
    args = parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()

    if args.list:
        print(f"Suite '{args.suite}' has {suite.n_tasks} tasks:")
        for i in range(suite.n_tasks):
            print(f"  [{i:2d}] {suite.get_task(i).language}")
        return

    checkpoint = args.checkpoint or SUITES[args.suite][0]
    max_steps = args.max_steps or SUITES[args.suite][1]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    unnorm_key = args.suite

    task = suite.get_task(args.task_id)
    instruction = args.prompt if args.prompt is not None else task.language
    print(f"device={device}  checkpoint={checkpoint}")
    print(f"suite={args.suite}  task_id={args.task_id}")
    print(f"task language : {task.language!r}")
    print(f"PROMPT used   : {instruction!r}"
          f"{'  (overridden)' if args.prompt is not None else '  (built-in)'}")

    # ---- load checkpoint (fp16 on MPS, eager attention; no bitsandbytes) ----
    print("loading model (first run downloads ~14 GB)...", flush=True)
    t0 = time.time()
    mcfg = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    for a in ("attn_implementation", "_attn_implementation"):
        try:
            setattr(mcfg, a, "eager")
        except Exception:
            pass
    proc = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint, config=mcfg, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(device).eval()
    img_dtype = next(model.vision_backbone.parameters()).dtype
    print(f"loaded in {time.time()-t0:.0f}s  (norm_stats keys: {list(getattr(model,'norm_stats',{}).keys())})")

    def preprocess(obs):
        img = obs["agentview_image"][::-1, ::-1]                    # 180-rotate to match training
        im = Image.fromarray(img).convert("RGB").resize((224, 224), LANCZOS)
        if args.center_crop:
            s = 0.9 ** 0.5; nw = int(round(224 * s)); off = (224 - nw) // 2
            im = im.crop((off, off, off + nw, off + nw)).resize((224, 224), LANCZOS)
        return im

    def act(im):
        inputs = proc(f"In: What action should the robot take to {instruction.lower()}?\nOut:", im)
        inputs.pop("attention_mask", None)                         # avoid eager off-by-one
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device, dtype=img_dtype) if k == "pixel_values" else v.to(device)
        with torch.inference_mode():
            a = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        return np.asarray(a, dtype=float).reshape(-1)

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.render, camera_widths=args.render)
    init_states = suite.get_task_init_states(args.task_id)

    writer = None
    if args.video:
        import imageio
        writer = imageio.get_writer(args.video_path, fps=20)
    live = LiveWatch() if args.watch else None

    successes = 0
    for ep in range(args.episodes):
        env.reset()
        obs = env.set_init_state(init_states[ep % len(init_states)])
        done, t, grip_state = False, 0, "open"
        while t < max_steps + args.num_steps_wait:
            frame = obs["agentview_image"][::-1, ::-1]
            if writer is not None:
                writer.append_data(frame)
            if live is not None:
                step = max(t - args.num_steps_wait, 0)
                phase = "settling" if t < args.num_steps_wait else f"grip {grip_state}"
                live.update(frame, f"ep{ep}  step {step}/{max_steps}  |  {phase}")
            if t < args.num_steps_wait:
                obs, r, done, info = env.step([0, 0, 0, 0, 0, 0, -1]); t += 1; continue
            a = act(preprocess(obs))
            a[-1] = -np.sign(2 * a[-1] - 1)                        # gripper: normalize+binarize+invert
            grip_state = "CLOSE" if a[-1] > 0 else "open"
            obs, r, done, info = env.step(a.tolist())
            if done:
                successes += 1
                if writer is not None:
                    for _ in range(20):
                        writer.append_data(obs["agentview_image"][::-1, ::-1])
                break
            t += 1
            if t % 20 == 0:
                print(f"  ep{ep} t={t-args.num_steps_wait}/{max_steps}", flush=True)
        print(f"episode {ep}: success={done}", flush=True)

    if writer is not None:
        writer.close()
        print(f"video saved: {args.video_path}")
    if live is not None:
        live.close()
    env.close()
    print(f"\n==== {successes}/{args.episodes} success on '{instruction}' ====")


if __name__ == "__main__":
    main()
