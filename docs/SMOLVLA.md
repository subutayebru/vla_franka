# Fine-tuning SmolVLA on our yellow/blue cube task

We switched the model from OpenVLA-7B to **SmolVLA (~0.45B, LeRobot)** — far
lighter, fine-tunes without QLoRA, and runs much faster on the Mac. Same task
and the same robosuite demos; only the data format, trainer, and deploy change.

## Environment

```bash
python3.10 -m venv .venv_lerobot
.venv_lerobot/bin/pip install "lerobot[smolvla]"     # pulls transformers + SmolVLM backbone
```
(LeRobot 0.4.4; module paths are `lerobot.datasets`, `lerobot.policies.smolvla`,
`lerobot.scripts.lerobot_train`.)

## 1. Data (already done on the Mac)

`robosuite_collect.py` records `(agentview image, 7-D OSC action, 9-D state
[eef pos+quat+gripper], instruction)` — 196 demos in `rs_demos/`. Convert to a
LeRobotDataset:

```bash
.venv_lerobot/bin/python rs_to_lerobot.py --root lerobot_data/rs_pick
```
Notes:
- We store images as **`image` (PNG), not `video`** — avoids the
  torchcodec/ffmpeg version mismatch on macOS (reads fine on Mac and the GPU box).
- Output: `lerobot_data/rs_pick/` (~1.6 GB), features
  `observation.images.agentview`, `observation.state` (9), `action` (7), `task`.

## 2. Fine-tune SmolVLA

SmolVLA (450M) fits easily — **no QLoRA needed**. It runs on the **24 GB GPU**
comfortably, and is small enough to try on the **M5 (MPS)** too (slower, free).

```bash
.venv_lerobot/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=rs_pick \
  --dataset.root=lerobot_data/rs_pick \
  --batch_size=64 \
  --steps=20000 \
  --save_freq=5000 \
  --output_dir=outputs/smolvla_rs_pick \
  --job_name=smolvla_rs_pick \
  --policy.device=cuda \
  --wandb.enable=false
```
- `--policy.path=lerobot/smolvla_base` loads the pretrained SmolVLA (config +
  weights) and fine-tunes it.
- **On the Mac instead:** `--policy.device=mps --batch_size=8` (slower; fine for a
  small dataset).
- Defaults freeze the vision encoder and train the action expert
  (`train_expert_only=True`) — good for small datasets.
- Checkpoints land in `outputs/smolvla_rs_pick/checkpoints/...`.

## 3. Deploy on the Mac (robosuite)

```bash
.venv_lerobot/bin/python smolvla_eval_mac.py \
  --checkpoint outputs/smolvla_rs_pick/checkpoints/last/pretrained_model \
  --color yellow --watch
```
`smolvla_eval_mac.py` rebuilds the recolored Stack env and runs the lerobot
inference pipeline: `preprocessor(obs) → policy.select_action → postprocessor`
(the preprocessor tokenizes the `task` string and normalizes; SmolVLA resizes the
image to 224 internally). **Validate this script against your first checkpoint** —
the exact observation key formatting is the one part not yet run end-to-end.

## Why SmolVLA over OpenVLA-7B here
- Fine-tunes without QLoRA gymnastics; fits the 24 GB GPU with room to spare.
- ~15× smaller ⇒ much faster inference on the M5.
- Native language conditioning, so "pick up the {yellow|blue} cube" works.
- Possible to train **without renting a GPU** (M5/MPS), if you accept slower steps.
