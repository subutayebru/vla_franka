# Running OpenVLA on LIBERO (Apple Silicon) + the fine-tuning plan

This documents two things proven/decided in this work:
1. **Inference works on the Mac** — an OpenVLA LIBERO-finetuned checkpoint drives
   the LIBERO MuJoCo-Franka sim on M-series (MPS) and completes tasks.
2. **The plan to fine-tune on our own task** (train on a CUDA GPU, collect data
   and run inference on the Mac).

See also `docs/MACOS_PORT.md` (base-model MPS port) and
`docs/OPENVLA_PIPELINE.md` (how OpenVLA generates actions).

## 1. LIBERO inference on the Mac — VALIDATED

OpenVLA `openvla/openvla-7b-finetuned-libero-spatial` completed task 0
("pick up the black bowl … place it on the plate") in the LIBERO sim on MPS
(`done=True`), at ~2.5 s/step. Reproduce with [run_libero_mac.py](../run_libero_mac.py).

### Isolated environment (kept separate from the base-model `.venv`)
LIBERO/robosuite pin different versions than our base port, so they live in a
dedicated venv. The repos are cloned **outside** this repo (e.g. `~/dev-bru/`):

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git   # ~/dev-bru/LIBERO
git clone https://github.com/openvla/openvla.git                  # ~/dev-bru/openvla (for reference)

python3.10 -m venv .venv_libero
.venv_libero/bin/pip install torch==2.2.0 torchvision==0.17.0 \
  transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 peft==0.11.1 \
  accelerate draccus einops huggingface_hub sentencepiece jsonlines json-numpy rich protobuf \
  numpy==1.23.5 robosuite==1.4.1 bddl==1.0.1 easydict cloudpickle gym==0.25.2 \
  hydra-core==1.2.0 opencv-python "imageio[ffmpeg]" matplotlib termcolor future
.venv_libero/bin/pip install -e ~/dev-bru/LIBERO --no-deps --config-settings editable_mode=compat
echo "N" | .venv_libero/bin/python -c "import libero.libero"   # writes default LIBERO config
```

Key Mac gotchas (all handled in `run_libero_mac.py`):
- **No prismatic/TF/dlimp** — the official eval pulls in TensorFlow + the
  `prismatic` package + `dlimp`, which are painful on arm64. We bypass them:
  load the checkpoint via `AutoModelForVision2Seq(..., trust_remote_code=True)`
  (the same fp16/eager/MPS approach as the base-model port) and reimplement the
  image preprocessing (180° rotate → resize 224 → center-crop 0.9) in PIL.
- `bddl==1.0.1` (newer bddl 3.x breaks LIBERO); `robosuite==1.4.1`.
- LIBERO is a namespace package → install with `editable_mode=compat`.
- `_supports_sdpa=False` shim, `PYTORCH_ENABLE_MPS_FALLBACK=1`, no `MUJOCO_GL=egl`.

### Run / watch / change the prompt
```bash
.venv_libero/bin/python run_libero_mac.py --list                 # list task ids + prompts
.venv_libero/bin/python run_libero_mac.py --task_id 0            # run task 0
.venv_libero/bin/python run_libero_mac.py --task_id 0 --watch    # live window, updates each step
.venv_libero/bin/python run_libero_mac.py --task_id 0 --video --video_path ~/Desktop/r.mp4
.venv_libero/bin/python run_libero_mac.py --task_id 0 --prompt "put the black bowl on the plate"
.venv_libero/bin/python run_libero_mac.py --suite libero_object --task_id 0
```
Task IDs index the suite's ordered task list; each maps to a `.bddl` file whose
`language` field is the built-in instruction. `--prompt` overrides it (the model
was finetuned on the built-in phrasings, so rewordings can degrade).

## 2. Fine-tuning on our own task — PLAN (decided)

Zero-shot OpenVLA fails on our custom Franka scene (wrong embodiment/camera).
Making it work there requires fine-tuning. Decisions made:

- **Train on a rented NVIDIA GPU**; collect data + run inference on the Mac.
  (Fine-tuning a 7B model — even LoRA — needs CUDA: flash-attn, bitsandbytes,
  FSDP. Not practical on MPS.)
- **Data via robosuite/LIBERO, not this repo's scene.** This repo's hand-rolled
  PID/IK tracks poses poorly (joint error ~1 rad), so a reliable scripted grasp
  is impractical here. robosuite's OSC_POSE controller tracks EE-delta commands
  precisely, and OpenVLA's data→RLDS→finetune pipeline is built around it.
- **Instruction-conditioned strategy:** both candidate objects present every
  episode; the instruction names the target; ~50/50 split; randomized positions
  — so the policy must read the instruction to disambiguate (not memorize).
- **Base it on an existing LIBERO multi-object task** (reuse its reliable scene,
  success check, and controller) rather than authoring a new task from scratch.

### Pipeline
1. Pick a LIBERO multi-object pick-place task.
2. Scripted **OSC closed-loop expert** (read object pose from obs → servo EE →
   grasp → lift → place), recording `(agentview image, 7-D OSC action,
   instruction)`; keep only successes.
3. Convert to **RLDS** via OpenVLA's `rlds_dataset_builder`.
4. **LoRA-finetune** with `vla-scripts/finetune.py` on the NVIDIA GPU.
5. Run the finetuned checkpoint back on the Mac (as in part 1).

`demo_collect.py` in this repo is an earlier attempt at step 2 **in our own
MuJoCo scene** (option A); it is superseded by the robosuite path above because
of the controller-tracking issue, but its instruction-conditioned recording
logic is a useful reference.
