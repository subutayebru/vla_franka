# 🐼 VLA_FRANKA

**Simulation and Control of the Franka Emika Panda Arm in MuJoCo with Vision-Language-Action Model Integration**

> 🍎 **You are on the `macos` branch — the Apple Silicon (MPS) port.**
> OpenVLA-7B is loaded in fp16 on the Mac GPU (MPS), with no `bitsandbytes`
> 4-bit quantization (CUDA-only). The original NVIDIA/CUDA path lives on `main`.
> See [docs/MACOS_PORT.md](docs/MACOS_PORT.md) for the full rationale and gotchas.

---

## 🧠 Overview

This repo connects **OpenVLA** with a **MuJoCo Franka pick-and-place** environment and a **PID-based joint controller**. It lets you:

- 🚀 run **“smart” closed-loop control with OpenVLA** from camera images,
- run **classical / hand-crafted pick-and-place** for sanity checks,
- 🧠 switch between **different model variants** and **prompts** via config,
- 🌍 change **environment**, **camera**, **action scales**, and **PID gains** from a single YAML file,
- log **robot state, actions and torques** in a structured way 
- ready to be turned into a dataset for fine-tuning.


## 🗂️ Repository Structure 

```bash
vla_franka/
  pnp.py                        # classical pick-and-place for sanity check
  run_vla_control.py            # main entrypoint for VLA-based control

  core/
    config.py                   # dataclass + YAML + CLI config loader
    env_wrapper.py              # MuJoCo env + viewer + camera + HUD
    control_utils.py            # action → pose mapping + PID creation
    vla_agent.py                # OpenVLA wrapper (load + act())
    ik_solver.py                # numerical IK for Panda end-effector
    diagnostics_n_logging.py    # HUD & CSV logger helpers (init_hud, make_control_logger, ...)

  configs/
    default.yaml                # main config (camera, model name, PID gains, etc.)
```

## 📦 Dependencies

- mujoco (the `mujoco` wheel ships the `mjpython` launcher used on macOS)
- torch (Apple Silicon wheels include the MPS backend)
- transformers==4.40.1 (pinned — OpenVLA's `trust_remote_code` modeling expects it)
- timm==0.9.16, tokenizers==0.19.1

> `bitsandbytes` is **not** used on this branch (it is CUDA/Linux-only). The
> model runs in fp16 with ~14 GB of weights, so you want a Mac with comfortable
> unified memory (this branch was set up for a 48 GB machine).

## 🚀 Quickstart (macOS / Apple Silicon)

### 1. Create the environment

```bash
# Python 3.10 matches the original environment (Linux used 3.10.19).
# Install it once via Homebrew if you don't have it: brew install python@3.10
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the VLA closed-loop control

**Use `mjpython`, not `python`** — on macOS the interactive MuJoCo viewer must
run on the main thread, and only `mjpython` (shipped with the `mujoco` wheel)
sets that up. Plain `python` will fail to open the viewer.

```bash
# Default instruction from configs/default.yaml
mjpython run_vla_control.py

# Or override the instruction on the fly
mjpython run_vla_control.py --prompt "go over the yellow cube"
```

The classical sanity check does not load the model and runs fine with `mjpython pnp.py`.

**Tip** Run commands from the repo root so relative asset paths like `asset/...` resolve correctly.

> First run downloads ~14 GB of OpenVLA-7B weights from the HuggingFace Hub.
> Inference is not real-time on MPS (expect a few seconds per policy step); the
> simulation stays correct, it just advances the policy less often.

---

## 🐧 Linux / NVIDIA path (on the `main` branch)

The sections below describe the original CUDA + Docker workflow. They are
**Linux/NVIDIA only** (`MUJOCO_GL=egl`, `bitsandbytes` 4-bit loading) and are
kept here for reference; use the `main` branch for that setup.

### Option A — Run on your local machine (w/ creating a conda env)

```bash 
conda env create -f requirements.yml
conda activate vla_franka

# Run VLA control loop
python run_vla_control.py

#### OR! ####

# Run classical sanity check
python pnp.py
```
**Tip** Make sure you run commands from the repo root so relative asset paths like asset/... resolve correctly.

### Option B — Run with Docker (GPU + MuJoCo Viewer)

1) **Allow Docker to use your display**

```bash
xhost +local:docker
```

2) **Build image**

From repo root:
```bash
docker build -t vla_franka:latest .
```

3) **Run default demo (Dockerfile CMD → pnp.py)**

```bash
   sudo docker run -it --rm \
  --gpus all \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e MUJOCO_GL=egl \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)":/workspace/vla_franka \
  -w /workspace/vla_franka \
  --ipc=host \
  vla_franka:latest
```
4) **Run VLA control (override the default CMD)**

```bash
   sudo docker run -it --rm \
  --gpus all \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e MUJOCO_GL=egl \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)":/workspace/vla_franka \
  -w /workspace/vla_franka \
  --ipc=host \
  vla_franka:latest \
  python run_vla_control.py
```
5) **If you just want to enter the container**
   
```bash
   sudo docker run -it --rm \
  --gpus all \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e MUJOCO_GL=egl \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)":/workspace/vla_franka \
  -w /workspace/vla_franka \
  --ipc=host \
  vla_franka:latest \
  bash
```
**Then inside the container:**
```bash
python pnp.py
python run_vla_control.py
```

## 🙏 Acknowledgements
> 🧩 Basic pick-and-place environment setup is adapted from  
> https://github.com/volunt4s/mujoManipulation

