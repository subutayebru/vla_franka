# 🐼 VLA_FRANKA

**Simulation and Control of the Franka Emika Panda Arm in MuJoCo with Vision-Language-Action Model Integration**

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

- mujoco
- torch 
- transformers==1.40.1 
- accelerate==0.19.1 / bitsandbytes for 4-bit loading

## 🚀 Quickstart

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

