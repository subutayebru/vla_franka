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

```bash 
pip install -r requirements.txt 
```

## 🙏 Acknowledgements
> 🧩 Basic pick-and-place environment setup is adapted from  
> https://github.com/volunt4s/mujoManipulation

