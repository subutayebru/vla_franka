# macOS / Apple Silicon Port — Decisions & Rationale

This document records *why* the `macos` branch differs from `main`, so the
choices are clear to anyone (including future us) reading the diff.

## Summary

`main` targets **Ubuntu + NVIDIA GPU**. The `macos` branch runs the same
**OpenVLA-7B → MuJoCo Franka** closed loop on **Apple Silicon (M-series, MPS)**.
There is no real robot in this project — OpenVLA's 7D actions drive a MuJoCo
simulation through IK + PID, so the *entire* pipeline can run on a Mac. The only
hard blocker was how the model is loaded.

Target machine for this branch: **Mac with 48 GB unified memory.**

## The one real blocker: model loading

On `main`, `core/vla_agent.py` loads OpenVLA-7B with `bitsandbytes` 4-bit
quantization (`BitsAndBytesConfig`, `load_in_4bit=True`) and
`device_map="auto"`. **`bitsandbytes` is CUDA/Linux-only** — it does not install
or run on Apple Silicon, and `device_map="auto"` (Accelerate) is built around
CUDA placement.

**Decision:** drop quantization entirely and load the model in **fp16 (~14 GB
weights)** directly onto the **MPS** GPU. With 48 GB of unified memory there is
ample headroom, so no Mac-compatible quantizer (e.g. `optimum-quanto`) is
needed.

## Code changes on this branch

### `core/vla_agent.py` — rewritten `OpenVLAAgent`
- **No `BitsAndBytesConfig`, no `quantization_config`, no `device_map="auto"`.**
- Device pick: `"mps"` if `torch.backends.mps.is_available()` else `"cpu"`.
- Load with `torch_dtype=torch.float16, low_cpu_mem_usage=True,
  trust_remote_code=True`, then `model.to(device).eval()`.
- **Force eager attention:** an `AutoConfig` is built with
  `attn_implementation="eager"` / `_attn_implementation="eager"`. This avoids
  flash-attn / SDPA kernels that are unavailable or misbehave on MPS.
- **`_supports_sdpa` shim:** OpenVLA's `trust_remote_code` modeling predates a
  flag newer `transformers` checks; we set
  `PreTrainedModel._supports_sdpa = False` if missing (same shim the original
  `backup/vla_openvla.py` used).
- **Set `generation_config.pad_token_id` / `eos_token_id`** from the tokenizer
  to silence per-step generation warnings.
- `act()` moves **all** processor outputs to the device — `pixel_values` cast to
  the vision backbone's fp16 dtype, `input_ids` left integer. The 7D action
  return contract is unchanged, so the rest of the loop is untouched.
- **Drops the processor `attention_mask`** before calling `predict_action`.
  Upstream `predict_action` appends a special token (`29871`) to `input_ids` but
  does not extend a caller-supplied mask, so the mask ends up one token short.
  The multimodal `forward` then splices 256 image-patch tokens into both embeds
  and mask, turning that off-by-one into a hard shape mismatch
  (`attn_weights + causal_mask`, e.g. `276 vs 275`) on the **eager** attention
  path. With no mask passed, `generate()` rebuilds an all-ones mask at the
  correct post-append length. On CUDA/SDPA the stale mask is silently discarded,
  which is why upstream never hit this. Verified working: load ~10s from cache,
  ~2.5s per inference on MPS.

### `run_vla_control.py` — runtime env (top of file, before torch/mujoco import)
- `PYTORCH_ENABLE_MPS_FALLBACK=1` — any op OpenVLA uses that MPS lacks falls back
  to CPU instead of crashing.
- Clears `MUJOCO_GL` if it is `egl` (that's the Linux/Docker offscreen backend;
  macOS uses the default CGL backend).

### `requirements.txt`
- Removed `bitsandbytes==0.48.2`. Kept the OpenVLA-critical pins:
  `transformers==4.40.1`, `tokenizers==0.19.1`, `timm==0.9.16`,
  `mujoco==3.1.6`, `numpy==1.26.4`, `torch==2.4.1`, `torchvision==0.19.1`.

## Critical runtime gotcha: `mjpython`

`src/mujoco_parser.py` opens the interactive viewer via
`mujoco.viewer.launch_passive`. On macOS the viewer **must** run on the main
thread (Cocoa requirement), which only the **`mjpython`** launcher (shipped with
the `mujoco` pip wheel) sets up. Therefore:

```bash
mjpython run_vla_control.py --prompt "..."   # ✅ viewer opens
python   run_vla_control.py                  # ❌ viewer fails on macOS
```

## Matplotlib HUD caveat

`core/env_wrapper.py` always opens a matplotlib HUD (`init_hud` / `update_hud`)
alongside the GLFW viewer. Two GUI event loops under `mjpython` can be flaky on
macOS. If the HUD misbehaves, guard those calls so `get_image()` just returns the
rendered frame. (Not changed yet — only if needed.)

## Performance expectations

- fp16 7B on MPS generates 7 action tokens per policy step → expect roughly a
  few **seconds per step**; the first call is slower (warmup + a ~14 GB
  HuggingFace weight download).
- `policy_hz: 5` in `configs/default.yaml` is aspirational, not enforced in real
  time. The simulation remains physically correct; it simply advances the policy
  less often in wall-clock terms.

## Fallbacks

- If a hard MPS op bug surfaces, force `device="cpu"` in `OpenVLAAgent` — fine on
  48 GB, just slower.
- **Do not** casually upgrade `transformers` past 4.40.1; OpenVLA's remote
  modeling code expects it.

## Branch intent

These edits intentionally diverge from `main` and are **not** meant to merge
back. `main` stays the canonical CUDA path; `macos` is the Apple Silicon port.
