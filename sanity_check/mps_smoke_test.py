# sanity_check/mps_smoke_test.py
#
# Standalone OpenVLA inference smoke test for Apple Silicon (MPS).
#
# Loads OpenVLA-7B in fp16 on the Mac GPU, runs ONE forward pass on a dummy
# image, and prints the 7D action. No MuJoCo, no viewer, no robot loop — this is
# the fastest way to confirm the model loads and infers on MPS before running
# the full closed loop with `mjpython run_vla_control.py`.
#
# Run from the repo root:
#     python sanity_check/mps_smoke_test.py
#     python sanity_check/mps_smoke_test.py --prompt "pick up the cube"
#
# Note: the first run downloads ~14 GB of weights from HuggingFace.

import os

# Fall back to CPU for any op MPS doesn't implement, instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
import numpy as np
from PIL import Image

# Make sure repo-root imports (core/...) work when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_cfg
from core.vla_agent import OpenVLAAgent


def main():
    cfg, _ = load_cfg()  # reads configs/default.yaml; honours --prompt / --cfg

    print(f"[smoke] model       : {cfg.vla_model_name}")
    print(f"[smoke] instruction : {cfg.instruction!r}")
    print("[smoke] loading model (first run downloads ~14 GB)...", flush=True)

    t0 = time.time()
    agent = OpenVLAAgent(cfg)
    print(f"[smoke] loaded in {time.time() - t0:.1f}s on device: {agent.device}")

    # Dummy 480x640 RGB image (matches the sim's render size). A mid-grey frame
    # rather than pure black, just so it isn't a degenerate all-zero input.
    dummy = np.full((480, 640, 3), 128, dtype=np.uint8)
    image = Image.fromarray(dummy)

    print("[smoke] running one inference step...", flush=True)
    t1 = time.time()
    action = agent.act(image)
    dt = time.time() - t1

    action = np.asarray(action).reshape(-1)
    ok = action.shape == (7,) and np.isfinite(action).all()

    print(f"[smoke] inference took {dt:.2f}s")
    print(f"[smoke] action shape : {action.shape}")
    print(f"[smoke] action       : {np.array2string(action, precision=4)}")
    print(f"[smoke] finite & 7D  : {ok}")

    if not ok:
        print("[smoke] FAILED: action is not a finite length-7 vector.")
        sys.exit(1)
    print("[smoke] OK — OpenVLA inference works on this Mac. "
          "Now try: mjpython run_vla_control.py")


if __name__ == "__main__":
    main()
