#!/usr/bin/env python
# rs_to_lerobot.py
#
# Convert our robosuite demos (rs_demos/*.npz from robosuite_collect.py) into a
# LeRobotDataset for fine-tuning SmolVLA. Run with the LeRobot venv:
#   .venv_lerobot/bin/python rs_to_lerobot.py --root lerobot_data/rs_pick
#   .venv_lerobot/bin/python rs_to_lerobot.py --root /tmp/rs_pick_test --limit 3   # quick test

import argparse
import glob
import os
import shutil

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="rs_demos")
    ap.add_argument("--repo_id", default="rs_pick")
    ap.add_argument("--root", default="lerobot_data/rs_pick")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="only convert N episodes (testing)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.demos, "*.npz")))
    if args.limit:
        files = files[: args.limit]
    assert files, f"no .npz in {args.demos}"
    d0 = np.load(files[0], allow_pickle=True)
    H, W, _ = d0["images"].shape[1:]
    state_dim = d0["states"].shape[1]
    action_dim = d0["actions"].shape[1]
    print(f"{len(files)} episodes | img {H}x{W} | state {state_dim} | action {action_dim}")

    features = {
        # "image" (PNG) not "video": avoids the torchcodec/ffmpeg version-matching
        # mess on macOS; reads fine on Mac and the GPU box. SmolVLA treats
        # observation.images.* the same regardless of storage backend.
        "observation.images.agentview": {
            "dtype": "image", "shape": (H, W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32", "shape": (state_dim,),
            "names": [f"s{i}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32", "shape": (action_dim,),
            "names": [f"a{i}" for i in range(action_dim)],
        },
    }

    if os.path.exists(args.root):
        shutil.rmtree(args.root)
    ds = LeRobotDataset.create(args.repo_id, args.fps, features=features,
                               root=args.root, use_videos=False)

    for f in files:
        d = np.load(f, allow_pickle=True)
        imgs, acts, states = d["images"], d["actions"], d["states"]
        instr = str(d["instruction"])
        for t in range(len(acts)):
            ds.add_frame({
                "observation.images.agentview": imgs[t].astype(np.uint8),
                "observation.state": states[t].astype(np.float32),
                "action": acts[t].astype(np.float32),
                "task": instr,
            })
        ds.save_episode()
    print(f"done -> {args.root}")


if __name__ == "__main__":
    main()
