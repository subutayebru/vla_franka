# rs_pick_dataset_builder.py
#
# TFDS/RLDS builder that converts our robosuite demos (rs_demos/*.npz from
# robosuite_collect.py) into the RLDS format OpenVLA's finetuner consumes.
#
# USAGE (on the CUDA/GPU box, NOT the Mac):
#   git clone https://github.com/moojink/rlds_dataset_builder
#   cd rlds_dataset_builder
#   conda env create -f environment_ubuntu.yml && conda activate rlds_env
#   mkdir rs_pick && cp /path/to/rs_pick_dataset_builder.py rs_pick/ && touch rs_pick/__init__.py
#   export RS_DEMOS=/path/to/rs_demos
#   cd rs_pick && tfds build
#   # -> ~/tensorflow_datasets/rs_pick/1.0.0  (dataset name: "rs_pick")
#
# Then register "rs_pick" in OpenVLA's dataset configs/mixtures and run
# vla-scripts/finetune.py --dataset_name rs_pick (see docs/FINETUNE.md).

import glob
import os

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf


class RsPick(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Yellow/blue cube pick demos (robosuite, OSC)."}

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(256, 256, 3), dtype=np.uint8,
                            encoding_format="png", doc="agentview RGB"),
                        # OpenVLA ignores proprio but the field is expected.
                        "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,), dtype=np.float32,
                        doc="OSC delta [dx,dy,dz,drx,dry,drz, gripper(+1 close/-1 open)]"),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                }),
            }))

    def _split_generators(self, dl_manager):
        data_dir = os.environ.get("RS_DEMOS", "rs_demos")
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        assert files, f"no .npz found in {data_dir} (set RS_DEMOS)"
        return {"train": self._generate_examples(files)}

    def _generate_examples(self, files):
        for path in files:
            d = np.load(path, allow_pickle=True)
            images = d["images"].astype(np.uint8)        # (T,256,256,3)
            actions = d["actions"].astype(np.float32)     # (T,7)
            instr = str(d["instruction"])
            T = len(actions)
            steps = []
            for i in range(T):
                steps.append({
                    "observation": {
                        "image": images[i],
                        "state": np.zeros(8, dtype=np.float32),
                    },
                    "action": actions[i],
                    "discount": np.float32(1.0),
                    "reward": np.float32(1.0 if i == T - 1 else 0.0),
                    "is_first": i == 0,
                    "is_last": i == T - 1,
                    "is_terminal": i == T - 1,
                    "language_instruction": instr,
                })
            yield path, {"steps": steps, "episode_metadata": {"file_path": path}}
