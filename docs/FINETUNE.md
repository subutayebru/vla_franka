# Fine-tuning OpenVLA on our LIBERO demos (GPU side) → deploy on Mac

End-to-end recipe for the plan in `docs/LIBERO_EVAL.md`: we collect demos on the
Mac, **fine-tune on a CUDA GPU**, then run the result back on the Mac.

## Which dataset?

Two collectors exist; **`robosuite_collect.py` is the active one** — it produces
demos for *our own task* (Franka + **yellow/blue cube**, pick-by-colour) using
robosuite's reliable OSC controller (`rs_demos/`). `libero_collect.py` (LIBERO
`libero_object`, `libero_demos/`) was the earlier stepping stone. **Both write
the identical `.npz` schema below**, so the RLDS + finetune steps are the same;
only the deploy env differs (robosuite Stack vs LIBERO).

## 0. What we collected (Mac)

The collectors write one `.npz` per successful episode (`rs_demos/` or
`libero_demos/`):

| key | shape / type | meaning |
|---|---|---|
| `images` | `(T, 256, 256, 3)` uint8 | agentview frames (already 180°-rotated to training orientation) |
| `actions` | `(T, 7)` float32 | OSC deltas `[dx,dy,dz, drx,dry,drz, gripper]`, gripper −1 open / +1 close |
| `instruction` | str | e.g. `"pick up the alphabet soup and place it in the basket"` |

All episodes share one scene with **all objects present** and the instruction
names the target, so the data is instruction-conditioned (the policy must read
the instruction to disambiguate).

Copy `libero_demos/` to the GPU box.

## 1. Convert to RLDS (GPU box)

OpenVLA's trainer consumes **RLDS/TFDS**. Use the official template
[rlds_dataset_builder](https://github.com/moojink/rlds_dataset_builder):

```bash
git clone https://github.com/moojink/rlds_dataset_builder
cd rlds_dataset_builder
conda env create -f environment_ubuntu.yml && conda activate rlds_env
```

Create a builder (e.g. `libero_pick/libero_pick_dataset_builder.py`) whose
`_generate_examples` reads our `.npz` files and yields steps. The feature spec
must match OpenVLA's expectations:
- `observation/image`: `Image(256,256,3, uint8)`
- `observation/state`: `Tensor(8,) float32` (proprio; can be zeros — OpenVLA
  ignores proprio, but the field is expected)
- `action`: `Tensor(7,) float32`
- `language_instruction`: `Text`
- per-step `reward`/`discount`/`is_first`/`is_last`/`is_terminal`

Then build:
```bash
tfds build   # writes ~/tensorflow_datasets/libero_pick/1.0.0
```

> The action normalization (q01/q99 per dimension) is computed by OpenVLA's
> dataloader from the RLDS dataset at train time and saved into the checkpoint as
> `norm_stats[<dataset_name>]`; we pass that name as `--unnorm_key` at inference.

## 2. LoRA fine-tune (GPU box)

In the `openvla` repo, register the dataset in
`prismatic/vla/datasets/rlds/oxe/configs.py` + `transforms.py` + the mixture in
`mixtures.py` (one entry pointing at `libero_pick`), then:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir ~/tensorflow_datasets \
  --dataset_name libero_pick \
  --run_root_dir runs \
  --use_lora True --lora_rank 32 \
  --batch_size 8 --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --save_steps 5000
```
- Needs a CUDA GPU ~24–48 GB (flash-attn + bitsandbytes are fine here).
- `--image_aug True` ⇒ at inference use **center crop** (our `run_libero_mac.py`
  already does crop 0.9).
- LoRA adapters are merged into a checkpoint under `runs/...`.

## 3. Deploy back on the Mac

Copy the merged checkpoint to the Mac, then run it on the **same env it was
collected in**, with the same fp16/eager/MPS load path we proved earlier:

- **robosuite yellow/blue task** → `robosuite_eval_mac.py` (recolored Stack;
  prompt `pick up the {yellow|blue} cube`; identical agentview flip + resize-224
  + center-crop preprocessing as the collector):
  ```bash
  .venv_libero/bin/python robosuite_eval_mac.py \
    --checkpoint /path/to/our_finetuned_ckpt --color yellow --watch
  ```
- **LIBERO task** → `run_libero_mac.py --checkpoint /path/...`.

The `--unnorm_key` must equal the RLDS dataset name used in training (the
scripts default it; override if your dataset name differs).

## Notes / gotchas
- **Zero-variance rotation dims.** Our expert grasps top-down, so action dims
  `drx,dry,drz` are all exactly 0 across the dataset (verified). OpenVLA's action
  normalization does `2*(x-q01)/(q99-q01)-1`, which **divides by zero** when
  `q99==q01`. Fixes (pick one): (a) ensure the dataset-stats **mask** is `False`
  for those 3 dims so they pass through un-normalized (OpenVLA already masks the
  gripper dim this way); or (b) add tiny noise to the rotation dims at collection
  time (`actions[:,3:6] += 1e-4 * randn`). Option (a) is cleaner.
- 249 episodes, ~33k transitions, 5 instructions (~50 each), all in one scene
  with all objects present (instruction-conditioned).
- The 5 reliable objects (alphabet soup, bbq sauce, tomato sauce, butter,
  chocolate pudding) are what we trained on; tall bottles need per-object grasp
  tuning in `libero_collect.py` before they can be added.
- Keep the inference preprocessing identical to training: agentview, 180° rotate,
  resize 224, center-crop 0.9 — all already in `run_libero_mac.py`.
- Steps 1–2 are CUDA-only; step 3 is the Mac MPS path we've already proven.
