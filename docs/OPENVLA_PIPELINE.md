# How OpenVLA turns a command + image into robot motion

A reference for understanding the action-generation pipeline in this repo, and
why the simulated arm moves the way it does. Grounded in the actual checkpoint
(`openvla/openvla-7b`, see its `config.json`).

## TL;DR — there is no segmentation

OpenVLA does **no segmentation, no object detection, and no explicit "find the
cube" step**. It is one end-to-end network trained by imitation that maps
**pixels + your sentence → 7 numbers** directly. Any sense of "where the cube is"
is *implicit* in the weights (via attention), never an inspectable mask or box.

## The checkpoint

From `config.json`:

| Field | Value | Meaning |
|---|---|---|
| `vision_backbone_id` | `dinosiglip-vit-so-224px` | fused **DINOv2 + SigLIP** ViTs, 224×224 input |
| `use_fused_vision_backbone` | `true` | the two encoders' features are concatenated |
| `llm_backbone_id` | `llama2-7b-pure` | **Llama-2-7B** language model |
| `arch_specifier` | `no-align+fused-gelu-mlp` | the vision→LLM projector is a GELU MLP |
| `n_action_bins` | `256` | each action dimension is discretized into 256 bins |
| action dim | `7` | `[dx, dy, dz, droll, dpitch, dyaw, gripper]` |

## The pipeline, step by step

For a prompt like *"What action should the robot take to grasp the yellow cube?"*:

1. **Vision encoding.** The camera frame (the left panel of the live view — what
   OpenVLA sees) is resized to **224×224** and run through the fused
   **DINOv2 + SigLIP** backbone, producing a grid of **patch features** (256
   visual tokens). These are generic visual features — *not* labels.

2. **Projector.** A small GELU MLP maps each patch feature into the **same
   embedding space as Llama's word tokens**. The image becomes "256 visual words."

3. **Prompt tokenization.** The text is wrapped in the training template and
   tokenized:
   ```
   In: What action should the robot take to grasp the yellow cube?\nOut:
   ```
   "yellow", "cube", "grasp" are ordinary tokens. The 256 image tokens are
   spliced in right after the `<BOS>` token, then the text tokens
   (the `multimodal_embeddings` concatenation in `modeling_prismatic.py`).

4. **Llama-2-7B generates 7 action tokens.** The LLM autoregressively emits
   **exactly 7 tokens**, one per action dimension. The trick: OpenVLA
   **repurposed the 256 least-used entries of Llama's vocabulary as action bins**.
   So each generated token is really "which of 256 buckets" for that dimension.
   The only "localization" is soft cross-attention between text and visual tokens.

5. **De-tokenize → un-normalize** (`predict_action`):
   - token id → bin index (0–255) → **bin center** in `[-1, 1]`
   - then un-normalized using the **`bridge_orig`** dataset's `q01`/`q99`
     statistics (the `vla_unnorm_key` in `configs/default.yaml`) back to physical
     units.

   Result = a **7D delta**: relative end-effector motion + a gripper command.
   This is exactly what the console `action_report` and the action-graph window
   show.

## From the 7D delta to joint motion (this repo)

```
camera image ─▶ OpenVLA ─▶ 7D delta  [dx,dy,dz,droll,dpitch,dyaw,grip]
   delta ─▶ apply_action_to_pose()  (scale by step_scale_xyz=0.5, rpy=0.8) ─▶ target EE pose + gripper
   target pose ─▶ solve_IK()  (damped least squares) ─▶ target joint angles
   joints ─▶ PID (kp=800, ki=20, kd=100) ─▶ torques ─▶ MuJoCo step
```
Files: `core/vla_agent.py` (inference), `core/control_utils.py`
(`apply_action_to_pose`), `core/ik_solver.py` (`solve_IK`), `run_vla_control.py`
(loop + PID). Between the model's number and the joint moving there are **three**
more transforms (scale → IK → PID), each shaping the visible motion.

## Why the arm moves "like that"

The behavior is dominated by **distribution mismatch**, not a bug:

1. **Camera viewpoint.** `bridge_orig` (and most OpenVLA training data) uses a
   **fixed third-person** camera. Feeding the **eye-in-hand wrist camera** is a
   viewpoint the model essentially never saw → out-of-distribution input →
   unreliable deltas. (This is why this branch defaults the experiment to the
   third-person `standing_cam`; see below.)
2. **Embodiment.** Bridge data is a **WidowX** arm; here it drives a **Franka
   Panda** in MuJoCo. Action sign/scale conventions don't transfer cleanly.
3. **Un-norm scale.** `bridge_orig` rescales the bins to WidowX/Bridge
   magnitudes, then `step_scale_*` rescales again — absolute motion sizes are
   somewhat arbitrary here.
4. **Reactive, no planning.** It's imitation-trained: each step predicts "given
   what I see, what tiny move would the demonstrator make." Unfamiliar view in →
   erratic move out.

The honest test is the **EE→cube distance trend** (red curve in the action-graph
window): a sustained ↓ means the policy is perceiving and approaching the target.

## Camera experiment

To move *toward* the training distribution, this branch feeds OpenVLA the
third-person `standing_cam` by default (`camera_name` in `configs/default.yaml`),
while the live-view's right panel keeps the wrist cam as a reference. Flip the
input camera per-run without editing the YAML:

```bash
python run_vla_control.py --prompt "grasp the yellow cube"                  # standing_cam (default)
python run_vla_control.py --camera panda_eye_in_hand --prompt "grasp ..."   # wrist cam
```

Compare which viewpoint makes the distance curve trend ↓ more reliably.

## What this is NOT

- Not a planner or search — no trajectory optimization, no goal reasoning.
- Not a perception stack — no detector, segmenter, or pose estimator.
- Not calibrated to this Franka/MuJoCo setup out of the box — expect to fine-tune
  (or at least match camera/embodiment) for reliable task success.
