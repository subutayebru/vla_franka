# vla_openvla.py
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import re
import numpy as np
from PIL import Image
import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,        # works with your current stack
    AutoModelForImageTextToText,   # newer alias; we try it first
    BitsAndBytesConfig,
    AutoConfig,
)
import transformers

# Some older remote model implementations don't define this flag that newer HF checks.
if not hasattr(transformers.modeling_utils.PreTrainedModel, "_supports_sdpa"):
    transformers.modeling_utils.PreTrainedModel._supports_sdpa = False

# Force "eager" attention on some stacks (avoids SDPA oddities)
if hasattr(torch.backends, "cuda"):
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

# ========== Prompt & decoding ==========

ACTION_SYSTEM_PROMPT = (
    "You control a robot gripper. "
    "Output ONLY 7 comma-separated numbers in [-1,1] as: "
    "dx, dy, dz, droll, dpitch, dyaw, grip "
    "(EE-frame deltas; positive grip means close). No other words."
)

# deterministic generation
GEN_KW = dict(
    max_new_tokens=24,
    do_sample=False,
    temperature=0.0,
)

# float regex and "first 7 floats anywhere" parser
_FLOAT = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
FIND_FLOATS = re.compile(_FLOAT)

def parse_seven_floats(text: str):
    """
    Be forgiving: extract the first 7 numbers we see, clamp to [-1, 1].
    Returns np.ndarray shape (7,) or None.
    """
    nums = FIND_FLOATS.findall(text)
    if len(nums) < 7:
        return None
    vals = np.array([float(x) for x in nums[:7]], dtype=np.float32)
    return np.clip(vals, -1.0, 1.0)

# ========== Wrapper ==========

class OpenVLA:
    """
    Contract:
      predict(rgb, instruction) -> np.ndarray shape (7,), in [-1,1]
      [dx, dy, dz, droll, dpitch, dyaw, grip] in EE frame.
      Positive grip => close.
    """

    def __init__(self, ckpt="openvla/openvla-v01-7b", device="cuda", debug=False):
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.debug  = debug

        # processor
        self.proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)

        # try the new class first, fall back to Vision2Seq to match your env
        self.model = None
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
        for attr in ("attn_implementation", "_attn_implementation"):
            try:
                setattr(cfg, attr, "eager")
            except Exception:
                pass

        # prefer AutoModelForImageTextToText when available (reduces deprecation noise)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                ckpt,
                trust_remote_code=True,
                config=cfg,
                quantization_config=bnb_cfg,
                device_map="auto",
            )
        except Exception:
            # fallback
            self.model = AutoModelForVision2Seq.from_pretrained(
                ckpt,
                trust_remote_code=True,
                config=cfg,
                quantization_config=bnb_cfg,
                device_map="auto",
            )

        if hasattr(self.model, "tie_weights"):
            try:
                self.model.tie_weights()
            except Exception:
                pass
        self.model.eval()

        # Ensure pad/eos ids are set for clean generation (some checkpoints omit them)
        try:
            tok = getattr(self.proc, "tokenizer", None)
            if tok is not None:
                if getattr(self.model.generation_config, "pad_token_id", None) is None and tok.pad_token_id is not None:
                    self.model.generation_config.pad_token_id = tok.pad_token_id
                if getattr(self.model.generation_config, "eos_token_id", None) is None and tok.eos_token_id is not None:
                    self.model.generation_config.eos_token_id = tok.eos_token_id
        except Exception:
            pass

        # Optional warmup (ignore failures)
        try:
            _ = self.predict(np.zeros((480, 640, 3), dtype=np.uint8), "warmup")
        except Exception:
            pass

    @torch.no_grad()
    def predict(self, rgb, instruction: str) -> np.ndarray:
        # Convert image to PIL
        if isinstance(rgb, np.ndarray):
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(rgb[..., :3])
        else:
            pil_img = rgb

        # Single-turn prompt: system instruction + user instruction
        user_prompt = f"{ACTION_SYSTEM_PROMPT}\nInstruction: {instruction}\nAction:"

        inputs = self.proc(images=[pil_img], text=user_prompt, return_tensors="pt")

        # send to first parameter device (works with device_map='auto')
        first_dev = next(self.model.parameters()).device
        inputs = {k: v.to(first_dev) for k, v in inputs.items()}

        out_ids = self.model.generate(**inputs, **GEN_KW)

        # Decode using processor tokenizer when available
        try:
            text = self.proc.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        except Exception:
            try:
                text = self.model.generation_config._tokenizer.decode(out_ids[0], skip_special_tokens=True)
            except Exception:
                text = str(out_ids)

        if self.debug:
            # Print raw once in a while (set debug=True in constructor to enable)
            if np.random.rand() < 0.05:
                print("[OpenVLA raw]:", text)

        a7 = parse_seven_floats(text)
        if a7 is None or not np.isfinite(a7).all() or a7.shape != (7,):
            # Safe fallback: no motion, open gripper
            return np.array([0,0,0, 0,0,0, -1], dtype=np.float32)

        return a7.astype(np.float32)

# quick local test (optional)
if __name__ == "__main__":
    vla = OpenVLA(debug=True)
    fake = np.zeros((360, 360, 3), dtype=np.uint8)
    out = vla.predict(fake, "Move 1 cm forward and close gripper.")
    print("a7:", out)


# # vla_openvla.py
# import os
# os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# import re
# import numpy as np
# from PIL import Image
# import torch
# from transformers import (
#     AutoProcessor,
#     AutoModelForVision2Seq,        # works with your current stack
#     AutoModelForImageTextToText,   # newer alias; we try it first
#     BitsAndBytesConfig,
#     AutoConfig,
# )
# import transformers

# # ---------------------------------------------------------------------
# # Compatibility knobs
# # ---------------------------------------------------------------------
# # Some older remote model implementations don't define this flag that newer HF checks.
# if not hasattr(transformers.modeling_utils.PreTrainedModel, "_supports_sdpa"):
#     transformers.modeling_utils.PreTrainedModel._supports_sdpa = False

# # Force "eager" attention on some stacks (avoids SDPA oddities / version drift)
# if hasattr(torch.backends, "cuda"):
#     try:
#         torch.backends.cuda.enable_flash_sdp(False)
#         torch.backends.cuda.enable_mem_efficient_sdp(False)
#         torch.backends.cuda.enable_math_sdp(True)
#     except Exception:
#         pass

# # ---------------------------------------------------------------------
# # Numeric-only constrained decoding (fallback path)
# # ---------------------------------------------------------------------
# _FLOAT = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
# FIND_FLOATS = re.compile(_FLOAT)

# def parse_seven_floats(text: str):
#     """
#     Forgiving parser: grab the first 7 numbers; clip to [-1, 1].
#     Returns np.ndarray shape (7,) or None.
#     """
#     nums = FIND_FLOATS.findall(text or "")
#     if len(nums) < 7:
#         return None
#     vals = np.array([float(x) for x in nums[:7]], dtype=np.float32)
#     return np.clip(vals, -1.0, 1.0)

# def allowed_token_ids_for_numbers(tokenizer):
#     """
#     Build a set of token ids whose decoded string is made of numeric-ish characters.
#     This lets us constrain generation to digits, signs, commas, spaces, decimals, and exponents.
#     """
#     allowed_chars = set("0123456789-+., eE")
#     ids = []
#     vocab_size = getattr(tokenizer, "vocab_size", None)
#     # Some tokenizers (sentencepiece/BPE) don't have simple vocab_size; guard accordingly.
#     if vocab_size is None:
#         # Fallback: allow everything (won't constrain), but try to get eos
#         eos = getattr(tokenizer, "eos_token_id", None)
#         return set([eos] if eos is not None else [])

#     for tid in range(vocab_size):
#         try:
#             tok = tokenizer.convert_ids_to_tokens(tid)
#         except Exception:
#             try:
#                 tok = tokenizer.decode([tid], skip_special_tokens=True)
#             except Exception:
#                 continue
#         # Strip common leading markers (like Ġ, ▁) and keep only numeric-ish characters
#         plain = "".join(ch for ch in tok if (ch.isdigit() or ch in "-+., eE"))
#         if plain and all(ch in allowed_chars for ch in plain):
#             ids.append(tid)

#     # Always allow EOS to stop
#     eos_id = getattr(tokenizer, "eos_token_id", None)
#     if eos_id is not None:
#         ids.append(eos_id)
#     return set(ids)

# # ---------------------------------------------------------------------
# # Prompts / generation defaults used in the text fallback
# # ---------------------------------------------------------------------
# ACTION_PROMPT = (
#     "Output 7 numbers in [-1,1] as: dx,dy,dz,droll,dpitch,dyaw,grip.\nAction:"
# )

# GEN_KW = dict(
#     max_new_tokens=24,
#     do_sample=False,
#     temperature=0.0,
# )

# # ---------------------------------------------------------------------
# # Main wrapper
# # ---------------------------------------------------------------------
# class OpenVLA:
#     """
#     Contract:
#       predict(rgb, instruction) -> np.ndarray shape (7,) in [-1, 1] (or metric if unnorm_key applies).
#       Semantics: [dx, dy, dz, droll, dpitch, dyaw, grip]
#         - Usually EE-frame deltas in [-1,1]; your controller scales/applies frame transform.
#         - If the checkpoint exposes a numeric head and you pass `unnorm_key`, outputs may be unnormalized metric deltas.

#     Args:
#       ckpt (str): HF repo or local path (e.g., "openvla/openvla-v01-7b").
#       device (str): "cuda" or "cpu" (we still use device_map='auto' for sharding).
#       quant4bit (bool): use 4-bit quantization to reduce VRAM.
#       unnorm_key (str|None): e.g., "bridge_orig". Passed to predict_action() if available.
#       debug (bool): print occasional raw generations in text fallback.
#       force_text (bool): force the text fallback even if numeric head exists (for debugging).
#     """

#     def __init__(
#         self,
#         ckpt="openvla/openvla-v01-7b",
#         device="cuda",
#         quant4bit=True,
#         unnorm_key="bridge_orig",
#         debug=False,
#         force_text=False,
#     ):
#         self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
#         self.debug  = debug
#         self.unnorm_key = unnorm_key
#         self.force_text = force_text

#         # Processor
#         self.proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)

#         # Config (force eager attention if present)
#         cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
#         for attr in ("attn_implementation", "_attn_implementation"):
#             try:
#                 setattr(cfg, attr, "eager")
#             except Exception:
#                 pass

#         # 4-bit quantization (optional)
#         quant_cfg = None
#         if quant4bit:
#             quant_cfg = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_use_double_quant=True,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_compute_dtype=torch.float16,
#             )

#         # Prefer the newer class, fall back to Vision2Seq to match your env
#         self.model = None
#         try:
#             self.model = AutoModelForImageTextToText.from_pretrained(
#                 ckpt, trust_remote_code=True, config=cfg,
#                 quantization_config=quant_cfg, device_map="auto",
#             )
#         except Exception:
#             self.model = AutoModelForVision2Seq.from_pretrained(
#                 ckpt, trust_remote_code=True, config=cfg,
#                 quantization_config=quant_cfg, device_map="auto",
#             )

#         if hasattr(self.model, "tie_weights"):
#             try: self.model.tie_weights()
#             except Exception: pass
#         self.model.eval()

#         # Ensure pad/eos ids for clean generation (some checkpoints omit them)
#         try:
#             tok = getattr(self.proc, "tokenizer", None)
#             if tok is not None:
#                 if getattr(self.model.generation_config, "pad_token_id", None) is None and tok.pad_token_id is not None:
#                     self.model.generation_config.pad_token_id = tok.pad_token_id
#                 if getattr(self.model.generation_config, "eos_token_id", None) is None and tok.eos_token_id is not None:
#                     self.model.generation_config.eos_token_id = tok.eos_token_id
#         except Exception:
#             pass

#         # Precompute constrained token set for numeric-only fallback
#         self._num_token_ids = set()
#         try:
#             tok = getattr(self.proc, "tokenizer", None)
#             if tok is not None:
#                 self._num_token_ids = allowed_token_ids_for_numbers(tok)
#         except Exception:
#             self._num_token_ids = set()

#         # Optional warmup (ignore failures)
#         try:
#             _ = self.predict(np.zeros((360, 360, 3), dtype=np.uint8), "warmup")
#         except Exception:
#             pass

#     # --- constrained decoding hook (text fallback) ---
#     def _prefix_allowed_tokens_fn(self, _, __):
#         # Only allow numeric-ish tokens (+ EOS)
#         if self._num_token_ids:
#             return list(self._num_token_ids)
#         # If we failed to build a set, don't constrain
#         return None

    
#     @torch.no_grad()
#     def predict(self, rgb, instruction: str) -> np.ndarray:
#         """
#         Returns a 7-vector. Tries numeric action head first; falls back to constrained text parsing.
#         """
#         # Convert to PIL
#         if isinstance(rgb, np.ndarray):
#             if rgb.dtype != np.uint8:
#                 rgb = np.clip(rgb, 0, 255).astype(np.uint8)
#             pil_img = Image.fromarray(rgb[..., :3])
#         else:
#             pil_img = rgb

#         # Pack inputs
#         inputs = self.proc(images=[pil_img], text=instruction, return_tensors="pt")
#         dev = next(self.model.parameters()).device
#         inputs = {k: v.to(dev) for k, v in inputs.items()}

#         # ---- Preferred: numeric action head ----
#         if (not self.force_text) and hasattr(self.model, "predict_action"):
#             try:
#                 action = self.model.predict_action(
#                     **inputs,
#                     unnorm_key=self.unnorm_key,
#                     do_sample=False,
#                 )
#                 # Accept tensor / numpy / list
#                 if isinstance(action, torch.Tensor):
#                     arr = action.detach().float().cpu().numpy()
#                 elif isinstance(action, np.ndarray):
#                     arr = action
#                 else:
#                     # list / tuple / scalar -> to numpy
#                     arr = np.array(action, dtype=np.float32)

#                 # Expect (batch, 7) or (7,)
#                 if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] == 7:
#                     a7 = arr[0]
#                 elif arr.ndim == 1 and arr.shape[0] == 7:
#                     a7 = arr
#                 else:
#                     raise ValueError(f"predict_action returned shape {arr.shape}, expected (7,) or (B,7)")

#                 a7 = a7.astype(np.float32)
#                 # If this is already metric deltas due to unnorm_key, consider reducing step_scale in controller
#                 return a7
#             except Exception as e:
#                 if self.debug:
#                     print("[OpenVLA] predict_action failed, falling back to text:",
#                         type(e).__name__, str(e)[:160])

#         # ---- Fallback: constrained numeric-only text generation ----
#         # Short, single-turn prompt; keep order: Instruction then Action
#         user_prompt = f"Instruction: {instruction}\nOutput 7 numbers in [-1,1] as: dx,dy,dz,droll,dpitch,dyaw,grip.\nAction:"
#         inputs_txt = self.proc(images=[pil_img], text=user_prompt, return_tensors="pt")
#         inputs_txt = {k: v.to(dev) for k, v in inputs_txt.items()}

#         try:
#             out_ids = self.model.generate(
#                 **inputs_txt,
#                 # Constrain to numeric-ish tokens if we have the set
#                 prefix_allowed_tokens_fn=(self._prefix_allowed_tokens_fn if self._num_token_ids else None),
#                 max_new_tokens=24,
#                 do_sample=False,
#             )
#             try:
#                 text = self.proc.tokenizer.decode(out_ids[0], skip_special_tokens=True)
#             except Exception:
#                 text = self.model.generation_config._tokenizer.decode(out_ids[0], skip_special_tokens=True)

#             if self.debug:
#                 print("[OpenVLA raw]:", text)

#             a7 = parse_seven_floats(text)
#             if a7 is None or not np.isfinite(a7).all() or a7.shape != (7,):
#                 return np.array([0,0,0, 0,0,0, -1], dtype=np.float32)
#             return a7.astype(np.float32)

#         except Exception as e:
#             if self.debug:
#                 print("[OpenVLA] text fallback failed:", type(e).__name__, str(e)[:160])
#             return np.array([0,0,0, 0,0,0, -1], dtype=np.float32)



# # quick local smoke test (optional)
# if __name__ == "__main__":
#     vla = OpenVLA(debug=True, force_text=True)  # force text path for a quick check
#     fake = np.zeros((360, 360, 3), dtype=np.uint8)
#     out = vla.predict(fake, "Move above the red cube and close.")
#     print("a7:", out)
