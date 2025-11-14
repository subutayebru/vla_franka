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



