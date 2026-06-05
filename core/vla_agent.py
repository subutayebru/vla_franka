# core/vla_agent.py
#
# Apple Silicon (MPS) port.
#
# This branch loads OpenVLA-7B in fp16 directly onto the Mac GPU (MPS) with no
# bitsandbytes 4-bit quantization and no device_map="auto" (both are CUDA/Linux
# only). See docs/MACOS_PORT.md for the full rationale.

import torch
import transformers
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig

from core.config import Cfg  # just for type hints


# --- Compatibility shims for OpenVLA's trust_remote_code modeling on MPS ---
# Some older remote model implementations don't define this flag that newer HF
# checks; defining it avoids an attribute error during loading.
if not hasattr(transformers.modeling_utils.PreTrainedModel, "_supports_sdpa"):
    transformers.modeling_utils.PreTrainedModel._supports_sdpa = False


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OpenVLAAgent:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.device = _pick_device()

        self.processor = AutoProcessor.from_pretrained(
            cfg.vla_model_name,
            trust_remote_code=True,
        )

        # Force eager attention (avoids flash-attn / SDPA paths that are not
        # available or misbehave on MPS).
        model_config = AutoConfig.from_pretrained(
            cfg.vla_model_name,
            trust_remote_code=True,
        )
        for attr in ("attn_implementation", "_attn_implementation"):
            try:
                setattr(model_config, attr, "eager")
            except Exception:
                pass

        self.model = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_model_name,
            config=model_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

        # Ensure pad/eos ids are set for clean generation (some checkpoints omit
        # them, which otherwise emits warnings on every step).
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None:
            gen = self.model.generation_config
            if getattr(gen, "pad_token_id", None) is None and tok.pad_token_id is not None:
                gen.pad_token_id = tok.pad_token_id
            if getattr(gen, "eos_token_id", None) is None and tok.eos_token_id is not None:
                gen.eos_token_id = tok.eos_token_id

    def _format_prompt(self) -> str:
        # This actually injects the instruction into the template
        # If you *really* want the old buggy behavior, remove `.format(...)`.
        return self.cfg.prompt_template.format(instruction=self.cfg.instruction)

    def act(self, image):
        prompt = self._format_prompt()
        inputs = self.processor(prompt, image)

        # Drop the processor's attention_mask. predict_action() appends a special
        # token (29871) to input_ids but does NOT extend a caller-supplied mask,
        # so the mask ends up one shorter than the sequence. In the multimodal
        # forward the 256 image-patch tokens are spliced into both embeds and
        # mask, turning that off-by-one into a hard shape mismatch
        # (`attn_weights + causal_mask`) on the eager attention path we use for
        # MPS. With no mask passed, generate() rebuilds an all-ones mask at the
        # correct post-append length. (On CUDA/SDPA the stale mask is silently
        # discarded, which is why upstream never hit this.)
        inputs.pop("attention_mask", None)

        # Move every tensor to the compute device. pixel_values must match the
        # vision backbone dtype (fp16); ids stay integer.
        vision_param = next(self.model.vision_backbone.parameters())
        img_dtype = vision_param.dtype

        for key, val in inputs.items():
            if not isinstance(val, torch.Tensor):
                continue
            if key == "pixel_values":
                inputs[key] = val.to(device=self.device, dtype=img_dtype)
            else:
                inputs[key] = val.to(device=self.device)

        with torch.inference_mode():
            action = self.model.predict_action(
                **inputs,
                unnorm_key=self.cfg.vla_unnorm_key,
                do_sample=False,
            )

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        return action
