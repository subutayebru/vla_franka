# core/vla_agent.py

from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
import torch

from core.config import Cfg  # just for type hints


class OpenVLAAgent:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.processor = AutoProcessor.from_pretrained(
            cfg.vla_model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=self.bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    def _format_prompt(self) -> str:
        # This actually injects the instruction into the template
        # If you *really* want the old buggy behavior, remove `.format(...)`.
        return self.cfg.prompt_template.format(instruction=self.cfg.instruction)

    def act(self, image):
        prompt = self._format_prompt()
        inputs = self.processor(prompt, image)

        vision_mod = self.model.vision_backbone
        vision_param = next(vision_mod.parameters())
        img_dtype = vision_param.dtype
        img_device = vision_param.device

        inputs["pixel_values"] = inputs["pixel_values"].to(
            device=img_device, dtype=img_dtype
        )

        with torch.inference_mode():
            action = self.model.predict_action(
                **inputs,
                unnorm_key=self.cfg.vla_unnorm_key,
                do_sample=False,
            )

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        return action
