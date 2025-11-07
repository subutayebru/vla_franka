# vla_openvla.py
import numpy as np
from PIL import Image                          # <-- add this
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

class OpenVLA:
    def __init__(self, ckpt="openvla/openvla-v01-7b", device="cuda"):
        self.proc  = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            ckpt, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        self.device = device

    def predict(self, rgb, instruction: str):
        # rgb is HxWx3 uint8 from MuJoCo; convert to PIL for the processor
        if isinstance(rgb, np.ndarray):
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(rgb[..., :3])    # ensure 3-channel RGB
        else:
            pil_img = rgb

        # Processor expects an iterable of images
        inp = self.proc(images=[pil_img], text=instruction, return_tensors="pt")
        inp = {k: v.to(self.model.device) for k, v in inp.items()}

        # If you haven't implemented token→action decoding yet, you can skip generate()
        # with torch.no_grad():
        #     out = self.model.generate(**inp)
        # a7 = decode_tokens_to_action(out)  # -> np.ndarray shape (7,), in [-1, 1]
        # return a7.astype(np.float32)

        # TEMP: keep a small nudge so the loop runs end-to-end
        return np.array([0.03, 0.0, 0.0, 0, 0, 0, 0], dtype=np.float32)
