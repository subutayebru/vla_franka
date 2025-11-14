# core/vla_agent.py
import numpy as np
import inspect
from openvla_wrap import OpenVLA as _OpenVLA

class VLAAgent:
    def __init__(self, force_text=False, unnorm_key=None, debug=True, device="cuda", log_text_debug=False):
        # Build kwargs only if the user's OpenVLA supports them
        sig = inspect.signature(_OpenVLA.__init__)
        maybe = {}

        if "device" in sig.parameters:
            maybe["device"] = device
        if "debug" in sig.parameters:
            maybe["debug"] = debug
        if "force_text" in sig.parameters:
            maybe["force_text"] = force_text
        if "unnorm_key" in sig.parameters:
            maybe["unnorm_key"] = unnorm_key
        if "log_text_debug" in sig.parameters:               
            maybe["log_text_debug"] = log_text_debug        

        try:
            self.model = _OpenVLA(**maybe)
        except TypeError:
            # Fallback to minimal constructor, then set attrs if they exist
            self.model = _OpenVLA()
        # Best-effort: set attributes even if they weren't ctor args
        if hasattr(self.model, "force_text"):
            self.model.force_text = force_text
        if hasattr(self.model, "unnorm_key"):
            self.model.unnorm_key = unnorm_key
        self.debug = debug
        if hasattr(self.model, "log_text_debug"):             
            self.model.log_text_debug = log_text_debug       

    def act(self, rgb, instruction: str) -> np.ndarray:
        out = self.model.predict(rgb, instruction)
        if out is None or not np.isfinite(out).all() or getattr(out, "shape", (0,))[0] != 7:
            return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        return out.astype(np.float32)
