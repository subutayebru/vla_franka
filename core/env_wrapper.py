# core/env_wrapper.py
#
# macOS (Apple Silicon) note:
# Run this with plain `python`, NOT `mjpython`. We do not use MuJoCo's
# interactive passive viewer (it requires mjpython on macOS, which forces the
# script onto a worker thread and makes any GUI window — e.g. the camera feed —
# crash). Instead we render two cameras off-screen and show them in a matplotlib
# "live view" window on the main thread:
#   - left  : the camera OpenVLA actually sees  (camera_name, e.g. standing_cam)
#   - right : a reference camera for context     (view_camera,  e.g. wrist cam)
# See docs/MACOS_PORT.md and docs/OPENVLA_PIPELINE.md.

import mujoco
from PIL import Image

from src.mujoco_parser import MuJoCoParserClass
from core.diagnostics_n_logging import init_live_view, update_live_view


class PandaEnv:
    def __init__(self, xml_path: str, camera_name: str,
                 view_camera: str = "panda_eye_in_hand", show_view: bool = True):
        self.camera_name = camera_name      # camera fed to the VLA
        self.view_camera = view_camera      # reference camera for the live window
        self.show_view = show_view

        # ---- MuJoCo env (no passive viewer) ----
        self.env = MuJoCoParserClass(name="Panda", rel_xml_path=xml_path, VERBOSE=False)
        self.env.forward()
        self.env.reset()

        # ---- Off-screen renderers: VLA-input cam + reference cam ----
        self.vla_renderer = mujoco.Renderer(self.env.model, 480, 640)
        self.view_renderer = mujoco.Renderer(self.env.model, 480, 640)

        self._last_vla = self._render(self.vla_renderer, self.camera_name)
        if self.show_view:
            view0 = self._render(self.view_renderer, self.view_camera)
            self.fig, self.axes, self.ims = init_live_view(
                self._last_vla, view0,
                title_left=f"what OpenVLA sees ({self.camera_name})",
                title_right=f"reference ({self.view_camera})",
            )

    @property
    def model(self):
        return self.env.model

    @property
    def data(self):
        return self.env.data

    def _render(self, renderer, camera):
        renderer.update_scene(self.env.data, camera)
        return renderer.render()

    def get_image(self):
        # The frame OpenVLA actually sees.
        self._last_vla = self._render(self.vla_renderer, self.camera_name)
        return Image.fromarray(self._last_vla)

    def step(self, torque):
        self.env.step(ctrl=torque, ctrl_idxs=self.env.ctrl_joint_idxs)

    def render(self):
        if not self.show_view:
            return
        view = self._render(self.view_renderer, self.view_camera)
        update_live_view(self.fig, self.ims, self._last_vla, view,
                         f"tick={self.env.tick}")

    def close(self):
        if self.show_view:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
