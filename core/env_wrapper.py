# core/env_wrapper.py

import os
#os.environ["MUJOCO_GL"] = "egl"  # same behavior as original script

import mujoco
from PIL import Image

from src.mujoco_parser import MuJoCoParserClass
from core.diagnostics_n_logging import init_hud, update_hud


class PandaEnv:
    def __init__(self, xml_path: str, camera_name: str):
        self.camera_name = camera_name

        # ---- MuJoCo env ----
        self.env = MuJoCoParserClass(name="Panda", rel_xml_path=xml_path, VERBOSE=False)
        self.env.forward()

        self.env.init_viewer(
            viewer_title="VLA control",
            viewer_width=1600,
            viewer_height=900,
            viewer_hide_menus=False,
        )
        # original used cam_id = 0 for viewer
        self.env.update_viewer(cam_id=0)
        self.env.reset()

        # ---- Renderer + HUD ----
        self.renderer = mujoco.Renderer(self.env.model, 480, 640)
        self.renderer.update_scene(self.env.data, self.camera_name)
        initial_rgb = self.renderer.render()
        self.fig, self.ax, self.im = init_hud(initial_rgb)

    @property
    def model(self):
        return self.env.model

    @property
    def data(self):
        return self.env.data

    def get_image(self):
        self.renderer.update_scene(self.env.data, self.camera_name)
        rgb = self.renderer.render()
        # HUD title same style as original
        update_hud(self.ax, self.im, rgb, f"tick={self.env.tick}")
        return Image.fromarray(rgb)

    def step(self, torque):
        self.env.step(ctrl=torque, ctrl_idxs=self.env.ctrl_joint_idxs)

    def render(self):
        self.env.render()
