# core/config.py

from dataclasses import dataclass
from typing import Tuple
import yaml, argparse


@dataclass
class PIDCfg:
    kp: float
    ki: float
    kd: float


@dataclass
class StepsCfg:
    step_scale_xyz: float
    step_scale_rpy: float


@dataclass
class Cfg:
    camera_name: str
    xml_path: str
    startup_ignore_ticks: int
    phase_agnostic: bool
    show_vla_feed: bool
    overlay_hud: bool
    ablate_vla: bool

    vla_model_name: str
    vla_unnorm_key: str | None

    instruction: str
    prompt_template: str

    max_tick: int
    policy_hz: int
    sim_hz: int

    pid: PIDCfg
    workspace: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    steps: StepsCfg


def load_cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--force_text", action="store_true")
    ap.add_argument("--unnorm_key", default=None)
    ap.add_argument("--prompt", default=None, type=str)  # << your new flag
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        y = yaml.safe_load(f)

    # allow CLI override of unnorm_key if provided
    vla_unnorm_key = args.unnorm_key or y.get("vla_unnorm_key")

    # allow CLI override of instruction / prompt via --prompt
    instruction = args.prompt or y["instruction"]

    ws = y["workspace"]

    cfg = Cfg(
        camera_name=y["camera_name"],
        xml_path=y["xml_path"],
        startup_ignore_ticks=int(y["startup_ignore_ticks"]),
        phase_agnostic=bool(y["phase_agnostic"]),
        show_vla_feed=bool(y["show_vla_feed"]),
        overlay_hud=bool(y["overlay_hud"]),
        ablate_vla=bool(y["ablate_vla"]),
        vla_model_name=y["vla_model_name"],
        vla_unnorm_key=vla_unnorm_key,
        instruction=instruction,
        prompt_template=y["prompt_template"],
        max_tick=int(y["max_tick"]),
        policy_hz=int(y["policy_hz"]),
        sim_hz=int(y["sim_hz"]),
        pid=PIDCfg(**y["pid"]),
        workspace=(tuple(ws["x"]), tuple(ws["y"]), tuple(ws["z"])),
        steps=StepsCfg(**y["steps"]),
    )

    return cfg, args
