from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
import torch
import mujoco
import numpy as np

######################
# 1) MuJoCo setup
######################
model = mujoco.MjModel.from_xml_path(
    "/home/es_admin/vla-franka/Simple-MuJoCo-PickNPlace/asset/panda/franka_panda_w_objs.xml"
)
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, 480, 640)

# Step sim a bit to settle
for _ in range(100):
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    rgb = renderer.render()     # shape (480, 640, 3), uint8

# Convert last frame to PIL image
image = Image.fromarray(rgb)

######################
# 2) OpenVLA setup
######################
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(
    "openvla/openvla-v01-7b",
    trust_remote_code=True,
)

vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-v01-7b",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

######################
# 3) Prompt + processing
######################
instruction = "pick up the cube and place it on the black platform"
prompt = f"What action should the robot take to {instruction}?"

inputs = processor(prompt, image)

vision_mod = vla.vision_backbone
vision_param = next(vision_mod.parameters())
img_dtype = vision_param.dtype
img_device = vision_param.device
inputs["pixel_values"] = inputs["pixel_values"].to(device=img_device, dtype=img_dtype)

######################
# 4) Get action & inspect components
######################
with torch.inference_mode():
    action = vla.predict_action(
        **inputs,
        unnorm_key='nyu_franka_play_dataset_converted_externally_to_rlds',
        do_sample=False,
    )

print("Full action:", action)
dx, dy, dz, droll, dpitch, dyaw, dgrip = action
print(f"Δpos:   dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
print(f"Δrot:   droll={droll:.4f}, dpitch={dpitch:.4f}, dyaw={dyaw:.4f}")
print(f"grip:   {dgrip:.4f}")