from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
import torch
import mujoco
from src.mujoco_parser import MuJoCoParserClass

model = mujoco.MjModel.from_xml_path("/home/es_admin/vla-franka/Simple-MuJoCo-PickNPlace/asset/panda/franka_panda_w_objs.xml")
data = mujoco.MjData(model)

# Offscreen renderer (no GLFW)
renderer = mujoco.Renderer(model, 480, 640) 
for _ in range(100):
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    rgb = renderer.render()


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,   # keep compute in fp16 on 12GB cards
)


processor = AutoProcessor.from_pretrained("openvla/openvla-v01-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-v01-7b",
    #attn_implementation="flash_attention_2",  # optional
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    #device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)#.to("cuda:0")

image = Image.open("god_mode.png").convert("RGB")
prompt = "In: What action should the robot take to pick up the cube and place it on the black platform?\nOut:"

inputs = processor(prompt, image)#.to("cuda:0", dtype=torch.bfloat16)

vision_mod = vla.vision_backbone
vision_param = next(vision_mod.parameters())
img_dtype = vision_param.dtype
img_device = vision_param.device

inputs["pixel_values"] = inputs["pixel_values"].to(device=img_device, dtype=img_dtype)

action = vla.predict_action(**inputs, unnorm_key='nyu_franka_play_dataset_converted_externally_to_rlds', do_sample=False)  # start with None
print('Action:', action)  # 7 floats (normalized deltas)
print('Action shape:', action.shape)  
