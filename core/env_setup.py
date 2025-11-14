#env_setup.py
import mujoco, numpy as np
from src.mujoco_parser import MuJoCoParserClass

EE_SITE="grip_site"; EEF_BODY="panda_eef"; OBJ_BODY="obj_box_06"; PLT_BODY="object table"

def create_env(camera_name: str):
    env = MuJoCoParserClass(name="Panda", rel_xml_path="asset/panda/franka_panda_w_objs.xml", VERBOSE=False)
    env.forward()
    env.init_viewer(viewer_title="PNP + VLA", viewer_width=1600, viewer_height=900, viewer_hide_menus=False)
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    assert cam_id != -1, f"Camera '{camera_name}' not found."
    env.update_viewer(cam_id=cam_id)
    env.reset()
    return env, cam_id

def get_ids(env):
    m=env.model
    ee_sid      = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    eef_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, EEF_BODY)
    obj_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJ_BODY)
    plt_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PLT_BODY)
    assert min(ee_sid,eef_body_id,obj_body_id,plt_body_id) != -1, "Bad names"
    return ee_sid, eef_body_id, obj_body_id, plt_body_id

def print_ik_map(env):
    m = env.model
    ctrl_jids = list(np.array(env.ctrl_joint_idxs, dtype=int))
    arm_jids  = ctrl_jids[:7] if len(ctrl_jids) >= 7 else ctrl_jids
    def dof_count(jid):
        jt = int(m.jnt_type[jid])
        return 6 if jt==mujoco.mjtJoint.mjJNT_FREE else 3 if jt==mujoco.mjtJoint.mjJNT_BALL else 1
    arm_dofs=[]
    for jid in arm_jids:
        adr=int(m.jnt_dofadr[jid]); arm_dofs.extend(range(adr, adr+dof_count(jid)))
    arm_qpos=[int(m.jnt_qposadr[j]) for j in arm_jids]
    print("[IK MAP] arm_jids      :", arm_jids)
    print("[IK MAP] arm_dofs      :", arm_dofs)
    print("[IK MAP] arm_qpos_adrs :", arm_qpos)
