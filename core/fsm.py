import enum, numpy as np
class Phase(enum.Enum):
    SEARCH=0; APPROACH=1; GRASP=2; LIFT=3; MOVE_TO_PLACE=4; RELEASE=5; RETREAT=6; RUN=7

def decide_instruction_and_mode(phase, PHASE_AGNOSTIC, p_ee, p_obj, p_plt, z_grasp, z_lift, z_place, env):
    instr = "Pick up the red cube and place it on top of the black platform"
    env._mode, env._fine_mode = "", False
    if PHASE_AGNOSTIC:
        return "Pick up the red cube and place it on the black platform", phase

    if phase == Phase.SEARCH:
        instr = "Move above the red cube"; env._mode, env._fine_mode = "APPROACH_OBJ", False
        if np.linalg.norm(p_ee[:2]-p_obj[:2]) < 0.04 and p_ee[2] > z_lift-0.02:
            phase = Phase.APPROACH

    elif phase == Phase.APPROACH:
        instr = "Descend to grasp height over the red cube"; env._mode, env._fine_mode = "APPROACH_OBJ", True
        if np.linalg.norm(p_ee[:2]-p_obj[:2]) < 0.02 and p_ee[2] <= z_grasp+0.01:
            phase = Phase.GRASP

    elif phase == Phase.GRASP:
        env._mode, env._fine_mode = "", False

    elif phase == Phase.LIFT:
        instr = "Lift up"; env._mode, env._fine_mode = "", False
        if p_ee[2] >= z_lift: phase = Phase.MOVE_TO_PLACE

    elif phase == Phase.MOVE_TO_PLACE:
        instr = "Move above the black platform and descend"; env._mode, env._fine_mode = "APPROACH_PLT", True
        if np.linalg.norm(p_ee[:2]-p_plt[:2]) < 0.03 and p_ee[2] <= z_place+0.02:
            phase = Phase.RELEASE

    elif phase == Phase.RELEASE:
        env._mode, env._fine_mode = "", False

    elif phase == Phase.RETREAT:
        instr = "Move up to a safe height"; env._mode, env._fine_mode = "", False

    return instr, phase
