import numpy as np
import matplotlib.pyplot as plt
import csv, time
import os

# Diagnostics # 

def occlude_flip_alt_probes(env, renderer, cam_id, obj_body_id, vla_agent, instr, a7_now):
    from numpy import where
    m=env.model; d=env.data
    # occlude cube magenta
    geom_ids = where(m.geom_bodyid == obj_body_id)[0]
    if geom_ids.size:
        orig = m.geom_rgba.copy()
        for gid in geom_ids: m.geom_rgba[gid,:] = (1,0,1,1)
        renderer.update_scene(d, camera=cam_id); rgb_mask = renderer.render()
        for gid in geom_ids: m.geom_rgba[gid,:] = orig[gid,:]
        a7_mask = vla_agent.act(rgb_mask, instr) - a7_now
        print("[SENSE] occlude cube Δa7 =", np.round(a7_mask,4))

    # flip
    renderer.update_scene(d, camera=cam_id); rgb = renderer.render()
    a7_flip = vla_agent.act(rgb[:, ::-1].copy(), instr) - a7_now
    print("[SENSE] flip image Δa7 =", np.round(a7_flip,4))

    # alt instruction
    a7_alt = vla_agent.act(rgb, "Do nothing.") - a7_now
    print("[SENSE] alt instruction Δa7 =", np.round(a7_alt,4))

def progress_watchdog(state, env, p_ee, p_obj, tick_stride=200, stalls_to_fallback=10):
    if state["last_prog_check_tick"] is None:
        state["last_prog_check_tick"] = env.tick
        state["last_d_obj"] = float(np.linalg.norm(p_ee[:2]-p_obj[:2]))
        return False
    if env.tick - state["last_prog_check_tick"] >= tick_stride:
        d_now = float(np.linalg.norm(p_ee[:2]-p_obj[:2]))
        if d_now < (state["last_d_obj"] - 0.01): state["stalled"]=0
        else: state["stalled"] += 1
        state["last_d_obj"]=d_now; state["last_prog_check_tick"]=env.tick
    return state["stalled"] >= stalls_to_fallback

# HUD #

def init_hud(initial_rgb):
    plt.ion()
    fig, ax = plt.subplots(num="VLA feed", figsize=(6.4, 4.8))
    im = ax.imshow(initial_rgb); ax.axis("off")
    fig.canvas.draw(); fig.canvas.flush_events()
    return fig, ax, im

def update_hud(ax, im, frame, title):
    im.set_data(frame); ax.set_title(title, fontsize=9)
    im.figure.canvas.draw(); im.figure.canvas.flush_events()


# Logging #

def make_logger(path=None):
    path = path or f"vla_logs/vla_logs_{int(time.time())}.csv"
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["tick","time","phase","dx","dy","dz","droll","dpitch","dyaw","grip",
                "ee_x","ee_y","ee_z","obj_x","obj_y","obj_z","plt_x","plt_y","plt_z"])
    return f, w, path

def make_vla_logger(n_actions=7, n_joints=9, path=None):
    """
    Create a CSV logger for VLA actions and joint positions.
    Columns: tick, time, act_0..act_{n_actions-1}, q_0..q_{n_joints-1}
    """
    path = path or f"vla_logs/vla_logs_{int(time.time())}.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    f = open(path, "w", newline="")
    w = csv.writer(f)

    header = ["tick", "time"]
    header += [f"act_{i}" for i in range(n_actions)]
    header += [f"q_{i}"   for i in range(n_joints)]
    w.writerow(header)

    return f, w, path

def make_control_logger(n_actions=7, n_joints=9, path=None):
    """
    Log VLA actions, joint states, desired joint states, and torques.
    Saved under logs_control/.
    """
    path = path or f"logs_control/logs_control_{int(time.time())}.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    f = open(path, "w", newline="")
    w = csv.writer(f)
    print("Logging control data to:", path)
    header = ["tick", "time"]
    header += [f"act_{i}" for i in range(n_actions)]
    header += [f"q_{i}" for i in range(n_joints)]
    header += [f"q_des_{i}" for i in range(n_joints)]
    header += [f"tau_{i}" for i in range(n_joints)]
    w.writerow(header)

    return f, w, path

def init_control_logger(n_actions, n_joints):
    log_f, log_w, log_path = make_control_logger(
        n_actions=n_actions,
        n_joints=n_joints,
    )
    
    return log_f, log_w, log_path