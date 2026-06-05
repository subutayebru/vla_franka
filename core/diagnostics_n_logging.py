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


# Live view (two panels: scene overview + wrist cam that OpenVLA sees) #
# Runs on the main thread with an interactive backend; used by the macOS
# plain-`python` run path (no mjpython / no MuJoCo passive viewer).

def init_live_view(left_rgb, right_rgb,
                   title_left="left", title_right="right"):
    plt.ion()
    fig, axes = plt.subplots(1, 2, num="VLA live view", figsize=(12.8, 4.8))
    ims = []
    for ax, img, title in zip(axes, (left_rgb, right_rgb), (title_left, title_right)):
        im = ax.imshow(img); ax.axis("off"); ax.set_title(title, fontsize=10)
        ims.append(im)
    fig.tight_layout()
    fig.canvas.draw(); fig.canvas.flush_events()
    plt.show(block=False)
    return fig, axes, ims

def update_live_view(fig, ims, scene_rgb, wrist_rgb, suptitle=None):
    ims[0].set_data(scene_rgb)
    ims[1].set_data(wrist_rgb)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=10)
    fig.canvas.draw_idle(); fig.canvas.flush_events()
    plt.pause(0.001)  # let the GUI event loop process & repaint


# Human-readable action / behavior report (console) #

def action_report(tick, action, ee_pos, target_pos, target_name="target",
                  prev_ee_pos=None, prev_dist=None):
    """
    Turn one VLA step into a legible line so you can tell if the arm behaves:
    what the model commands (direction + gripper), where the end-effector is
    relative to the target, whether that distance is shrinking, and whether the
    arm actually moved. Returns (text, dist) so the caller can carry `dist`
    forward as next step's `prev_dist`.
    """
    a = np.asarray(action, dtype=float).reshape(-1)
    dx, dy, dz, dr, dp, dyaw, grip = a
    thr = 0.05  # below this a component is treated as "no command" (·)

    def sym(v, pos, neg):
        return pos if v > thr else (neg if v < -thr else "·")

    move = f"x{sym(dx, '+', '−')} y{sym(dy, '+', '−')} z{sym(dz, '↑', '↓')}"
    rot_mag = max(abs(dr), abs(dp), abs(dyaw))
    rot = "~0" if rot_mag < thr else f"r{sym(dr,'+','−')} p{sym(dp,'+','−')} w{sym(dyaw,'+','−')}"
    gripper = "CLOSE" if grip > 0 else "OPEN"

    ee = np.asarray(ee_pos, dtype=float)
    tgt = np.asarray(target_pos, dtype=float)
    dist = float(np.linalg.norm(ee - tgt))

    if prev_dist is None:
        trend = " ·"
    else:
        d = dist - prev_dist
        arrow = "↓" if d < -1e-4 else ("↑" if d > 1e-4 else "→")
        trend = f"{arrow}{abs(d):.3f}"

    if prev_ee_pos is None:
        step = "  · "
    else:
        step_m = float(np.linalg.norm(ee - np.asarray(prev_ee_pos, dtype=float)))
        step = f"{step_m * 1000:4.0f}mm"

    raw = " ".join(f"{v:+.2f}" for v in a)
    line = (f"[t={tick:>5}] [{raw}]\n"
            f"          move: {move} | rot: {rot} | grip: {gripper} | "
            f"EE→{target_name}: {dist:.3f}m {trend} | EE step: {step}")
    return line, dist


# Live action graph (separate window) #
# A watchable sanity-check plot updated each policy step. Top: EE→target
# distance over time (should trend down if the policy works). Bottom: the
# commanded dx/dy/dz + gripper over time (what the model is doing). Runs on the
# main thread with an interactive backend — plain `python`, no mjpython.

class ActionMonitor:
    def __init__(self, target_name="target"):
        self.ticks = []
        self.dist_hist = []
        self.comp_hist = {"dx": [], "dy": [], "dz": [], "grip": []}

        plt.ion()
        self.fig, (self.ax_d, self.ax_a) = plt.subplots(
            2, 1, num="VLA action sanity check", figsize=(7.0, 6.0), sharex=True
        )

        (self.dist_line,) = self.ax_d.plot([], [], color="tab:red", lw=1.8)
        self.ax_d.set_ylabel(f"EE→{target_name} (m)")
        self.ax_d.set_title("distance to target (want ↓)", fontsize=10)
        self.ax_d.grid(True, alpha=0.3)

        self.comp_lines = {}
        for name, color in (("dx", "tab:blue"), ("dy", "tab:green"),
                            ("dz", "tab:orange"), ("grip", "tab:purple")):
            (self.comp_lines[name],) = self.ax_a.plot([], [], label=name, color=color, lw=1.4)
        self.ax_a.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        self.ax_a.set_ylabel("action")
        self.ax_a.set_xlabel("tick")
        self.ax_a.set_title("commanded action components", fontsize=10)
        self.ax_a.legend(loc="upper right", ncol=4, fontsize=8)
        self.ax_a.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.fig.canvas.draw(); self.fig.canvas.flush_events()
        plt.show(block=False)

    def update(self, tick, dist, action):
        import numpy as np
        a = np.asarray(action, dtype=float).reshape(-1)
        self.ticks.append(tick)
        self.dist_hist.append(dist)
        self.comp_hist["dx"].append(a[0])
        self.comp_hist["dy"].append(a[1])
        self.comp_hist["dz"].append(a[2])
        self.comp_hist["grip"].append(a[6])

        self.dist_line.set_data(self.ticks, self.dist_hist)
        for name, line in self.comp_lines.items():
            line.set_data(self.ticks, self.comp_hist[name])

        for ax in (self.ax_d, self.ax_a):
            ax.relim(); ax.autoscale_view()
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)


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