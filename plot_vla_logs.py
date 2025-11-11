# plot_vla_logs.py
import sys, pandas as pd
import matplotlib.pyplot as plt

fn = sys.argv[1] if len(sys.argv) > 1 else "vla_logs.csv"
df = pd.read_csv(fn)

# EE pose vs time
plt.figure()
plt.plot(df["time"], df["ee_z"], label="ee_z")
plt.plot(df["time"], df["ee_x"], label="ee_x")
plt.plot(df["time"], df["ee_y"], label="ee_y")
plt.xlabel("time [s]"); plt.ylabel("EE pose [m]"); plt.legend(); plt.title("End-effector pose")
plt.tight_layout()

# VLA actions (dx,dy,dz,grip)
plt.figure()
plt.plot(df["time"], df["dx"], label="dx")
plt.plot(df["time"], df["dy"], label="dy")
plt.plot(df["time"], df["dz"], label="dz")
plt.xlabel("time [s]"); plt.ylabel("normalized deltas [-1,1]"); plt.legend(); plt.title("VLA action (xyz)")
plt.tight_layout()

plt.figure()
plt.plot(df["time"], df["grip"])
plt.xlabel("time [s]"); plt.ylabel("grip (>-0 open, >0 close)"); plt.title("VLA grip")
plt.tight_layout()

plt.show()
