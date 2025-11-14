#plot_control_logs.py
import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    fn = sys.argv[1] if len(sys.argv) > 1 else "control_logs.csv"
    df = pd.read_csv(fn)

    t = df["time"]

    # ---- Arm joints: actual vs desired (q_0..q_6, q_des_0..q_des_6) ----
    # Plot in a small loop so you can see tracking behaviour
    for j in range(7):  # arm joints
        q_col = f"q_{j}"
        qd_col = f"q_des_{j}"
        if q_col not in df.columns or qd_col not in df.columns:
            continue

        plt.figure()
        plt.plot(t, df[q_col], label=f"{q_col} (actual)")
        plt.plot(t, df[qd_col], linestyle="--", label=f"{qd_col} (desired)")
        plt.xlabel("time [s]")
        plt.ylabel("joint angle [rad]")
        plt.title(f"Joint {j}: actual vs desired")
        plt.legend()
        plt.tight_layout()

    # ---- Torques: tau_0..tau_6 ----
    plt.figure()
    for j in range(7):  # arm joints only
        col = f"tau_{j}"
        if col in df.columns:
            plt.plot(t, df[col], label=col)
    plt.xlabel("time [s]")
    plt.ylabel("torque")
    plt.title("Joint torques (tau_0..tau_6)")
    plt.legend()
    plt.tight_layout()

    # ---- Gripper: q_7,q_8 and tau_7,tau_8, if present ----
    if "q_7" in df.columns and "q_8" in df.columns:
        plt.figure()
        plt.plot(t, df["q_7"], label="q_7 (gripper left)")
        plt.plot(t, df["q_8"], label="q_8 (gripper right)")
        plt.xlabel("time [s]")
        plt.ylabel("joint angle [rad]")
        plt.title("Gripper joints (q_7, q_8)")
        plt.legend()
        plt.tight_layout()

    if "tau_7" in df.columns and "tau_8" in df.columns:
        plt.figure()
        plt.plot(t, df["tau_7"], label="tau_7 (gripper left)")
        plt.plot(t, df["tau_8"], label="tau_8 (gripper right)")
        plt.xlabel("time [s]")
        plt.ylabel("torque")
        plt.title("Gripper torques (tau_7, tau_8)")
        plt.legend()
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
