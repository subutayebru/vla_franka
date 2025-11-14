#plot_vla_logs.py 
import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    fn = sys.argv[1] if len(sys.argv) > 1 else "vla_logs.csv"
    df = pd.read_csv(fn)

    t = df["time"]

    # ---- Actions: act_0..act_6 ----
    plt.figure()
    for i in range(7):
        plt.plot(t, df[f"act_{i}"], label=f"act_{i}")
    plt.xlabel("time [s]")
    plt.ylabel("VLA action values")
    plt.title("VLA actions (act_0..act_6)")
    plt.legend()
    plt.tight_layout()

    # ---- Arm joints: q_0..q_6 ----
    plt.figure()
    for i in range(7):
        plt.plot(t, df[f"q_{i}"], label=f"q_{i}")
    plt.xlabel("time [s]")
    plt.ylabel("joint angle [rad]")
    plt.title("Arm joints (q_0..q_6)")
    plt.legend()
    plt.tight_layout()

    # ---- Gripper joints: q_7, q_8 ----
    if "q_7" in df.columns and "q_8" in df.columns:
        plt.figure()
        plt.plot(t, df["q_7"], label="q_7 (gripper left)")
        plt.plot(t, df["q_8"], label="q_8 (gripper right)")
        plt.xlabel("time [s]")
        plt.ylabel("joint angle [rad]")
        plt.title("Gripper joints (q_7, q_8)")
        plt.legend()
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
