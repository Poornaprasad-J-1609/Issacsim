from pathlib import Path
import csv
from collections import defaultdict

FILES = {
    "old_grallator": "rl_log_exports/grallator_flat_scalars.csv",
    "new_grallator": "rl_log_exports/grallator_flat_new_actuator_scalars.csv",
}

KEYS = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Reward/track_lin_vel_xy_exp",
    "Episode_Reward/lin_vel_z_l2",
    "Episode_Reward/dof_torques_l2",
    "Episode_Reward/dof_acc_l2",
    "Episode_Reward/action_rate_l2",
    "Episode_Reward/flat_orientation_l2",
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
    "Episode_Termination/time_out",
    "Episode_Termination/base_contact",
]

data = defaultdict(lambda: defaultdict(list))

for run, file in FILES.items():
    with open(file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row["tag"]
            step = int(float(row["step"]))
            val = float(row["value"])
            data[run][tag].append((step, val))

print("\n==============================")
print("GRALLATOR OLD vs NEW ACTUATOR")
print("==============================")

for key in KEYS:
    print(f"\n{key}")
    print("-" * 60)

    for run in ["old_grallator", "new_grallator"]:
        vals = data[run].get(key, [])

        if not vals:
            print(f"{run:15s}: NOT FOUND")
            continue

        vals = sorted(vals, key=lambda x: x[0])
        first = vals[0]
        mid = vals[len(vals)//2]
        last = vals[-1]
        min_v = min(vals, key=lambda x: x[1])
        max_v = max(vals, key=lambda x: x[1])

        print(
            f"{run:15s}: "
            f"first={first[1]:+8.4f} | "
            f"mid={mid[1]:+8.4f} | "
            f"last={last[1]:+8.4f} | "
            f"min={min_v[1]:+8.4f} | "
            f"max={max_v[1]:+8.4f}"
        )
