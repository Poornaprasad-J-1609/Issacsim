from pathlib import Path
import csv
from collections import defaultdict

CSV_DIR = Path("rl_log_exports")

KEYWORDS = [
    "mean_reward",
    "mean_episode_length",
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "dof_torques_l2",
    "dof_acc_l2",
    "action_rate_l2",
    "feet_air_time",
    "flat_orientation_l2",
    "error_vel_xy",
    "error_vel_yaw",
    "time_out",
    "base_contact",
]

data = defaultdict(lambda: defaultdict(list))

for csv_file in CSV_DIR.glob("*_scalars.csv"):
    with csv_file.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            run = row["run"]
            tag = row["tag"]
            step = int(float(row["step"]))
            value = float(row["value"])
            data[run][tag].append((step, value))

print("\n==============================")
print("RL LOG COMPARISON SUMMARY")
print("==============================")

for run in sorted(data.keys()):
    print(f"\n\nRUN: {run}")
    print("-" * 60)

    tags = sorted(data[run].keys())

    for key in KEYWORDS:
        matched = [t for t in tags if key.lower() in t.lower()]

        for tag in matched:
            values = sorted(data[run][tag], key=lambda x: x[0])
            if len(values) < 2:
                continue

            first_step, first_val = values[0]
            last_step, last_val = values[-1]

            mid_step, mid_val = values[len(values) // 2]
            min_step, min_val = min(values, key=lambda x: x[1])
            max_step, max_val = max(values, key=lambda x: x[1])

            print(f"\n{tag}")
            print(f"  first : step={first_step:8d}, value={first_val:+.4f}")
            print(f"  middle: step={mid_step:8d}, value={mid_val:+.4f}")
            print(f"  last  : step={last_step:8d}, value={last_val:+.4f}")
            print(f"  min   : step={min_step:8d}, value={min_val:+.4f}")
            print(f"  max   : step={max_step:8d}, value={max_val:+.4f}")
