from pathlib import Path
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = {
    "anymal_c_flat": "logs/rsl_rl/anymal_c_flat/2026-05-16_08-04-06",
    "unitree_go2_flat": "logs/rsl_rl/unitree_go2_flat/2026-05-16_07-48-26",
    "grallator_flat": "logs/rsl_rl/grallator_flat/2026-05-15_12-58-12",
}

OUT_DIR = Path("rl_log_exports")
OUT_DIR.mkdir(exist_ok=True)

for run_name, run_dir in RUNS.items():
    run_path = Path(run_dir)
    event_files = list(run_path.glob("events.out.tfevents.*"))

    if not event_files:
        print(f"[MISSING] No event file found in {run_dir}")
        continue

    print(f"\nExporting: {run_name}")
    print(f"Folder: {run_dir}")

    ea = EventAccumulator(str(run_path))
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])
    print(f"Scalar tags found: {len(scalar_tags)}")

    out_csv = OUT_DIR / f"{run_name}_scalars.csv"

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "tag", "step", "value", "wall_time"])

        for tag in scalar_tags:
            events = ea.Scalars(tag)
            for e in events:
                writer.writerow([run_name, tag, e.step, e.value, e.wall_time])

    print(f"Saved: {out_csv}")

print("\nDone.")
