from pathlib import Path
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN_DIR = Path("logs/rsl_rl/grallator_flat/2026-05-16_11-39-45")
OUT = Path("rl_log_exports/grallator_flat_new_actuator_scalars.csv")
OUT.parent.mkdir(exist_ok=True)

ea = EventAccumulator(str(RUN_DIR))
ea.Reload()

tags = ea.Tags().get("scalars", [])

with OUT.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run", "tag", "step", "value", "wall_time"])

    for tag in tags:
        for e in ea.Scalars(tag):
            writer.writerow(["grallator_flat_new_actuator", tag, e.step, e.value, e.wall_time])

print("Exported:", OUT)
print("Scalar tags:", len(tags))
