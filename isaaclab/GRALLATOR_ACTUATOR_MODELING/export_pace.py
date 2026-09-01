#!/usr/bin/env python3
"""Convert a validated Grallator PACE CSV to chirp_data.pt."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pace_modeling.constants import JOINT_ORDER, PACE_EXPORT_ORDER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--output", default="chirp_data.pt")
    args = parser.parse_args()

    times = []
    q_actual = []
    q_des = []
    with Path(args.csv).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = ["time_s", "feedback_complete", "safety_event"]
        required += [f"{name}_q_actual" for name in JOINT_ORDER]
        required += [f"{name}_q_des" for name in JOINT_ORDER]
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"CSV is missing required fields: {missing}")
        for row_number, row in enumerate(reader, start=2):
            if str(row["feedback_complete"]).lower() not in ("true", "1"):
                raise SystemExit(f"row {row_number}: feedback is incomplete")
            if row["safety_event"].strip():
                raise SystemExit(f"row {row_number}: safety event: {row['safety_event']}")
            times.append(float(row["time_s"]))
            q_actual.append([
                float(row[f"{name}_q_actual"]) for name in PACE_EXPORT_ORDER
            ])
            q_des.append([
                float(row[f"{name}_q_des"]) for name in PACE_EXPORT_ORDER
            ])

    time_array = np.asarray(times, dtype=np.float32)
    actual_array = np.asarray(q_actual, dtype=np.float32)
    desired_array = np.asarray(q_des, dtype=np.float32)
    if time_array.ndim != 1 or actual_array.shape != desired_array.shape:
        raise SystemExit("invalid PACE array shapes")
    if actual_array.shape != (len(time_array), len(PACE_EXPORT_ORDER)):
        raise SystemExit(f"expected [N,12], got {actual_array.shape}")
    if not np.all(np.isfinite(time_array)) or not np.all(np.diff(time_array) > 0.0):
        raise SystemExit("time_s must be finite and strictly increasing")
    if not np.all(np.isfinite(actual_array)) or not np.all(np.isfinite(desired_array)):
        raise SystemExit("position arrays contain NaN or Inf")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required to write chirp_data.pt") from exc
    output = Path(args.output)
    torch.save({
        "time": torch.from_numpy(time_array),
        "des_dof_pos": torch.from_numpy(desired_array),
        "dof_pos": torch.from_numpy(actual_array),
    }, output)
    print(
        f"Saved {output}: time={time_array.shape}, dof_pos={actual_array.shape}, "
        f"PACE order={PACE_EXPORT_ORDER}"
    )


if __name__ == "__main__":
    main()
