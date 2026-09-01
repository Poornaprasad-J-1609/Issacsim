"""Command-line entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .constants import JOINT_COUNT
from .controller import load_yaml, run_dry, run_hardware
from .trajectory import joint_vector


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_deploy_root():
    configured = os.environ.get("GRALLATOR_DEPLOY_ROOT")
    if configured:
        return Path(configured)
    candidates = [
        PROJECT_ROOT.parent / "GRALLATOR_DEPLOY",
        PROJECT_ROOT.parent,
    ]
    for candidate in candidates:
        if (candidate / "src" / "motor_command_layer.py").exists():
            return candidate
    return candidates[0]


def parser():
    result = argparse.ArgumentParser(
        description="Grallator 50 Hz PACE actuator-modeling logger/controller"
    )
    result.add_argument(
        "--config", default=str(PROJECT_ROOT / "config" / "pace_config.yaml")
    )
    result.add_argument("--trajectory", required=True)
    result.add_argument("--dataset", default="dataset_A")
    result.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "actuator_modeling_logs"),
    )
    result.add_argument("--deploy-root", default=str(default_deploy_root()))
    result.add_argument("--can-front", default="slcan0")
    result.add_argument("--can-back", default="slcan1")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "--initial-q",
        nargs=JOINT_COUNT,
        type=float,
        default=None,
        metavar="Q",
        help=(
            "12 logical initial angles used only by --dry-run; defaults to "
            "trajectory dry_run_initial_q, then zeros"
        ),
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_yaml(args.config)
    configured_hz = float(config.get("control_hz", 0.0))
    if abs(configured_hz - 50.0) > 1.0e-9:
        raise SystemExit(f"ERROR: PACE control_hz must be exactly 50, got {configured_hz}")
    if args.dry_run:
        trajectory_spec = load_yaml(args.trajectory)
        if args.initial_q is not None:
            initial_q = np.asarray(args.initial_q, dtype=np.float64)
        else:
            initial_q = joint_vector(
                trajectory_spec.get("dry_run_initial_q"),
                field="dry_run_initial_q",
            )
        csv_path, count = run_dry(
            config, args.config, args.trajectory, args.output_root, args.dataset,
            initial_q,
        )
    else:
        csv_path, count = run_hardware(
            config, args.config, args.trajectory, args.output_root, args.dataset,
            args.deploy_root, args.can_front, args.can_back,
        )
    print(f"PACE dataset saved: {csv_path}")
    print(f"Samples: {count}")
    return 0
