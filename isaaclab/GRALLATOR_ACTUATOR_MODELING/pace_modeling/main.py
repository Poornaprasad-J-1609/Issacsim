"""Command-line entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .constants import JOINT_COUNT, JOINT_ORDER
from .controller import load_yaml, run_dry, run_hardware, run_timing_validation
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
        description="Grallator configured-rate PACE actuator-modeling logger/controller"
    )
    result.add_argument(
        "--config", default=str(PROJECT_ROOT / "config" / "pace_config.yaml")
    )
    result.add_argument("--trajectory")
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
        "--timing-validation",
        action="store_true",
        help="passive stop/poll transport qualification; never enables motors",
    )
    result.add_argument("--timing-duration", type=float, default=20.0)
    result.add_argument(
        "--command-rate-limit",
        type=float,
        default=None,
        metavar="RAD_S",
        help=(
            "explicit per-joint command-rate ceiling for this run; must be "
            "positive and no greater than max_velocity_rad_s"
        ),
    )
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
    if args.command_rate_limit is not None:
        rate = float(args.command_rate_limit)
        maximum = float(config["max_velocity_rad_s"])
        if not 0.0 < rate <= maximum:
            raise SystemExit(
                f"ERROR: --command-rate-limit must be in (0,{maximum}], got {rate}"
            )
        config["max_command_rate_rad_s"] = {
            name: rate for name in JOINT_ORDER
        }
        config["command_rate_override_rad_s"] = rate
    configured_hz = float(config.get("control_hz", 0.0))
    if configured_hz <= 0.0:
        raise SystemExit(f"ERROR: PACE control_hz must be positive, got {configured_hz}")
    if args.dry_run and args.timing_validation:
        raise SystemExit("ERROR: choose either --dry-run or --timing-validation")
    if args.timing_validation:
        qualified = run_timing_validation(
            config,
            args.deploy_root,
            args.can_front,
            args.can_back,
            args.timing_duration,
        )
        return 0 if qualified else 2
    if not args.trajectory:
        raise SystemExit("ERROR: --trajectory is required unless --timing-validation is used")
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
