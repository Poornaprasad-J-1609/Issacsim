"""Configured-rate PACE hardware and dry-run controller."""

from __future__ import annotations

import gc
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .constants import (
    JOINT_COUNT,
    JOINT_ORDER,
    PACE_EXPORT_ORDER,
    to_pace_order,
)
from .data_logger import PaceDataLogger
from .trajectory import compile_trajectory, joint_vector, load_trajectory


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def per_joint_values(value, field):
    if isinstance(value, (int, float)):
        result = np.full(JOINT_COUNT, float(value), dtype=np.float64)
    else:
        result = joint_vector(value, field=field)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field} contains NaN or Inf")
    return result


def hard_limit_arrays(config):
    limits = config["joint_limits"]
    q_min = np.array([float(limits[name]["min"]) for name in JOINT_ORDER])
    q_max = np.array([float(limits[name]["max"]) for name in JOINT_ORDER])
    if np.any(q_min >= q_max):
        raise ValueError("every joint limit must satisfy min < max")
    return q_min, q_max


def final_command_target(
    requested, previous, q_actual, qd_actual, config, dt,
    return_diagnostics=False,
):
    """Return the logical target that is allowed to reach packet construction."""
    requested = np.asarray(requested, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    q_actual = np.asarray(q_actual, dtype=np.float64)
    qd_actual = np.asarray(qd_actual, dtype=np.float64)
    q_min, q_max = hard_limit_arrays(config)
    max_rate = per_joint_values(config["max_command_rate_rad_s"], "max_command_rate_rad_s")
    kp = per_joint_values(config["kp"], "kp")
    kd = per_joint_values(config["kd"], "kd")
    torque_limit = per_joint_values(config["max_estimated_torque_nm"], "max_estimated_torque_nm")

    arrays = (requested, previous, q_actual, qd_actual)
    if any(array.shape != (JOINT_COUNT,) for array in arrays):
        raise ValueError(f"command arrays must all have shape ({JOINT_COUNT},)")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("command arrays contain NaN or Inf")

    q_des = np.clip(requested, q_min, q_max)
    hard_limit_active = np.abs(q_des - requested) > 1.0e-12
    before_rate = q_des.copy()
    q_des = np.clip(q_des, previous - max_rate * dt, previous + max_rate * dt)
    rate_limit_active = np.abs(q_des - before_rate) > 1.0e-12
    before_second_hard_limit = q_des.copy()
    q_des = np.clip(q_des, q_min, q_max)
    hard_limit_active |= np.abs(q_des - before_second_hard_limit) > 1.0e-12

    # Preserve damping first, then restrict the position target to the remaining
    # torque budget. The resulting q_des is what gets logged and transmitted.
    damping = -kd * qd_actual
    before_torque_budget = q_des.copy()
    for index in range(JOINT_COUNT):
        if torque_limit[index] <= 0.0 or kp[index] <= 0.0:
            continue
        position_budget_low = -torque_limit[index] - damping[index]
        position_budget_high = torque_limit[index] - damping[index]
        error_low = min(position_budget_low, position_budget_high) / kp[index]
        error_high = max(position_budget_low, position_budget_high) / kp[index]
        q_des[index] = np.clip(
            q_des[index],
            q_actual[index] + error_low,
            q_actual[index] + error_high,
        )
    torque_budget_limit_active = np.abs(q_des - before_torque_budget) > 1.0e-12
    before_final_hard_limit = q_des.copy()
    q_des = np.clip(q_des, q_min, q_max)
    hard_limit_active |= np.abs(q_des - before_final_hard_limit) > 1.0e-12
    diagnostics = {
        "hard_limit_active": hard_limit_active,
        "rate_limit_active": rate_limit_active,
        "torque_budget_limit_active": torque_budget_limit_active,
        "limiter_delta": q_des - requested,
        "limiter_abs_delta": np.abs(q_des - requested),
    }
    if return_diagnostics:
        return q_des, diagnostics
    return q_des


def safe_limit_arrays(config, trajectory_spec):
    """Return the intersection of deployment and trajectory-specific limits."""
    q_min, q_max = hard_limit_arrays(config)
    safe_limits = trajectory_spec.get("safe_joint_limits", {})
    if safe_limits is None:
        safe_limits = {}
    if not isinstance(safe_limits, dict):
        raise ValueError("safe_joint_limits must be a joint-name mapping")
    unknown = sorted(set(safe_limits) - set(JOINT_ORDER))
    if unknown:
        raise ValueError(f"safe_joint_limits contains unknown joints: {unknown}")
    for index, name in enumerate(JOINT_ORDER):
        if name not in safe_limits:
            continue
        bounds = safe_limits[name]
        if not isinstance(bounds, dict) or "min" not in bounds or "max" not in bounds:
            raise ValueError(f"safe_joint_limits.{name} needs min and max")
        q_min[index] = max(q_min[index], float(bounds["min"]))
        q_max[index] = min(q_max[index], float(bounds["max"]))
        if q_min[index] >= q_max[index]:
            raise ValueError(f"safe range for {name} is empty after intersection")
    return q_min, q_max


def validate_requested_trajectory(config, trajectory_spec, samples):
    """Reject requests outside the safe envelope, except an inward entry move."""
    if not samples:
        raise ValueError("trajectory contains no samples")
    q_min, q_max = safe_limit_arrays(config, trajectory_spec)
    hard_min, hard_max = hard_limit_arrays(config)
    tolerance = float(config.get("initial_pose_hard_limit_tolerance_rad", 0.02))
    if tolerance < 0.0:
        raise ValueError("initial_pose_hard_limit_tolerance_rad cannot be negative")
    initial = np.asarray(samples[0].q_requested, dtype=np.float64)
    if np.any(initial < hard_min - tolerance) or np.any(initial > hard_max + tolerance):
        index = int(np.flatnonzero(
            (initial < hard_min - tolerance) | (initial > hard_max + tolerance)
        )[0])
        raise ValueError(
            f"measured initial {JOINT_ORDER[index]}={initial[index]:+.6f} rad "
            f"is outside hard limit plus tolerance "
            f"[{hard_min[index] - tolerance:+.6f}, "
            f"{hard_max[index] + tolerance:+.6f}] rad"
        )

    segments = trajectory_spec.get("segments", [])
    first_segment_name = str(segments[0].get("name", "segment_0")) if segments else ""
    first_is_entry = bool(
        segments and str(segments[0].get("type", "")).lower() == "minimum_jerk"
    )
    initial_violation = np.maximum(q_min - initial, 0.0) + np.maximum(
        initial - q_max, 0.0
    )
    previous_violation = initial_violation.copy()
    for sample in samples:
        values = np.asarray(sample.q_requested, dtype=np.float64)
        if values.shape != (JOINT_COUNT,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"sample {sample.index} ({sample.segment}) has invalid q_requested"
            )
        invalid = np.flatnonzero((values < q_min) | (values > q_max))
        rejected = []
        for raw_index in invalid:
            index = int(raw_index)
            violation = max(q_min[index] - values[index], 0.0) + max(
                values[index] - q_max[index], 0.0
            )
            entering = bool(
                first_is_entry
                and sample.segment == first_segment_name
                and initial_violation[index] > 0.0
                and violation <= previous_violation[index] + 1.0e-10
            )
            if not entering:
                rejected.append(index)
            previous_violation[index] = violation
        if rejected:
            index = rejected[0]
            raise ValueError(
                "offline trajectory validation failed: "
                f"sample={sample.index} segment={sample.segment} "
                f"joint={JOINT_ORDER[index]} requested={values[index]:+.6f} rad "
                f"allowed=[{q_min[index]:+.6f}, {q_max[index]:+.6f}] rad"
            )
    return q_min, q_max


def trajectory_report(trajectory_spec, samples, control_hz=200.0):
    requested = np.stack([sample.q_requested for sample in samples])
    dt = 1.0 / float(control_hz)
    numerical_velocity = np.zeros_like(requested)
    if len(samples) > 1:
        numerical_velocity[1:] = np.diff(requested, axis=0) / dt
    chirps = [segment for segment in trajectory_spec.get("segments", [])
              if str(segment.get("type", "")).lower() == "chirp"]
    print("\nOFFLINE TRAJECTORY VALIDATION PASSED")
    print(f"Samples: {len(samples)} at {control_hz:.1f} Hz")
    print(f"Duration: {len(samples) * dt:.3f} s")
    for segment in chirps:
        center = joint_vector(segment.get("center"), field="chirp.center")
        amplitude = joint_vector(segment.get("amplitude"), field="chirp.amplitude")
        f_start = float(segment["f_start_hz"])
        f_end = float(segment.get("f_end_hz", f_start))
        print(
            f"Chirp: {f_start:.3f} -> {f_end:.3f} Hz, "
            f"duration={float(segment['duration_s']):.3f} s, "
            f"law={segment.get('law', 'linear')}"
        )
        print("Center:", np.array2string(center, precision=3))
        print("Signed amplitude:", np.array2string(amplitude, precision=3))
        if np.any(np.abs(amplitude) <= 0.0):
            missing = [JOINT_ORDER[i] for i in np.flatnonzero(amplitude == 0.0)]
            raise ValueError(f"chirp amplitudes must be nonzero for all joints: {missing}")
        theoretical = 2.0 * math.pi * max(f_start, f_end) * np.abs(amplitude)
        print("All 12 chirp amplitudes are nonzero.")
    print("Per-joint requested range and maximum velocity:")
    theoretical = np.zeros(JOINT_COUNT, dtype=np.float64)
    for segment in chirps:
        amplitude = joint_vector(segment.get("amplitude"), field="chirp.amplitude")
        f_max = max(float(segment["f_start_hz"]), float(segment.get("f_end_hz", 0.0)))
        theoretical = np.maximum(theoretical, 2.0 * math.pi * f_max * np.abs(amplitude))
    for i, name in enumerate(JOINT_ORDER):
        print(
            f"  {name:20s} [{requested[:, i].min():+.4f}, "
            f"{requested[:, i].max():+.4f}] rad  "
            f"theoretical={theoretical[i]:.4f} rad/s  "
            f"sampled={np.max(np.abs(numerical_velocity[:, i])):.4f} rad/s"
        )


def save_pace_dataset(path, times, q_des, q_actual):
    """Write the exact three-key PACE tensor contract in explicit PACE order."""
    if not times:
        return None
    time_array = np.asarray(times, dtype=np.float32)
    desired = to_pace_order(np.asarray(q_des, dtype=np.float32))
    actual = to_pace_order(np.asarray(q_actual, dtype=np.float32))
    if desired.shape != (len(time_array), JOINT_COUNT) or actual.shape != desired.shape:
        raise ValueError("PACE arrays must have shapes time=[N], positions=[N,12]")
    if not np.all(np.isfinite(time_array)) or not np.all(np.diff(time_array) > 0.0):
        raise ValueError("PACE timestamps must be finite and strictly increasing")
    if not np.all(np.isfinite(desired)) or not np.all(np.isfinite(actual)):
        raise ValueError("PACE position arrays contain NaN or Inf")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to save chirp_data.pt") from exc
    output = Path(path)
    torch.save(
        {
            "time": torch.from_numpy(time_array),
            "des_dof_pos": torch.from_numpy(desired),
            "dof_pos": torch.from_numpy(actual),
        },
        output,
    )
    return output


def csv_fieldnames():
    fields = [
        "sample_index", "time_s", "nominal_time_s", "command_time_s",
        "feedback_time_s", "control_dt_s", "loop_frequency_hz",
        "loop_lateness_s", "loop_work_s", "deadline_missed",
        "scheduler_resynchronized",
        "instantaneous_frequency_hz",
        "trajectory_segment", "feedback_complete", "feedback_max_age_s",
        "feedback_missing", "safety_event",
    ]
    suffixes = [
        "q_requested", "q_des", "q_actual", "qd_actual", "q_raw", "qd_raw",
        "tracking_error",
        "limiter_delta", "limiter_abs_delta", "hard_limit_active",
        "rate_limit_active", "torque_budget_limit_active",
        "kp", "kd", "v_des", "tau_ff", "motor_id", "bus",
        "feedback_time_s", "feedback_age_s", "torque_nm", "temperature_c",
        "fault_bits",
    ]
    fields.extend(f"{name}_{suffix}" for name in JOINT_ORDER for suffix in suffixes)
    return fields


def git_commit(path):
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.stdout.strip()
    except Exception:
        return None


def make_output_paths(output_root, dataset_name, dry_run):
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mode = "dry" if dry_run else "real"
    directory = Path(output_root).expanduser() / str(dataset_name) / f"{stamp}_{mode}"
    return directory / f"pace_{dataset_name}_{stamp}_{mode}.csv"


def metadata(config, trajectory_spec, config_path, trajectory_path, deploy_root, dry_run):
    control_hz = float(config["control_hz"])
    return {
        "format": "grallator_pace_v2",
        "created_at": datetime.now().isoformat(),
        "dry_run": bool(dry_run),
        "control_rate_hz": control_hz,
        "control_dt_s": 1.0 / control_hz,
        "joint_order": JOINT_ORDER,
        "internal_joint_order": JOINT_ORDER,
        "pace_export_order": PACE_EXPORT_ORDER,
        "q_des_definition": "logical q_des from the final MIT command after hard/rate/estimated-torque limiting",
        "q_actual_definition": "real MIT encoder feedback converted using direction and zero offset",
        "q_raw_definition": "mechanical position from MIT operation-status feedback (communication type 2)",
        "qd_raw_definition": "mechanical velocity from MIT operation-status feedback (communication type 2)",
        "parameter_read_note": "0x7019/0x701B parameter reads are not mixed into active MIT streaming",
        "feedback_pairing": "each row uses the latest complete feedback read before that cycle's command",
        "config_path": str(Path(config_path).resolve()),
        "trajectory_path": str(Path(trajectory_path).resolve()),
        "deploy_root": str(Path(deploy_root).resolve()) if deploy_root else None,
        "deploy_git_commit": git_commit(deploy_root) if deploy_root else None,
        "config": config,
        "trajectory": trajectory_spec,
    }


def _base_row(
    index, nominal_time, command_time, feedback_time, dt, lateness, work_s,
    deadline_missed, segment, instantaneous_frequency_hz,
):
    return {
        "sample_index": index,
        "time_s": command_time,
        "nominal_time_s": nominal_time,
        "command_time_s": command_time,
        "feedback_time_s": feedback_time,
        "control_dt_s": dt,
        "loop_frequency_hz": 1.0 / dt if dt > 0.0 else float("nan"),
        "loop_lateness_s": lateness,
        "loop_work_s": work_s,
        "deadline_missed": bool(deadline_missed),
        "scheduler_resynchronized": False,
        "instantaneous_frequency_hz": (
            "" if instantaneous_frequency_hz is None
            else float(instantaneous_frequency_hz)
        ),
        "trajectory_segment": segment,
        "feedback_complete": True,
        "feedback_max_age_s": max(0.0, feedback_time - command_time),
        "feedback_missing": "",
        "safety_event": "",
    }


def timing_summary(control_hz, loop_dts, loop_work, missed_deadlines, label="CONTROL"):
    dts = np.asarray(loop_dts, dtype=np.float64)
    work = np.asarray(loop_work, dtype=np.float64)
    valid = dts[np.isfinite(dts) & (dts > 0.0)]
    if valid.size == 0:
        print(f"\n{label} TIMING SUMMARY: no valid cycles")
        return
    frequencies = 1.0 / valid
    print(f"\n{label} TIMING SUMMARY")
    print(f"requested control rate : {control_hz:.3f} Hz")
    print(f"target dt              : {1000.0 / control_hz:.3f} ms")
    print(f"achieved mean rate     : {1.0 / valid.mean():.3f} Hz")
    print(f"minimum rate           : {frequencies.min():.3f} Hz")
    print(f"maximum rate           : {frequencies.max():.3f} Hz")
    print(f"mean dt                : {1000.0 * valid.mean():.3f} ms")
    print(f"dt jitter std          : {1000.0 * valid.std():.3f} ms")
    print(f"worst dt               : {1000.0 * valid.max():.3f} ms")
    print(f"worst loop duration    : {1000.0 * work.max():.3f} ms")
    print(f"missed deadlines       : {int(missed_deadlines)}")


def next_control_deadline(deadline, wake_time, dt, severe_lateness):
    """Advance one period, resynchronizing after an isolated long host pause."""
    lateness = max(0.0, float(wake_time) - float(deadline))
    resynchronized = lateness > float(severe_lateness)
    anchor = float(wake_time) if resynchronized else float(deadline)
    return anchor + float(dt), resynchronized


def run_dry(config, config_path, trajectory_path, output_root, dataset_name, initial_q):
    control_hz = float(config["control_hz"])
    dt = 1.0 / control_hz
    trajectory_spec, samples = load_trajectory(
        trajectory_path, initial_q, expected_hz=control_hz
    )
    validate_requested_trajectory(config, trajectory_spec, samples)
    trajectory_report(trajectory_spec, samples, control_hz=control_hz)
    csv_path = make_output_paths(output_root, dataset_name, dry_run=True)
    logger = PaceDataLogger(
        csv_path,
        csv_fieldnames(),
        metadata(config, trajectory_spec, config_path, trajectory_path, None, True),
    )
    kp = per_joint_values(config["kp"], "kp")
    kd = per_joint_values(config["kd"], "kd")
    directions = per_joint_values(config["motor_directions"], "motor_directions")
    offsets = per_joint_values(config["joint_offsets"], "joint_offsets")
    motor_ids = [int(config["motor_ids"][name]) for name in JOINT_ORDER]
    buses = [str(config["joint_can_bus"][name]) for name in JOINT_ORDER]
    q_actual = np.asarray(initial_q, dtype=np.float64).copy()
    qd_actual = np.zeros(JOINT_COUNT, dtype=np.float64)
    q_des_previous = q_actual.copy()
    pace_times = []
    pace_q_des = []
    pace_q_actual = []
    limiter_counts = {
        "hard_limit_active": np.zeros(JOINT_COUNT, dtype=np.int64),
        "rate_limit_active": np.zeros(JOINT_COUNT, dtype=np.int64),
        "torque_budget_limit_active": np.zeros(JOINT_COUNT, dtype=np.int64),
    }
    max_limiter_delta = np.zeros(JOINT_COUNT, dtype=np.float64)
    excessive_cycles = 0
    maximum_excessive_cycles = 0
    warned = False
    mock_time_constant_s = 0.1007
    mock_alpha = 1.0 - math.exp(-dt / mock_time_constant_s)
    try:
        for sample in samples:
            q_des, diagnostics = final_command_target(
                sample.q_requested, q_des_previous, q_actual, qd_actual,
                config, dt, return_diagnostics=True,
            )
            max_delta_this_cycle = float(np.max(diagnostics["limiter_abs_delta"]))
            max_limiter_delta = np.maximum(
                max_limiter_delta, diagnostics["limiter_abs_delta"]
            )
            for key in limiter_counts:
                limiter_counts[key] += diagnostics[key].astype(np.int64)
            if max_delta_this_cycle > float(config.get("limiter_warning_delta_rad", 0.005)):
                if not warned:
                    print(
                        "WARNING: dry run shows q_des-q_requested exceeding "
                        f"{float(config.get('limiter_warning_delta_rad', 0.005)):.3f} rad"
                    )
                    warned = True
            if max_delta_this_cycle > float(config.get("limiter_abort_delta_rad", 0.03)):
                excessive_cycles += 1
                maximum_excessive_cycles = max(maximum_excessive_cycles, excessive_cycles)
            else:
                excessive_cycles = 0
            old_q = q_actual.copy()
            q_actual += mock_alpha * (q_des - q_actual)
            qd_actual = (q_actual - old_q) / dt
            q_raw = offsets + directions * q_actual
            qd_raw = directions * qd_actual
            row = _base_row(
                sample.index, sample.time_s, sample.time_s, sample.time_s + 0.001,
                dt, 0.0, 0.0, False, sample.segment,
                sample.instantaneous_frequency_hz,
            )
            for i, name in enumerate(JOINT_ORDER):
                values = {
                    "q_requested": sample.q_requested[i], "q_des": q_des[i],
                    "q_actual": q_actual[i], "qd_actual": qd_actual[i],
                    "q_raw": q_raw[i], "qd_raw": qd_raw[i], "kp": kp[i],
                    "tracking_error": q_des[i] - q_actual[i],
                    "limiter_delta": diagnostics["limiter_delta"][i],
                    "limiter_abs_delta": diagnostics["limiter_abs_delta"][i],
                    "hard_limit_active": bool(diagnostics["hard_limit_active"][i]),
                    "rate_limit_active": bool(diagnostics["rate_limit_active"][i]),
                    "torque_budget_limit_active": bool(
                        diagnostics["torque_budget_limit_active"][i]
                    ),
                    "kd": kd[i], "v_des": 0.0, "tau_ff": 0.0,
                    "motor_id": motor_ids[i], "bus": buses[i],
                    "feedback_time_s": sample.time_s + 0.001,
                    "feedback_age_s": 0.001, "torque_nm": 0.0,
                    "temperature_c": 25.0, "fault_bits": 0,
                }
                row.update({f"{name}_{key}": value for key, value in values.items()})
            logger.write(row)
            pace_times.append(sample.time_s)
            pace_q_des.append(q_des.copy())
            pace_q_actual.append(q_actual.copy())
            q_des_previous = q_des
    finally:
        logger.close()
    pace_path = save_pace_dataset(
        csv_path.parent / "chirp_data.pt", pace_times, pace_q_des, pace_q_actual
    )
    print("\nDRY-RUN LIMITER REPORT")
    for i, name in enumerate(JOINT_ORDER):
        print(
            f"  {name:20s} max|delta|={max_limiter_delta[i]:.6f} rad  "
            f"hard={limiter_counts['hard_limit_active'][i]}  "
            f"rate={limiter_counts['rate_limit_active'][i]}  "
            f"torque={limiter_counts['torque_budget_limit_active'][i]}"
        )
    abort_cycles = int(config.get("limiter_abort_consecutive_cycles", 5))
    if maximum_excessive_cycles >= abort_cycles:
        print(
            "WARNING: this command would trigger the real-hardware limiter "
            f"distortion abort ({maximum_excessive_cycles} consecutive cycles; "
            f"threshold={abort_cycles}). Do not run hardware until reviewed."
        )
    else:
        print("Limiter distortion abort check: PASS")
    print(f"PACE tensor dataset: {pace_path}")
    timing_summary(
        control_hz,
        [dt] * len(samples),
        [0.0] * len(samples),
        0,
        label="DRY-RUN CONTROL",
    )
    return csv_path, len(samples)


def _import_deployment(deploy_root):
    deploy_root = Path(deploy_root).expanduser().resolve()
    src = deploy_root / "src"
    if not (src / "motor_command_layer.py").exists():
        raise FileNotFoundError(f"deployment source not found under {src}")
    sys.path.insert(0, str(src))
    from can_topology import close_can_buses, open_can_buses
    from motor_command_layer import MotorCommandLayer
    from state_estimator import MitFeedbackStateEstimator
    return MotorCommandLayer, MitFeedbackStateEstimator, open_can_buses, close_can_buses


def verify_deployment_contract(config, deploy_root):
    deploy_root = Path(deploy_root)
    sources = {
        "motor_ids": load_yaml(deploy_root / "config" / "motor_ids.yaml")["motor_ids"],
        "motor_directions": load_yaml(
            deploy_root / "config" / "motor_directions.yaml"
        )["motor_directions"],
        "joint_offsets": load_yaml(
            deploy_root / "config" / "joint_offsets.yaml"
        )["joint_offsets"],
        "joint_limits": load_yaml(
            deploy_root / "config" / "joint_limits.yaml"
        )["joint_limits"],
    }
    limit_reference = config.get(
        "deployment_joint_limits_reference",
        config["joint_limits"],
    )
    mismatches = []
    for name in JOINT_ORDER:
        if int(config["motor_ids"][name]) != int(sources["motor_ids"][name]):
            mismatches.append(f"{name} motor_id")
        if float(config["motor_directions"][name]) != float(
            sources["motor_directions"][name]
        ):
            mismatches.append(f"{name} motor_direction")
        if abs(float(config["joint_offsets"][name]) - float(
            sources["joint_offsets"][name]
        )) > 1.0e-9:
            mismatches.append(f"{name} joint_offset")
        for bound in ("min", "max"):
            reference_bounds = limit_reference.get(name, config["joint_limits"][name])
            if abs(float(reference_bounds[bound]) - float(
                sources["joint_limits"][name][bound]
            )) > 1.0e-9:
                mismatches.append(f"{name} limit.{bound}")
    if mismatches:
        raise RuntimeError(
            "PACE/deployment calibration contract differs: " + ", ".join(mismatches)
        )


def print_terminal_monitor(
    sample, q_requested, q_des, q_actual, qd_actual, diagnostics,
    limiter_warning_delta_rad,
):
    tracking = q_des - q_actual
    limiter_delta = q_des - q_requested
    active = []
    for label, key in (
        ("hard", "hard_limit_active"),
        ("rate", "rate_limit_active"),
        ("torque", "torque_budget_limit_active"),
    ):
        joints = [
            JOINT_ORDER[i] for i in np.flatnonzero(diagnostics[key])
        ]
        if joints:
            active.append(f"{label}=" + ",".join(joints))
    frequency = sample.instantaneous_frequency_hz
    frequency_text = "n/a" if frequency is None else f"{frequency:.3f} Hz"
    tracking_index = int(np.argmax(np.abs(tracking)))
    warning = (
        " LIMITER_WARNING"
        if np.max(np.abs(limiter_delta)) > limiter_warning_delta_rad else ""
    )
    print(
        f"PACE segment={sample.segment} t={sample.segment_time_s:.2f}s "
        f"frequency={frequency_text} "
        f"max|tracking|={np.max(np.abs(tracking)):.4f}rad "
        f"tracking_joint={JOINT_ORDER[tracking_index]} "
        f"max|limiter|={np.max(np.abs(limiter_delta)):.4f}rad "
        f"max|qd|={np.max(np.abs(qd_actual)):.4f}rad/s "
        f"active={'none' if not active else ';'.join(active)}{warning}"
    )


class AsyncTerminalMonitor:
    """Best-effort low-rate terminal output isolated from the control loop."""

    def __init__(self):
        self.queue = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, name="PaceTerminal", daemon=True)
        self.thread.start()

    def submit(self, args):
        try:
            self.queue.put_nowait(args)
        except queue.Full:
            pass

    def _run(self):
        while True:
            args = self.queue.get()
            if args is None:
                return
            print_terminal_monitor(*args)

    def close(self):
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(None)
        self.thread.join(timeout=2.0)


def run_timing_validation(config, deploy_root, can_front, can_back, duration_s):
    """Qualify passive two-adapter stop/poll transport without enabling motors."""
    MotorCommandLayer, Estimator, open_can_buses, close_can_buses = _import_deployment(
        deploy_root
    )
    verify_deployment_contract(config, deploy_root)
    control_hz = float(config["control_hz"])
    dt = 1.0 / control_hz
    motor_ids = {name: int(config["motor_ids"][name]) for name in JOINT_ORDER}
    routing = {name: str(config["joint_can_bus"][name]) for name in JOINT_ORDER}
    layer = MotorCommandLayer(JOINT_ORDER, motor_ids, joint_can_bus=routing)
    layer.joint_directions = {
        name: float(config["motor_directions"][name]) for name in JOINT_ORDER
    }
    layer.joint_offsets = {
        name: float(config["joint_offsets"][name]) for name in JOINT_ORDER
    }
    buses = open_can_buses(
        {"front": str(can_front), "back": str(can_back)},
        backend="socketcan",
        bitrate=int(config.get("can_bitrate", 1_000_000)),
        timeout=float(config.get("can_timeout_s", 0.01)),
    )
    loop_dts = []
    loop_work = []
    missed_deadlines = 0
    incomplete_feedback = 0
    incomplete_details = []
    consecutive_incomplete = 0
    maximum_consecutive_incomplete = 0
    try:
        feedback_types = (
            int(layer.proto.get("comm_type_feedback", 2)),
            int(layer.proto.get("comm_type_active_feedback", 24)),
        )
        seen = set()
        for bus in buses.values():
            if id(bus) in seen:
                continue
            seen.add(id(bus))
            bus.configure_feedback_filters(feedback_types)
        estimator = Estimator(
            q_initial=np.zeros(JOINT_COUNT, dtype=np.float32),
            policy_order=JOINT_ORDER,
            motor_ids=motor_ids,
            motor_layer=layer,
            bus=buses,
            joint_velocity_source="mit",
        )
        expected = estimator.expected_feedback_bus_motor_ids()
        feedback_timeout = float(config.get("feedback_timeout_s", 0.003))
        tolerance = float(config.get("deadline_miss_tolerance_s", 0.0005))
        warmup_cycles = int(config.get("timing_validation_warmup_cycles", 20))
        if warmup_cycles < 0:
            raise ValueError("timing_validation_warmup_cycles cannot be negative")

        # Fill the request/reply pipeline before measuring it. Marking the
        # batch before transmission is important: front-lane motors can reply
        # while the remainder of the two-CAN batch is still being sent.
        for _ in range(warmup_cycles):
            warmup_start = time.monotonic()
            estimator.mark_command_sent(warmup_start)
            layer.send_raw_commands(buses, layer.build_feedback_poll_commands())
            estimator.refresh_from_bus(
                timeout=feedback_timeout,
                expected_bus_motor_ids=expected,
            )
            remaining = dt - (time.monotonic() - warmup_start)
            if remaining > 0.0:
                time.sleep(remaining)

        count = int(round(float(duration_s) * control_hz))
        start = time.monotonic()
        previous = start - dt
        for index in range(count):
            deadline = start + index * dt
            wait = deadline - time.monotonic()
            if wait > 0.0:
                time.sleep(wait)
            wake = time.monotonic()
            if wake - deadline > tolerance:
                missed_deadlines += 1
            command_send = time.monotonic()
            estimator.mark_command_sent(command_send)
            layer.send_raw_commands(buses, layer.build_feedback_poll_commands())
            estimator.refresh_from_bus(
                timeout=feedback_timeout,
                expected_bus_motor_ids=expected,
            )
            received = set(estimator.last_refresh_current_bus_motor_ids)
            if received != expected:
                incomplete_feedback += 1
                consecutive_incomplete += 1
                maximum_consecutive_incomplete = max(
                    maximum_consecutive_incomplete,
                    consecutive_incomplete,
                )
                if len(incomplete_details) < 10:
                    incomplete_details.append((index, sorted(expected - received)))
            else:
                consecutive_incomplete = 0
            end = time.monotonic()
            loop_dts.append(wake - previous)
            loop_work.append(end - wake)
            previous = wake
    finally:
        try:
            for _ in range(3):
                layer.send_raw_commands(buses, layer.build_stop_commands())
                time.sleep(max(dt, 0.01))
        finally:
            close_can_buses(buses)
    timing_summary(
        control_hz,
        loop_dts,
        loop_work,
        missed_deadlines,
        label="PASSIVE TWO-CAN",
    )
    print(f"excluded warm-up cycles: {warmup_cycles}")
    print(f"incomplete feedback cycles: {incomplete_feedback}")
    dropout_fraction = incomplete_feedback / max(count, 1)
    print(f"feedback dropout fraction: {dropout_fraction:.6%}")
    print(
        "maximum consecutive incomplete cycles: "
        f"{maximum_consecutive_incomplete}"
    )
    if incomplete_details:
        print("first incomplete measured cycles:")
        for index, missing in incomplete_details:
            formatted = ", ".join(
                f"{bus}:0x{motor_id:02X}" for bus, motor_id in missing
            )
            print(f"  cycle {index}: missing {formatted}")
    maximum_dropout_fraction = float(
        config.get("timing_validation_max_dropout_fraction", 0.001)
    )
    abort_consecutive = int(
        config.get("feedback_dropout_abort_consecutive_cycles", 3)
    )
    if not 0.0 <= maximum_dropout_fraction < 1.0:
        raise ValueError("timing_validation_max_dropout_fraction must be in [0,1)")
    if abort_consecutive < 1:
        raise ValueError("feedback_dropout_abort_consecutive_cycles must be positive")
    qualified = bool(
        loop_work
        and max(loop_work) <= dt
        and dropout_fraction <= maximum_dropout_fraction
        and maximum_consecutive_incomplete < abort_consecutive
    )
    if qualified and incomplete_feedback:
        result = "PASSED WITH ISOLATED DROPOUTS"
    else:
        result = "PASSED" if qualified else "FAILED"
    print(
        "qualification limits: dropout_fraction<="
        f"{maximum_dropout_fraction:.6%}, consecutive_incomplete<"
        f"{abort_consecutive}"
    )
    print("transport qualification:", result)
    return qualified


def run_hardware(
    config, config_path, trajectory_path, output_root, dataset_name,
    deploy_root, can_front, can_back,
):
    MotorCommandLayer, Estimator, open_can_buses, close_can_buses = _import_deployment(deploy_root)
    verify_deployment_contract(config, deploy_root)
    control_hz = float(config["control_hz"])
    if control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    dt = 1.0 / control_hz
    motor_ids = {name: int(config["motor_ids"][name]) for name in JOINT_ORDER}
    routing = {name: str(config["joint_can_bus"][name]) for name in JOINT_ORDER}
    ports = {"front": str(can_front), "back": str(can_back)}
    layer = MotorCommandLayer(JOINT_ORDER, motor_ids, joint_can_bus=routing)

    # The PACE experiment owns these values without mutating deployment files.
    layer.joint_directions = {
        name: float(config["motor_directions"][name]) for name in JOINT_ORDER
    }
    layer.joint_offsets = {
        name: float(config["joint_offsets"][name]) for name in JOINT_ORDER
    }
    layer.hard_joint_limits = {
        name: (
            float(config["joint_limits"][name]["min"]),
            float(config["joint_limits"][name]["max"]),
        ) for name in JOINT_ORDER
    }
    kp = per_joint_values(config["kp"], "kp")
    kd = per_joint_values(config["kd"], "kd")
    layer.gains["policy"] = {
        "hip": {"kp": float(kp[0]), "kd": float(kd[0])},
        "thigh": {"kp": float(kp[4]), "kd": float(kd[4])},
        "calf": {"kp": float(kp[8]), "kd": float(kd[8])},
        "joints": {
            name: {"kp": float(kp[i]), "kd": float(kd[i])}
            for i, name in enumerate(JOINT_ORDER)
        },
    }
    layer.feedforward = {"v_des": 0.0, "tau_ff": 0.0}
    layer.virtual_joint_stop_enabled = False
    layer.policy_pd_torque_limit = 0.0
    layer.policy_pd_torque_limits = {name: 0.0 for name in JOINT_ORDER}

    buses = open_can_buses(
        ports,
        backend="socketcan",
        bitrate=int(config.get("can_bitrate", 1_000_000)),
        timeout=float(config.get("can_timeout_s", 0.01)),
    )
    # The deployment SocketCan wrapper creates its send lock while installing
    # feedback filters.  Configure every unique adapter before any poll, enable,
    # or MIT command is sent; this also rejects echoed/non-feedback SLCAN frames.
    try:
        feedback_types = (
            int(layer.proto.get("comm_type_feedback", 2)),
            int(layer.proto.get("comm_type_active_feedback", 24)),
        )
        configured_bus_ids = set()
        for bus in buses.values():
            if id(bus) in configured_bus_ids:
                continue
            configured_bus_ids.add(id(bus))
            configure = getattr(bus, "configure_feedback_filters", None)
            if configure is None:
                raise RuntimeError(
                    "SocketCAN adapter does not expose configure_feedback_filters"
                )
            configure(feedback_types)
    except Exception:
        close_can_buses(buses)
        raise

    estimator = Estimator(
        q_initial=np.zeros(JOINT_COUNT, dtype=np.float32),
        policy_order=JOINT_ORDER,
        motor_ids=motor_ids,
        motor_layer=layer,
        bus=buses,
        joint_velocity_source="mit",
    )
    expected = estimator.expected_feedback_bus_motor_ids()
    feedback_timeout = float(config.get("feedback_timeout_s", 0.012))
    enabled = False
    logger = None
    csv_path = None
    pace_times = []
    pace_q_des = []
    pace_q_actual = []
    loop_dts = []
    loop_work = []
    missed_deadlines = 0
    scheduler_resynchronizations = 0
    feedback_dropout_cycles = 0
    consecutive_incomplete_feedback = 0
    maximum_consecutive_incomplete_feedback = 0
    completed = False
    stop_requested = False
    terminal_monitor = None
    gc_disabled_for_control = False

    def request_stop(_signal=None, _frame=None):
        nonlocal stop_requested
        stop_requested = True

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        # Passive feedback acquisition: stop/poll frames cannot move the motors.
        for _ in range(10):
            layer.send_raw_commands(buses, layer.build_feedback_poll_commands())
            time.sleep(max(dt, 0.01))
            estimator.refresh_from_bus(timeout=feedback_timeout, expected_bus_motor_ids=expected)
            if len(estimator.last_feedback_by_joint) == JOINT_COUNT:
                break
        if len(estimator.last_feedback_by_joint) != JOINT_COUNT:
            missing = sorted(set(JOINT_ORDER) - set(estimator.last_feedback_by_joint))
            raise RuntimeError(f"initial feedback missing joints: {missing}")

        initial_q = np.asarray(estimator.q_current, dtype=np.float64).copy()
        trajectory_spec, samples = load_trajectory(
            trajectory_path, initial_q, expected_hz=control_hz
        )
        trajectory_plan = compile_trajectory(
            trajectory_spec, initial_q, expected_hz=control_hz
        )
        validate_requested_trajectory(config, trajectory_spec, samples)
        trajectory_report(trajectory_spec, samples, control_hz=control_hz)
        warning_delta = float(config.get("limiter_warning_delta_rad", 0.005))
        abort_delta = float(config.get("limiter_abort_delta_rad", 0.03))
        abort_cycles = int(config.get("limiter_abort_consecutive_cycles", 5))
        status_hz = float(config.get("terminal_status_hz", 2.0))
        if warning_delta < 0.0 or abort_delta <= warning_delta:
            raise ValueError("limiter thresholds require 0 <= warning < abort")
        if abort_cycles < 1:
            raise ValueError("limiter_abort_consecutive_cycles must be positive")
        if status_hz <= 0.0 or status_hz > control_hz:
            raise ValueError("terminal_status_hz must be in (0, control_hz]")
        status_interval = max(1, int(round(control_hz / status_hz)))
        print("\nPACE REAL-HARDWARE TEST")
        print("Internal joint order:", ", ".join(JOINT_ORDER))
        print("PACE export order:", ", ".join(PACE_EXPORT_ORDER))
        print("Samples:", len(samples), "Duration:", trajectory_plan.duration_s, "s")
        print("Initial q:", np.array2string(initial_q, precision=4))
        print("The robot must be suspended and the area clear.")
        confirmation = input("Type ENABLE PACE to enable all motors: ").strip()
        if confirmation != "ENABLE PACE":
            raise RuntimeError("hardware enable was not confirmed")

        for _ in range(3):
            layer.send_raw_commands(buses, layer.build_enable_commands())
            time.sleep(0.03)
        enabled = True
        # Remove enable/status replies so sample zero can only consume replies
        # generated after its own MIT command batch.
        estimator.refresh_from_bus(timeout=0.0, expected_bus_motor_ids=expected)

        # One position-hold command at the measured pose establishes the first
        # complete enabled feedback snapshot without introducing a position step.
        warmup_commands = layer.build_mit_commands(
            initial_q,
            phase="policy",
            feedback_by_joint=estimator.last_feedback_by_joint,
        )
        warmup_send = time.monotonic()
        layer.send_raw_commands(buses, warmup_commands)
        estimator.mark_command_sent(warmup_send)
        time.sleep(dt)
        estimator.refresh_from_bus(
            timeout=feedback_timeout,
            expected_bus_motor_ids=expected,
        )
        if set(estimator.last_refresh_current_bus_motor_ids) != expected:
            raise RuntimeError("enabled warmup did not return complete motor feedback")

        csv_path = make_output_paths(output_root, dataset_name, dry_run=False)
        logger = PaceDataLogger(
            csv_path,
            csv_fieldnames(),
            metadata(
                config, trajectory_spec, config_path, trajectory_path,
                deploy_root, False,
            ),
        )
        terminal_monitor = AsyncTerminalMonitor()
        q_des_previous = initial_q.copy()
        start = time.monotonic()
        next_deadline = start
        previous_control_timestamp = start - dt
        consecutive_severe_misses = 0
        consecutive_limiter_distortion = 0
        consecutive_tracking_error = 0
        last_limiter_warning_sample = -status_interval
        deadline_tolerance = float(config.get("deadline_miss_tolerance_s", 0.0005))
        severe_lateness = float(config.get("max_loop_lateness_s", dt))
        max_severe_misses = int(config.get("max_consecutive_overruns", 3))
        feedback_dropout_abort_cycles = int(
            config.get("feedback_dropout_abort_consecutive_cycles", 3)
        )
        if feedback_dropout_abort_cycles < 1:
            raise ValueError(
                "feedback_dropout_abort_consecutive_cycles must be positive"
            )
        sample_index = 0
        if gc.isenabled():
            gc.collect()
            gc.disable()
            gc_disabled_for_control = True

        while True:
            if stop_requested:
                raise RuntimeError("operator stop requested")
            deadline = next_deadline
            wait = deadline - time.monotonic()
            if wait > 0.0:
                time.sleep(wait)
            cycle_wake = time.monotonic()
            lateness = max(0.0, cycle_wake - deadline)
            deadline_missed = lateness > deadline_tolerance
            if deadline_missed:
                missed_deadlines += 1
            next_deadline, scheduler_resynchronized = next_control_deadline(
                deadline,
                cycle_wake,
                dt,
                severe_lateness,
            )
            if scheduler_resynchronized:
                scheduler_resynchronizations += 1
                consecutive_severe_misses += 1
            else:
                consecutive_severe_misses = 0
            if consecutive_severe_misses >= max_severe_misses:
                raise RuntimeError(
                    "control timing failed: "
                    f"lateness={lateness:.6f}s for "
                    f"{consecutive_severe_misses} consecutive cycles"
                )

            # Read replies generated by the previous cycle before preparing the
            # next command. Both adapter reads remain inside the deployment API.
            if sample_index > 0:
                estimator.refresh_from_bus(
                    timeout=feedback_timeout,
                    expected_bus_motor_ids=expected,
                )
            received = set(estimator.last_refresh_current_bus_motor_ids)
            complete = received == expected
            control_timestamp_abs = time.monotonic()
            elapsed = control_timestamp_abs - start
            if elapsed >= trajectory_plan.duration_s:
                break
            actual_dt = control_timestamp_abs - previous_control_timestamp
            previous_control_timestamp = control_timestamp_abs
            sample = trajectory_plan.evaluate(elapsed, index=sample_index)
            q_actual_before = np.asarray(estimator.q_current, dtype=np.float64).copy()
            qd_actual_before = np.asarray(estimator.qd_current, dtype=np.float64).copy()
            q_prepared, diagnostics = final_command_target(
                sample.q_requested, q_des_previous, q_actual_before,
                qd_actual_before, config, dt, return_diagnostics=True,
            )
            commands = layer.build_mit_commands(
                q_prepared,
                phase="policy",
                feedback_by_joint=estimator.last_feedback_by_joint,
            )
            command_by_joint = {cmd["joint_name"]: cmd for cmd in commands}
            command_send_abs = time.monotonic()
            layer.send_raw_commands(buses, commands)
            estimator.mark_command_sent(command_send_abs)
            feedback_by_joint = estimator.last_feedback_by_joint
            feedback_times = [
                float(feedback_by_joint[name].get("timestamp", control_timestamp_abs))
                for name in JOINT_ORDER if name in feedback_by_joint
            ]
            feedback_abs = max(feedback_times, default=control_timestamp_abs)
            command_time = command_send_abs - start
            feedback_time = feedback_abs - start
            row = _base_row(
                sample_index, sample_index * dt, command_time, feedback_time,
                actual_dt, lateness, 0.0, deadline_missed, sample.segment,
                sample.instantaneous_frequency_hz,
            )
            row["scheduler_resynchronized"] = scheduler_resynchronized
            row["feedback_complete"] = bool(complete)

            safety_reasons = []
            if not complete:
                missing = sorted(expected - received)
                feedback_dropout_cycles += 1
                consecutive_incomplete_feedback += 1
                maximum_consecutive_incomplete_feedback = max(
                    maximum_consecutive_incomplete_feedback,
                    consecutive_incomplete_feedback,
                )
                row["feedback_missing"] = ",".join(
                    f"{bus}:0x{motor_id:02X}" for bus, motor_id in missing
                )
                if consecutive_incomplete_feedback >= feedback_dropout_abort_cycles:
                    safety_reasons.append(
                        "sustained incomplete feedback: "
                        f"{consecutive_incomplete_feedback} cycles; "
                        f"missing={row['feedback_missing']}"
                    )
            else:
                consecutive_incomplete_feedback = 0
            q_des_sent = np.zeros(JOINT_COUNT, dtype=np.float64)
            q_actual_cycle = np.full(JOINT_COUNT, np.nan, dtype=np.float64)
            qd_actual_cycle = np.full(JOINT_COUNT, np.nan, dtype=np.float64)
            for i, name in enumerate(JOINT_ORDER):
                cmd = command_by_joint[name]
                fb = feedback_by_joint.get(name, {})
                fb_abs = float(fb.get("timestamp", float("nan")))
                fb_age = (
                    control_timestamp_abs - fb_abs
                    if math.isfinite(fb_abs) else float("nan")
                )
                fault_bits = int(fb.get("fault_bits", 0))
                temperature = float(fb.get("temperature_c", float("nan")))
                q_actual = float(fb.get("joint_position", float("nan")))
                qd_actual = float(fb.get("joint_velocity_mit", float("nan")))
                q_raw = float(fb.get("position_raw", float("nan")))
                qd_raw = float(fb.get("velocity_raw", float("nan")))
                torque = float(fb.get("joint_torque", float("nan")))
                q_des_sent[i] = float(cmd["q_des"])
                q_actual_cycle[i] = q_actual
                qd_actual_cycle[i] = qd_actual
                if fault_bits:
                    safety_reasons.append(f"{name} fault_bits=0x{fault_bits:X}")
                q_min = float(config["joint_limits"][name]["min"])
                q_max = float(config["joint_limits"][name]["max"])
                if not math.isfinite(q_actual):
                    safety_reasons.append(f"{name} position feedback is invalid")
                elif q_actual < q_min - 0.02 or q_actual > q_max + 0.02:
                    safety_reasons.append(
                        f"{name} position={q_actual:.4f} outside [{q_min:.4f},{q_max:.4f}]"
                    )
                if not math.isfinite(qd_actual):
                    safety_reasons.append(f"{name} velocity feedback is invalid")
                elif abs(qd_actual) > float(config["max_velocity_rad_s"]):
                    safety_reasons.append(f"{name} velocity={qd_actual:.3f}rad/s")
                if math.isfinite(temperature) and temperature > float(config["max_temperature_c"]):
                    safety_reasons.append(f"{name} temperature={temperature:.1f}C")
                if math.isfinite(torque) and abs(torque) > float(config["max_measured_torque_nm"]):
                    safety_reasons.append(f"{name} torque={torque:.3f}Nm")
                if not math.isfinite(fb_age) or fb_age < 0.0:
                    safety_reasons.append(f"{name} feedback timestamp is invalid")
                elif fb_age > float(config["max_feedback_age_s"]):
                    safety_reasons.append(f"{name} feedback_age={fb_age:.4f}s")

                values = {
                    "q_requested": sample.q_requested[i], "q_des": cmd["q_des"],
                    "q_actual": q_actual, "qd_actual": qd_actual,
                    "q_raw": q_raw, "qd_raw": qd_raw,
                    "tracking_error": float(cmd["q_des"]) - q_actual,
                    "limiter_delta": float(cmd["q_des"]) - sample.q_requested[i],
                    "limiter_abs_delta": abs(
                        float(cmd["q_des"]) - sample.q_requested[i]
                    ),
                    "hard_limit_active": bool(diagnostics["hard_limit_active"][i]),
                    "rate_limit_active": bool(diagnostics["rate_limit_active"][i]),
                    "torque_budget_limit_active": bool(
                        diagnostics["torque_budget_limit_active"][i]
                    ),
                    "kp": cmd["kp_effective"], "kd": cmd["kd_effective"],
                    "v_des": cmd["joint_v_des"],
                    "tau_ff": cmd["joint_tau_ff_effective"],
                    "motor_id": cmd["motor_id"], "bus": cmd["bus_name"],
                    "feedback_time_s": fb_abs - start if math.isfinite(fb_abs) else "",
                    "feedback_age_s": fb_age if math.isfinite(fb_age) else "",
                    "torque_nm": torque, "temperature_c": temperature,
                    "fault_bits": fault_bits,
                }
                row.update({f"{name}_{key}": value for key, value in values.items()})

            row["feedback_max_age_s"] = max(
                [float(row[f"{name}_feedback_age_s"]) for name in JOINT_ORDER
                 if row[f"{name}_feedback_age_s"] != ""],
                default=float("nan"),
            )
            limiter_abs_delta = np.abs(q_des_sent - sample.q_requested)
            max_limiter_delta = float(np.max(limiter_abs_delta))
            if max_limiter_delta > warning_delta:
                if sample_index - last_limiter_warning_sample >= status_interval:
                    last_limiter_warning_sample = sample_index
            if max_limiter_delta > abort_delta:
                consecutive_limiter_distortion += 1
            else:
                consecutive_limiter_distortion = 0
            if consecutive_limiter_distortion >= abort_cycles:
                safety_reasons.append(
                    "sustained limiter distortion: "
                    f"max|q_des-q_requested|={max_limiter_delta:.6f}rad "
                    f"for {consecutive_limiter_distortion} cycles"
                )
            max_tracking_error = float(np.max(np.abs(q_des_sent - q_actual_cycle)))
            if max_tracking_error > float(config["max_tracking_error_rad"]):
                consecutive_tracking_error += 1
            else:
                consecutive_tracking_error = 0
            if consecutive_tracking_error >= int(
                config["max_tracking_error_consecutive_cycles"]
            ):
                safety_reasons.append(
                    "sustained tracking error: "
                    f"max|q_des-q_actual|={max_tracking_error:.6f}rad "
                    f"for {consecutive_tracking_error} cycles"
                )
            cycle_end = time.monotonic()
            work_s = cycle_end - cycle_wake
            row["loop_work_s"] = work_s
            row["safety_event"] = "; ".join(safety_reasons)
            logger.write(row)
            loop_dts.append(actual_dt)
            loop_work.append(work_s)
            if not safety_reasons and complete and np.all(np.isfinite(q_actual_cycle)):
                pace_times.append(control_timestamp_abs - start)
                pace_q_des.append(q_des_sent.copy())
                pace_q_actual.append(q_actual_cycle.copy())
            if sample_index % status_interval == 0:
                terminal_monitor.submit(
                    (
                        sample,
                        sample.q_requested.copy(),
                        q_des_sent.copy(),
                        q_actual_cycle.copy(),
                        qd_actual_cycle.copy(),
                        {key: value.copy() for key, value in diagnostics.items()},
                        warning_delta,
                    )
                )
            q_des_previous = q_des_sent
            if safety_reasons:
                raise RuntimeError(row["safety_event"])
            sample_index += 1

        completed = True
        return csv_path, sample_index
    finally:
        if enabled:
            try:
                for _ in range(3):
                    layer.send_raw_commands(buses, layer.build_stop_commands())
                    time.sleep(max(dt, 0.01))
            except Exception as exc:
                print(f"WARNING: motor stop failed: {exc}", file=sys.stderr)
        if gc_disabled_for_control:
            gc.enable()
        if terminal_monitor is not None:
            terminal_monitor.close()
        if logger is not None:
            try:
                logger.close()
            except Exception as exc:
                completed = False
                print(f"WARNING: buffered log write failed: {exc}", file=sys.stderr)
        if completed and csv_path is not None and pace_times:
            try:
                pace_path = save_pace_dataset(
                    csv_path.parent / "chirp_data.pt",
                    pace_times,
                    pace_q_des,
                    pace_q_actual,
                )
                print(f"PACE tensor dataset: {pace_path}")
            except Exception as exc:
                print(f"WARNING: PACE tensor export failed: {exc}", file=sys.stderr)
        elif csv_path is not None and pace_times:
            print(
                "PACE tensor dataset not written because the experiment did not "
                "complete successfully.",
                file=sys.stderr,
            )
        if loop_dts:
            timing_summary(
                control_hz,
                loop_dts,
                loop_work,
                missed_deadlines,
                label="REAL CONTROL",
            )
            print(f"feedback dropout cycles: {feedback_dropout_cycles}")
            print(
                "maximum consecutive incomplete feedback cycles: "
                f"{maximum_consecutive_incomplete_feedback}"
            )
            print(
                "scheduler resynchronizations: "
                f"{scheduler_resynchronizations}"
            )
        try:
            close_can_buses(buses)
        finally:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
