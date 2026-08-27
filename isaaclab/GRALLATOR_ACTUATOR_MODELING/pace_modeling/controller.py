"""50 Hz PACE hardware and dry-run controller."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .constants import JOINT_COUNT, JOINT_ORDER
from .data_logger import PaceDataLogger
from .trajectory import joint_vector, load_trajectory


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


def final_command_target(requested, previous, q_actual, qd_actual, config, dt):
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

    q_des = np.clip(requested, q_min, q_max)
    q_des = np.clip(q_des, previous - max_rate * dt, previous + max_rate * dt)
    q_des = np.clip(q_des, q_min, q_max)

    # Preserve damping first, then restrict the position target to the remaining
    # torque budget. The resulting q_des is what gets logged and transmitted.
    damping = -kd * qd_actual
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
    q_des = np.clip(q_des, q_min, q_max)
    return q_des


def csv_fieldnames():
    fields = [
        "sample_index", "time_s", "nominal_time_s", "command_time_s",
        "feedback_time_s", "control_dt_s", "loop_lateness_s",
        "trajectory_segment", "feedback_complete", "feedback_max_age_s",
        "safety_event",
    ]
    suffixes = [
        "q_requested", "q_des", "q_actual", "qd_actual", "q_raw", "qd_raw",
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
    return {
        "format": "grallator_pace_v1",
        "created_at": datetime.now().isoformat(),
        "dry_run": bool(dry_run),
        "control_rate_hz": 50.0,
        "control_dt_s": 0.02,
        "joint_order": JOINT_ORDER,
        "q_des_definition": "logical q_des from the final MIT command after hard/rate/estimated-torque limiting",
        "q_actual_definition": "real MIT encoder feedback converted using direction and zero offset",
        "q_raw_definition": "mechanical position from MIT operation-status feedback (communication type 2)",
        "qd_raw_definition": "mechanical velocity from MIT operation-status feedback (communication type 2)",
        "parameter_read_note": "0x7019/0x701B parameter reads are not mixed into active MIT streaming",
        "feedback_pairing": "each row pairs one 50 Hz command batch with operation-status replies received after that batch",
        "config_path": str(Path(config_path).resolve()),
        "trajectory_path": str(Path(trajectory_path).resolve()),
        "deploy_root": str(Path(deploy_root).resolve()) if deploy_root else None,
        "deploy_git_commit": git_commit(deploy_root) if deploy_root else None,
        "config": config,
        "trajectory": trajectory_spec,
    }


def _base_row(index, nominal_time, command_time, feedback_time, dt, lateness, segment):
    return {
        "sample_index": index,
        "time_s": command_time,
        "nominal_time_s": nominal_time,
        "command_time_s": command_time,
        "feedback_time_s": feedback_time,
        "control_dt_s": dt,
        "loop_lateness_s": lateness,
        "trajectory_segment": segment,
        "feedback_complete": True,
        "feedback_max_age_s": max(0.0, feedback_time - command_time),
        "safety_event": "",
    }


def run_dry(config, config_path, trajectory_path, output_root, dataset_name, initial_q):
    dt = 0.02
    trajectory_spec, samples = load_trajectory(trajectory_path, initial_q, expected_hz=50.0)
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
    try:
        for sample in samples:
            q_des = final_command_target(
                sample.q_requested, q_des_previous, q_actual, qd_actual, config, dt
            )
            old_q = q_actual.copy()
            q_actual += 0.18 * (q_des - q_actual)
            qd_actual = (q_actual - old_q) / dt
            q_raw = offsets + directions * q_actual
            qd_raw = directions * qd_actual
            row = _base_row(
                sample.index, sample.time_s, sample.time_s, sample.time_s + 0.001,
                dt, 0.0, sample.segment,
            )
            for i, name in enumerate(JOINT_ORDER):
                values = {
                    "q_requested": sample.q_requested[i], "q_des": q_des[i],
                    "q_actual": q_actual[i], "qd_actual": qd_actual[i],
                    "q_raw": q_raw[i], "qd_raw": qd_raw[i], "kp": kp[i],
                    "kd": kd[i], "v_des": 0.0, "tau_ff": 0.0,
                    "motor_id": motor_ids[i], "bus": buses[i],
                    "feedback_time_s": sample.time_s + 0.001,
                    "feedback_age_s": 0.001, "torque_nm": 0.0,
                    "temperature_c": 25.0, "fault_bits": 0,
                }
                row.update({f"{name}_{key}": value for key, value in values.items()})
            logger.write(row)
            q_des_previous = q_des
    finally:
        logger.close()
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
            if abs(float(config["joint_limits"][name][bound]) - float(
                sources["joint_limits"][name][bound]
            )) > 1.0e-9:
                mismatches.append(f"{name} limit.{bound}")
    if mismatches:
        raise RuntimeError(
            "PACE/deployment calibration contract differs: " + ", ".join(mismatches)
        )


def run_hardware(
    config, config_path, trajectory_path, output_root, dataset_name,
    deploy_root, can_front, can_back,
):
    MotorCommandLayer, Estimator, open_can_buses, close_can_buses = _import_deployment(deploy_root)
    verify_deployment_contract(config, deploy_root)
    dt = 0.02
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
    stop_requested = False

    def request_stop(_signal=None, _frame=None):
        nonlocal stop_requested
        stop_requested = True

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        # Passive feedback acquisition: stop/poll frames cannot move the motors.
        for _ in range(10):
            layer.send_raw_commands(buses, layer.build_feedback_poll_commands())
            time.sleep(0.02)
            estimator.refresh_from_bus(timeout=feedback_timeout, expected_bus_motor_ids=expected)
            if len(estimator.last_feedback_by_joint) == JOINT_COUNT:
                break
        if len(estimator.last_feedback_by_joint) != JOINT_COUNT:
            missing = sorted(set(JOINT_ORDER) - set(estimator.last_feedback_by_joint))
            raise RuntimeError(f"initial feedback missing joints: {missing}")

        initial_q = np.asarray(estimator.q_current, dtype=np.float64).copy()
        trajectory_spec, samples = load_trajectory(
            trajectory_path, initial_q, expected_hz=50.0
        )
        print("\nPACE REAL-HARDWARE TEST")
        print("Joint order:", ", ".join(JOINT_ORDER))
        print("Samples:", len(samples), "Duration:", len(samples) * dt, "s")
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

        csv_path = make_output_paths(output_root, dataset_name, dry_run=False)
        logger = PaceDataLogger(
            csv_path,
            csv_fieldnames(),
            metadata(
                config, trajectory_spec, config_path, trajectory_path,
                deploy_root, False,
            ),
        )
        q_des_previous = initial_q.copy()
        start = time.monotonic()
        previous_cycle = start
        consecutive_overruns = 0

        for sample in samples:
            if stop_requested:
                raise RuntimeError("operator stop requested")
            deadline = start + sample.index * dt
            wait = deadline - time.monotonic()
            if wait > 0.0:
                time.sleep(wait)
            cycle_start = time.monotonic()
            actual_dt = cycle_start - previous_cycle if sample.index else dt
            previous_cycle = cycle_start
            lateness = max(0.0, cycle_start - deadline)
            if lateness > float(config.get("max_loop_lateness_s", 0.010)):
                consecutive_overruns += 1
            else:
                consecutive_overruns = 0
            if consecutive_overruns >= int(config.get("max_consecutive_overruns", 3)):
                raise RuntimeError(f"control timing failed: lateness={lateness:.6f}s")

            q_actual_before = np.asarray(estimator.q_current, dtype=np.float64).copy()
            qd_actual_before = np.asarray(estimator.qd_current, dtype=np.float64).copy()
            q_prepared = final_command_target(
                sample.q_requested, q_des_previous, q_actual_before,
                qd_actual_before, config, dt,
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
            estimator.refresh_from_bus(
                timeout=feedback_timeout,
                expected_bus_motor_ids=expected,
            )
            received = set(estimator.last_refresh_current_bus_motor_ids)
            complete = received == expected
            feedback_by_joint = estimator.last_feedback_by_joint
            feedback_times = [
                float(feedback_by_joint[name].get("timestamp", command_send_abs))
                for name in JOINT_ORDER if name in feedback_by_joint
            ]
            feedback_abs = max(feedback_times, default=command_send_abs)
            command_time = command_send_abs - start
            feedback_time = feedback_abs - start
            row = _base_row(
                sample.index, sample.time_s, command_time, feedback_time,
                actual_dt, lateness, sample.segment,
            )
            row["feedback_complete"] = bool(complete)

            safety_reasons = []
            if not complete:
                missing = sorted(expected - received)
                safety_reasons.append(f"incomplete current-cycle feedback: {missing}")
            q_des_sent = np.zeros(JOINT_COUNT, dtype=np.float64)
            for i, name in enumerate(JOINT_ORDER):
                cmd = command_by_joint[name]
                fb = feedback_by_joint.get(name, {})
                fb_abs = float(fb.get("timestamp", float("nan")))
                fb_age = fb_abs - command_send_abs if math.isfinite(fb_abs) else float("nan")
                fault_bits = int(fb.get("fault_bits", 0))
                temperature = float(fb.get("temperature_c", float("nan")))
                q_actual = float(fb.get("joint_position", float("nan")))
                qd_actual = float(fb.get("joint_velocity_mit", float("nan")))
                q_raw = float(fb.get("position_raw", float("nan")))
                qd_raw = float(fb.get("velocity_raw", float("nan")))
                torque = float(fb.get("joint_torque", float("nan")))
                q_des_sent[i] = float(cmd["q_des"])
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
                if math.isfinite(temperature) and temperature > float(config["max_temperature_c"]):
                    safety_reasons.append(f"{name} temperature={temperature:.1f}C")
                if math.isfinite(qd_actual) and abs(qd_actual) > float(config["max_velocity_rad_s"]):
                    safety_reasons.append(f"{name} velocity={qd_actual:.3f}rad/s")
                if math.isfinite(torque) and abs(torque) > float(config["max_measured_torque_nm"]):
                    safety_reasons.append(f"{name} torque={torque:.3f}Nm")
                if math.isfinite(fb_age) and fb_age > float(config["max_feedback_age_s"]):
                    safety_reasons.append(f"{name} feedback_age={fb_age:.4f}s")

                values = {
                    "q_requested": sample.q_requested[i], "q_des": cmd["q_des"],
                    "q_actual": q_actual, "qd_actual": qd_actual,
                    "q_raw": q_raw, "qd_raw": qd_raw,
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
            row["safety_event"] = "; ".join(safety_reasons)
            logger.write(row)
            q_des_previous = q_des_sent
            if safety_reasons:
                raise RuntimeError(row["safety_event"])

        return csv_path, len(samples)
    finally:
        if logger is not None:
            logger.close()
        if enabled:
            try:
                for _ in range(3):
                    layer.send_raw_commands(buses, layer.build_stop_commands())
                    time.sleep(0.02)
            except Exception as exc:
                print(f"WARNING: motor stop failed: {exc}", file=sys.stderr)
        close_can_buses(buses)
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
