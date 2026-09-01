#!/usr/bin/env python3
"""Guarded 12-axis RS04 0.1-5 Hz chirp for real Grallator identification.

The default invocation is a calculation-only dry run. Real CAN interfaces are
opened only with --enable-motors, followed by an exact interactive RUN prompt.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

from pace_modeling.data_logger import PaceDataLogger


ROOT = Path(__file__).resolve().parent
CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ
FREQUENCY_START_HZ = 0.1
FREQUENCY_END_HZ = 5.0
CHIRP_DURATION_S = 20.0

# Exact PACE fitting order. Name-based routing converts this to CAN lanes/IDs.
JOINT_ORDER = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "BR_hip_joint",
    "BR_thigh_joint",
    "BR_calf_joint",
    "BL_hip_joint",
    "BL_thigh_joint",
    "BL_calf_joint",
]
JOINT_INDEX = {name: index for index, name in enumerate(JOINT_ORDER)}

EXPECTED_MOTOR_IDS = {
    "FR_hip_joint": 1,
    "FR_thigh_joint": 2,
    "FR_calf_joint": 3,
    "FL_hip_joint": 4,
    "FL_thigh_joint": 5,
    "FL_calf_joint": 6,
    "BR_hip_joint": 7,
    "BR_thigh_joint": 8,
    "BR_calf_joint": 9,
    "BL_hip_joint": 10,
    "BL_thigh_joint": 11,
    "BL_calf_joint": 12,
}

# Replace these only after the perfect real crouch pose has been verified.
CROUCH_MEAN_Q = {
    "FR_hip_joint": 0.00,
    "FR_thigh_joint": -0.69,
    "FR_calf_joint": -0.70,
    "FL_hip_joint": 0.00,
    "FL_thigh_joint": 0.69,
    "FL_calf_joint": 0.70,
    "BR_hip_joint": 0.00,
    "BR_thigh_joint": -0.69,
    "BR_calf_joint": -0.70,
    "BL_hip_joint": 0.00,
    "BL_thigh_joint": 0.69,
    "BL_calf_joint": 0.70,
}

# Logical joint-space symmetry. Real motor polarity remains exclusively in
# deployment config/motor_directions.yaml and is applied downstream.
CHIRP_DIRECTION = {
    "FL_hip_joint": 1.0,
    "FR_hip_joint": -1.0,
    "BL_hip_joint": 1.0,
    "BR_hip_joint": -1.0,
    "FL_thigh_joint": 1.0,
    "FR_thigh_joint": -1.0,
    "BL_thigh_joint": 1.0,
    "BR_thigh_joint": -1.0,
    "FL_calf_joint": 1.0,
    "FR_calf_joint": -1.0,
    "BL_calf_joint": 1.0,
    "BR_calf_joint": -1.0,
}

DEG = math.pi / 180.0
SAFE_JOINT_LIMITS = {
    "FL_hip_joint": (-0.79, 0.79),
    "FR_hip_joint": (-0.79, 0.79),
    "BL_hip_joint": (-0.79, 0.79),
    "BR_hip_joint": (-0.79, 0.79),
    "FL_thigh_joint": (-1.68, 1.68),
    "FR_thigh_joint": (-1.68, 1.68),
    "BL_thigh_joint": (-1.68, 1.68),
    "BR_thigh_joint": (-1.68, 1.68),
    "FR_calf_joint": (-70.0 * DEG, 2.0 * DEG),
    "FL_calf_joint": (-13.0 * DEG, 64.0 * DEG),
    "BR_calf_joint": (-68.0 * DEG, 8.0 * DEG),
    "BL_calf_joint": (4.0 * DEG, 78.0 * DEG),
}


class SafetyAbort(RuntimeError):
    """Raised when continuing could violate a hardware safety contract."""


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def vector(mapping: dict[str, float]) -> np.ndarray:
    result = np.asarray([float(mapping[name]) for name in JOINT_ORDER], dtype=np.float64)
    if result.shape != (12,) or not np.all(np.isfinite(result)):
        raise ValueError("joint vector must contain 12 finite values")
    return result


def default_deploy_root() -> Path:
    configured = os.environ.get("GRALLATOR_DEPLOY_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        ROOT.parent / "GRALLATOR_DEPLOY",
        Path.home() / "JetsonNanoDeploy",
    ]
    for candidate in candidates:
        if candidate and (candidate / "src" / "motor_command_layer.py").is_file():
            return candidate.resolve()
    return (Path.home() / "JetsonNanoDeploy").resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-motors", action="store_true")
    parser.add_argument("--deploy-root", type=Path, default=default_deploy_root())
    parser.add_argument("--can-front", default="slcan0")
    parser.add_argument("--can-back", default="slcan1")
    parser.add_argument("--can-bitrate", type=int, default=1_000_000)
    parser.add_argument("--kp", type=float, default=250.0)
    parser.add_argument("--kd", type=float, default=4.0)
    parser.add_argument("--torque-limit", type=float, default=100.0)
    parser.add_argument("--hip-amplitude", type=float, default=0.25)
    parser.add_argument("--thigh-amplitude", type=float, default=0.25)
    parser.add_argument("--calf-amplitude", type=float, default=0.25)
    parser.add_argument("--transition-seconds", type=float, default=4.0)
    parser.add_argument("--pre-chirp-hold-seconds", type=float, default=2.0)
    parser.add_argument("--post-chirp-hold-seconds", type=float, default=1.0)
    parser.add_argument("--max-velocity", type=float, default=18.0)
    parser.add_argument("--max-tracking-error", type=float, default=0.35)
    parser.add_argument("--max-temperature", type=float, default=70.0)
    parser.add_argument("--feedback-timeout", type=float, default=0.012)
    parser.add_argument("--max-feedback-age", type=float, default=0.020)
    parser.add_argument("--max-loop-lateness", type=float, default=0.010)
    parser.add_argument("--max-consecutive-overruns", type=int, default=3)
    parser.add_argument("--status-hz", type=float, default=5.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "actuator_modeling_logs" / "real_chirp_0_5hz",
    )
    return parser.parse_args(argv)


def amplitude_vector(args) -> np.ndarray:
    return np.asarray(
        [
            args.hip_amplitude if "_hip_" in name
            else args.thigh_amplitude if "_thigh_" in name
            else args.calf_amplitude
            for name in JOINT_ORDER
        ],
        dtype=np.float64,
    )


def validate_arguments(args):
    positive = {
        "kp": args.kp,
        "torque_limit": args.torque_limit,
        "transition_seconds": args.transition_seconds,
        "pre_chirp_hold_seconds": args.pre_chirp_hold_seconds,
        "post_chirp_hold_seconds": args.post_chirp_hold_seconds,
        "max_velocity": args.max_velocity,
        "max_tracking_error": args.max_tracking_error,
        "max_temperature": args.max_temperature,
        "feedback_timeout": args.feedback_timeout,
        "max_feedback_age": args.max_feedback_age,
        "max_loop_lateness": args.max_loop_lateness,
        "status_hz": args.status_hz,
    }
    for label, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"--{label.replace('_', '-')} must be finite and > 0")
    if not math.isfinite(args.kd) or args.kd < 0.0:
        raise ValueError("--kd must be finite and >= 0")
    amplitudes = amplitude_vector(args)
    if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0.0):
        raise ValueError("chirp amplitudes must be finite and >= 0")
    if args.torque_limit > 120.0:
        raise ValueError("--torque-limit cannot exceed the RS04 protocol range of 120 Nm")
    if args.max_consecutive_overruns < 1:
        raise ValueError("--max-consecutive-overruns must be >= 1")


def validate_target(q: np.ndarray, label: str):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (12,) or not np.all(np.isfinite(q)):
        raise SafetyAbort(f"{label}: target contains invalid values")
    violations = []
    for index, name in enumerate(JOINT_ORDER):
        lower, upper = SAFE_JOINT_LIMITS[name]
        if q[index] < lower or q[index] > upper:
            violations.append(
                f"{name}={q[index]:+.5f} outside [{lower:+.5f}, {upper:+.5f}]"
            )
    if violations:
        raise SafetyAbort(f"{label}: " + "; ".join(violations))


def validate_measured_state(q, qd, args, label):
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    if q.shape != (12,) or qd.shape != (12,):
        raise SafetyAbort(f"{label}: feedback shape is invalid")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
        raise SafetyAbort(f"{label}: encoder feedback contains NaN or Inf")
    validate_target(q, f"{label} actual position")
    fastest = int(np.argmax(np.abs(qd)))
    if abs(qd[fastest]) > args.max_velocity:
        raise SafetyAbort(
            f"{label}: {JOINT_ORDER[fastest]} velocity={qd[fastest]:+.4f} rad/s "
            f"exceeds {args.max_velocity:.4f} rad/s"
        )


def minimum_jerk(u: float) -> float:
    u = min(1.0, max(0.0, float(u)))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def chirp_values(t: float, q_mean, amplitudes, directions):
    t = min(CHIRP_DURATION_S, max(0.0, float(t)))
    slope = (FREQUENCY_END_HZ - FREQUENCY_START_HZ) / CHIRP_DURATION_S
    phase = 2.0 * math.pi * (FREQUENCY_START_HZ * t + 0.5 * slope * t * t)
    frequency = FREQUENCY_START_HZ + slope * t
    q_des = q_mean + directions * amplitudes * math.sin(phase)
    return q_des, frequency, phase


def preflight(args):
    q_mean = vector(CROUCH_MEAN_Q)
    amplitudes = amplitude_vector(args)
    directions = vector(CHIRP_DIRECTION)
    validate_target(q_mean, "CROUCH_MEAN_Q")
    validate_target(q_mean - amplitudes, "chirp lower envelope")
    validate_target(q_mean + amplitudes, "chirp upper envelope")
    peak_command_speed = 2.0 * math.pi * FREQUENCY_END_HZ * amplitudes
    if np.any(peak_command_speed >= args.max_velocity):
        name = JOINT_ORDER[int(np.argmax(peak_command_speed))]
        raise ValueError(
            f"commanded {name} peak speed {np.max(peak_command_speed):.3f} rad/s "
            f"reaches the emergency threshold {args.max_velocity:.3f} rad/s"
        )
    # Sample every control instant as a second guard against formula mistakes.
    for index in range(int(CHIRP_DURATION_S * CONTROL_HZ) + 1):
        q_des, _, _ = chirp_values(index * CONTROL_DT, q_mean, amplitudes, directions)
        validate_target(q_des, f"chirp sample {index}")
    return q_mean, amplitudes, directions, peak_command_speed


def print_configuration(args, q_mean, amplitudes, directions, peak_speed):
    print("\nGRALLATOR REAL RS04 CHIRP 0.1-5.0 HZ")
    print("Mode:", "REAL HARDWARE REQUESTED" if args.enable_motors else "DRY RUN")
    print(f"Control rate: {CONTROL_HZ:.1f} Hz ({CONTROL_DT:.3f} s)")
    print(f"Frequency: {FREQUENCY_START_HZ:.1f} -> {FREQUENCY_END_HZ:.1f} Hz")
    print(f"Chirp duration: {CHIRP_DURATION_S:.1f} s")
    print(f"Transition / holds: {args.transition_seconds:.1f} / "
          f"{args.pre_chirp_hold_seconds:.1f} / {args.post_chirp_hold_seconds:.1f} s")
    print(f"Kp={args.kp:.3f} Kd={args.kd:.3f} torque ceiling={args.torque_limit:.3f} Nm")
    print(f"Measured velocity abort: {args.max_velocity:.3f} rad/s")
    print(f"Tracking-error abort: {args.max_tracking_error:.3f} rad")
    print("\nJoint                 mean rad   amplitude  direction  safe range rad")
    for index, name in enumerate(JOINT_ORDER):
        lower, upper = SAFE_JOINT_LIMITS[name]
        print(
            f"{name:20s} {q_mean[index]:+9.4f} {amplitudes[index]:9.4f} "
            f"{directions[index]:+9.0f} [{lower:+.4f}, {upper:+.4f}]"
        )
    print("Maximum analytical commanded speed:", f"{np.max(peak_speed):.4f} rad/s")
    if not args.enable_motors:
        print("\nDRY RUN COMPLETE: no CAN interface was opened and no motor was enabled.")


def import_deployment(deploy_root: Path):
    deploy_root = deploy_root.expanduser().resolve()
    src = deploy_root / "src"
    if not (src / "motor_command_layer.py").is_file():
        raise FileNotFoundError(f"deployment source not found under {src}")
    sys.path.insert(0, str(src))
    from can_topology import close_can_buses, open_can_buses
    from motor_command_layer import MotorCommandLayer
    from state_estimator import MitFeedbackStateEstimator
    return MotorCommandLayer, MitFeedbackStateEstimator, open_can_buses, close_can_buses


def deployment_configuration(deploy_root: Path):
    config_dir = deploy_root / "config"
    motor_ids = load_yaml(config_dir / "motor_ids.yaml")["motor_ids"]
    directions = load_yaml(config_dir / "motor_directions.yaml")["motor_directions"]
    offsets = load_yaml(config_dir / "joint_offsets.yaml")["joint_offsets"]
    for name, expected in EXPECTED_MOTOR_IDS.items():
        actual = int(motor_ids.get(name, -1))
        if actual != expected:
            raise RuntimeError(f"{name}: deployment motor ID {actual} != required {expected}")
        if float(directions.get(name, 0.0)) not in (-1.0, 1.0):
            raise RuntimeError(f"{name}: deployment motor direction is not +/-1")
        if not math.isfinite(float(offsets.get(name, float("nan")))):
            raise RuntimeError(f"{name}: deployment encoder offset is invalid")
    routing = {
        name: "front" if int(motor_ids[name]) <= 6 else "back"
        for name in JOINT_ORDER
    }
    return (
        {name: int(motor_ids[name]) for name in JOINT_ORDER},
        {name: float(directions[name]) for name in JOINT_ORDER},
        {name: float(offsets[name]) for name in JOINT_ORDER},
        routing,
    )


def git_commit(path: Path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        ).stdout.strip()
    except Exception:
        return None


def fieldnames():
    fields = [
        "sample_index",
        "timestamp_wall",
        "timestamp_monotonic_s",
        "experiment_time_s",
        "chirp_time_s",
        "instantaneous_frequency_hz",
        "control_dt_s",
        "loop_lateness_s",
        "segment",
        "feedback_complete",
        "feedback_max_age_s",
        "safety_event",
    ]
    suffixes = [
        "q_des_rad",
        "q_rad",
        "qd_rad_s",
        "position_error_rad",
        "tau_cmd_nm",
        "q_raw_rad",
        "qd_raw_rad_s",
        "tau_measured_nm",
        "kp_effective",
        "kd_effective",
        "temperature_c",
        "fault_bits",
    ]
    fields.extend(f"{name}_{suffix}" for name in JOINT_ORDER for suffix in suffixes)
    return fields


def make_logger(args, deploy_root, motor_ids, directions, offsets, routing):
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = args.output_dir.expanduser() / f"grallator_real_chirp_0_5hz_{stamp}.csv"
    metadata = {
        "format": "grallator_real_chirp_0_5hz_v1",
        "created_at": datetime.now().isoformat(),
        "joint_order": JOINT_ORDER,
        "motor_ids": motor_ids,
        "joint_can_bus": routing,
        "motor_directions": directions,
        "joint_offsets": offsets,
        "crouch_mean_q": CROUCH_MEAN_Q,
        "chirp_direction": CHIRP_DIRECTION,
        "safe_joint_limits_rad": SAFE_JOINT_LIMITS,
        "control_hz": CONTROL_HZ,
        "frequency_start_hz": FREQUENCY_START_HZ,
        "frequency_end_hz": FREQUENCY_END_HZ,
        "chirp_duration_s": CHIRP_DURATION_S,
        "kp_requested": args.kp,
        "kd_requested": args.kd,
        "torque_limit_nm": args.torque_limit,
        "feedforward_torque_nm": 0.0,
        "desired_velocity_rad_s": 0.0,
        "amplitude_rad": {
            "hip": args.hip_amplitude,
            "thigh": args.thigh_amplitude,
            "calf": args.calf_amplitude,
        },
        "safety": {
            "max_velocity_rad_s": args.max_velocity,
            "max_tracking_error_rad": args.max_tracking_error,
            "max_temperature_c": args.max_temperature,
            "max_feedback_age_s": args.max_feedback_age,
        },
        "deploy_root": str(deploy_root),
        "deploy_git_commit": git_commit(deploy_root),
        "torque_command_definition": (
            "post-limiter MotorCommandLayer tau_pd_est from q_des, feedback, "
            "effective Kp/Kd, v_des=0, tau_ff=0"
        ),
        "torque_speed_note": (
            "No fabricated torque-speed derating is applied below 95 rpm; "
            "only the configured absolute software torque ceiling is active."
        ),
    }
    return PaceDataLogger(path, fieldnames(), metadata), path


def save_pace_data(path, time_samples, desired_samples, measured_samples):
    """Atomically save the exact three-key tensor contract consumed by PACE."""
    if not time_samples:
        return None
    import torch

    time_array = np.asarray(time_samples, dtype=np.float64)
    desired_array = np.asarray(desired_samples, dtype=np.float32)
    measured_array = np.asarray(measured_samples, dtype=np.float32)
    expected_shape = (len(time_array), len(JOINT_ORDER))
    if desired_array.shape != expected_shape or measured_array.shape != expected_shape:
        raise RuntimeError(
            "PACE buffer shape mismatch: "
            f"time={time_array.shape}, desired={desired_array.shape}, "
            f"measured={measured_array.shape}"
        )
    if not (
        np.all(np.isfinite(time_array))
        and np.all(np.isfinite(desired_array))
        and np.all(np.isfinite(measured_array))
    ):
        raise RuntimeError("PACE buffers contain NaN or Inf")
    if len(time_array) > 1 and np.any(np.diff(time_array) <= 0.0):
        raise RuntimeError("PACE monotonic timestamps are not strictly increasing")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "time": torch.from_numpy(time_array),
            "des_dof_pos": torch.from_numpy(desired_array),
            "dof_pos": torch.from_numpy(measured_array),
        },
        temporary,
    )
    os.replace(temporary, path)
    return path


def configure_layer(layer, args, motor_ids, directions, offsets):
    layer.joint_directions = dict(directions)
    layer.joint_offsets = dict(offsets)
    layer.hard_joint_limits = dict(SAFE_JOINT_LIMITS)
    layer.gains["policy"] = {
        "hip": {"kp": args.kp, "kd": args.kd},
        "thigh": {"kp": args.kp, "kd": args.kd},
        "calf": {"kp": args.kp, "kd": args.kd},
        "joints": {
            name: {"kp": args.kp, "kd": args.kd}
            for name in motor_ids
        },
    }
    layer.feedforward = {"v_des": 0.0, "tau_ff": 0.0}
    layer.virtual_joint_stop_enabled = False
    layer.policy_pd_torque_limit = float(args.torque_limit)
    layer.policy_pd_torque_limits = {
        name: float(args.torque_limit) for name in motor_ids
    }
    layer.policy_pd_torque_limit_start = dict(layer.policy_pd_torque_limits)
    layer.policy_pd_torque_limit_final = dict(layer.policy_pd_torque_limits)


def configure_feedback_filters(buses, layer):
    feedback_types = (
        int(layer.proto.get("comm_type_feedback", 2)),
        int(layer.proto.get("comm_type_active_feedback", 24)),
    )
    seen = set()
    for bus in buses.values():
        if id(bus) in seen:
            continue
        seen.add(id(bus))
        configure = getattr(bus, "configure_feedback_filters", None)
        if configure is None:
            raise RuntimeError("SocketCAN adapter lacks configure_feedback_filters")
        configure(feedback_types)


def acquire_initial_feedback(layer, estimator, buses, expected, timeout):
    for _ in range(12):
        layer.send_raw_commands(buses, layer.build_feedback_poll_commands())
        time.sleep(0.02)
        estimator.refresh_from_bus(
            timeout=timeout,
            expected_bus_motor_ids=expected,
        )
        if len(estimator.last_feedback_by_joint) == len(JOINT_ORDER):
            break
    missing = sorted(set(JOINT_ORDER) - set(estimator.last_feedback_by_joint))
    if missing:
        raise SafetyAbort(f"initial feedback missing motors: {missing}")


def wait_until(deadline: float):
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return
        if remaining > 0.001:
            time.sleep(remaining - 0.0005)


def requested_target(elapsed, initial_q, q_mean, amplitudes, directions, args):
    transition_end = args.transition_seconds
    pre_hold_end = transition_end + args.pre_chirp_hold_seconds
    chirp_end = pre_hold_end + CHIRP_DURATION_S
    return_end = chirp_end + args.transition_seconds
    final_end = return_end + args.post_chirp_hold_seconds
    if elapsed < transition_end:
        blend = minimum_jerk(elapsed / args.transition_seconds)
        return initial_q + blend * (q_mean - initial_q), "transition_to_crouch", float("nan"), float("nan")
    if elapsed < pre_hold_end:
        return q_mean.copy(), "pre_chirp_hold", float("nan"), float("nan")
    if elapsed < chirp_end:
        chirp_time = elapsed - pre_hold_end
        q_des, frequency, _ = chirp_values(chirp_time, q_mean, amplitudes, directions)
        return q_des, "chirp", chirp_time, frequency
    q_chirp_end, _, _ = chirp_values(CHIRP_DURATION_S, q_mean, amplitudes, directions)
    if elapsed < return_end:
        blend = minimum_jerk((elapsed - chirp_end) / args.transition_seconds)
        return q_chirp_end + blend * (q_mean - q_chirp_end), "return_to_crouch", float("nan"), float("nan")
    if elapsed < final_end:
        return q_mean.copy(), "post_chirp_hold", float("nan"), float("nan")
    return None, "complete", float("nan"), float("nan")


def stop_motors(layer, buses):
    if layer is None or not buses:
        return
    try:
        layer.stop_periodic_commands(buses)
    except Exception:
        pass
    errors = []
    for _ in range(3):
        try:
            layer.send_raw_commands(buses, layer.build_stop_commands())
        except Exception as exc:
            errors.append(str(exc))
        time.sleep(0.02)
    if errors:
        print("WARNING: one or more motor stop batches failed:", errors, file=sys.stderr)


def run_hardware(args, q_mean, amplitudes, directions):
    deploy_root = args.deploy_root.expanduser().resolve()
    MotorCommandLayer, Estimator, open_can_buses, close_can_buses = import_deployment(
        deploy_root
    )
    motor_ids, motor_directions, offsets, routing = deployment_configuration(deploy_root)
    layer = MotorCommandLayer(JOINT_ORDER, motor_ids, joint_can_bus=routing)
    configure_layer(layer, args, motor_ids, motor_directions, offsets)
    buses = {}
    logger = None
    output_path = None
    pace_path = None
    enabled = False
    stop_requested = False
    safety_event = ""
    pace_time = []
    pace_desired = []
    pace_measured = []

    def request_stop(_signal=None, _frame=None):
        nonlocal stop_requested
        stop_requested = True

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        buses = open_can_buses(
            {"front": args.can_front, "back": args.can_back},
            backend="socketcan",
            bitrate=args.can_bitrate,
            timeout=args.feedback_timeout,
        )
        configure_feedback_filters(buses, layer)
        estimator = Estimator(
            q_initial=np.zeros(12, dtype=np.float32),
            policy_order=JOINT_ORDER,
            motor_ids=motor_ids,
            motor_layer=layer,
            bus=buses,
            joint_velocity_source="mit",
        )
        expected = estimator.expected_feedback_bus_motor_ids()
        acquire_initial_feedback(
            layer, estimator, buses, expected, args.feedback_timeout
        )
        initial_q = np.asarray(estimator.q_current, dtype=np.float64).copy()
        initial_qd = np.asarray(estimator.qd_current, dtype=np.float64).copy()
        validate_measured_state(initial_q, initial_qd, args, "startup")
        print("\nMeasured starting joint positions:")
        for index, name in enumerate(JOINT_ORDER):
            print(
                f"  {name:20s} {initial_q[index]:+9.5f} rad "
                f"({math.degrees(initial_q[index]):+8.3f} deg)"
            )
        print("\nThe robot must be unloaded/suspended and the area clear.")
        confirmation = input("Type RUN to start real-hardware chirp: ").strip()
        if confirmation != "RUN":
            print("Confirmation rejected; motors remain disabled.")
            return None, None, 0

        for _ in range(3):
            layer.send_raw_commands(buses, layer.build_enable_commands())
            time.sleep(0.03)
        enabled = True
        estimator.refresh_from_bus(timeout=0.0, expected_bus_motor_ids=expected)
        logger, output_path = make_logger(
            args, deploy_root, motor_ids, motor_directions, offsets, routing
        )
        pace_path = output_path.parent / "chirp_data.pt"
        total_duration = (
            2.0 * args.transition_seconds
            + args.pre_chirp_hold_seconds
            + CHIRP_DURATION_S
            + args.post_chirp_hold_seconds
        )
        sample_count = int(math.ceil(total_duration * CONTROL_HZ)) + 1
        start_abs = time.monotonic()
        previous_cycle_abs = start_abs
        consecutive_overruns = 0
        next_status_abs = start_abs

        for sample_index in range(sample_count):
            if stop_requested:
                raise SafetyAbort("operator requested stop")
            deadline = start_abs + sample_index * CONTROL_DT
            wait_until(deadline)
            cycle_abs = time.monotonic()
            elapsed = cycle_abs - start_abs
            actual_dt = CONTROL_DT if sample_index == 0 else cycle_abs - previous_cycle_abs
            previous_cycle_abs = cycle_abs
            lateness = max(0.0, cycle_abs - deadline)
            if lateness > args.max_loop_lateness:
                consecutive_overruns += 1
            else:
                consecutive_overruns = 0
            if consecutive_overruns >= args.max_consecutive_overruns:
                raise SafetyAbort(
                    f"control timing missed {consecutive_overruns} consecutive cycles; "
                    f"lateness={lateness * 1000.0:.2f} ms"
                )

            q_requested, segment, chirp_time, frequency = requested_target(
                elapsed, initial_q, q_mean, amplitudes, directions, args
            )
            if q_requested is None:
                break
            validate_target(q_requested, f"sample {sample_index} desired position")
            feedback_before = estimator.last_feedback_by_joint
            q_before = np.asarray(estimator.q_current, dtype=np.float64).copy()
            qd_before = np.asarray(estimator.qd_current, dtype=np.float64).copy()
            validate_measured_state(q_before, qd_before, args, f"sample {sample_index}")
            commands = layer.build_mit_commands(
                q_requested,
                phase="policy",
                feedback_by_joint=feedback_before,
            )
            command_by_joint = {command["joint_name"]: command for command in commands}
            for index, name in enumerate(JOINT_ORDER):
                command = command_by_joint[name]
                if abs(float(command["q_des"]) - q_requested[index]) > 1.0e-6:
                    raise SafetyAbort(
                        f"{name}: command path altered q_des from "
                        f"{q_requested[index]:+.6f} to {float(command['q_des']):+.6f}"
                    )
                if abs(float(command["joint_tau_ff_effective"])) > 1.0e-9:
                    raise SafetyAbort(f"{name}: nonzero feedforward torque detected")
                tau_est = float(command["tau_pd_est"])
                if not math.isfinite(tau_est) or abs(tau_est) > args.torque_limit + 1.0e-5:
                    raise SafetyAbort(
                        f"{name}: estimated command torque {tau_est:+.3f} Nm exceeds "
                        f"{args.torque_limit:.3f} Nm"
                    )

            command_send_abs = time.monotonic()
            layer.send_raw_commands(buses, commands)
            estimator.mark_command_sent(command_send_abs)
            estimator.refresh_from_bus(
                timeout=args.feedback_timeout,
                expected_bus_motor_ids=expected,
            )
            received = set(estimator.last_refresh_current_bus_motor_ids)
            complete = received == expected
            if not complete:
                raise SafetyAbort(
                    "incomplete current-cycle feedback: "
                    + repr(sorted(expected - received))
                )

            feedback = estimator.last_feedback_by_joint
            q_after = np.asarray(estimator.q_current, dtype=np.float64).copy()
            qd_after = np.asarray(estimator.qd_current, dtype=np.float64).copy()
            q_des_sent = np.asarray(
                [float(command_by_joint[name]["q_des"]) for name in JOINT_ORDER]
            )
            validate_measured_state(q_after, qd_after, args, f"sample {sample_index}")
            # These values come from this real cycle, not a reconstructed 50 Hz
            # timebase. Append only after all 12 current-cycle replies exist.
            pace_time.append(command_send_abs - start_abs)
            pace_desired.append(q_des_sent.copy())
            pace_measured.append(q_after.copy())
            errors = q_des_sent - q_after
            worst_error_index = int(np.argmax(np.abs(errors)))
            if abs(errors[worst_error_index]) > args.max_tracking_error:
                raise SafetyAbort(
                    f"{JOINT_ORDER[worst_error_index]} tracking error "
                    f"{errors[worst_error_index]:+.4f} rad exceeds "
                    f"{args.max_tracking_error:.4f} rad"
                )

            feedback_times = [float(feedback[name]["timestamp"]) for name in JOINT_ORDER]
            feedback_ages = [stamp - command_send_abs for stamp in feedback_times]
            maximum_feedback_age = max(feedback_ages)
            if maximum_feedback_age > args.max_feedback_age:
                raise SafetyAbort(
                    f"feedback age {maximum_feedback_age:.4f}s exceeds "
                    f"{args.max_feedback_age:.4f}s"
                )

            row = {
                "sample_index": sample_index,
                "timestamp_wall": datetime.now().isoformat(),
                "timestamp_monotonic_s": command_send_abs,
                "experiment_time_s": command_send_abs - start_abs,
                "chirp_time_s": chirp_time if math.isfinite(chirp_time) else "",
                "instantaneous_frequency_hz": frequency if math.isfinite(frequency) else "",
                "control_dt_s": actual_dt,
                "loop_lateness_s": lateness,
                "segment": segment,
                "feedback_complete": complete,
                "feedback_max_age_s": maximum_feedback_age,
                "safety_event": "",
            }
            for index, name in enumerate(JOINT_ORDER):
                motor_feedback = feedback[name]
                command = command_by_joint[name]
                temperature = float(motor_feedback.get("temperature_c", float("nan")))
                fault_bits = int(motor_feedback.get("fault_bits", 0))
                if fault_bits:
                    raise SafetyAbort(f"{name}: motor fault 0x{fault_bits:X}")
                if math.isfinite(temperature) and temperature > args.max_temperature:
                    raise SafetyAbort(
                        f"{name}: temperature {temperature:.1f} C exceeds "
                        f"{args.max_temperature:.1f} C"
                    )
                values = {
                    "q_des_rad": q_des_sent[index],
                    "q_rad": q_after[index],
                    "qd_rad_s": qd_after[index],
                    "position_error_rad": errors[index],
                    "tau_cmd_nm": float(command["tau_pd_est"]),
                    "q_raw_rad": float(motor_feedback.get("position_raw", float("nan"))),
                    "qd_raw_rad_s": float(motor_feedback.get("velocity_raw", float("nan"))),
                    "tau_measured_nm": float(motor_feedback.get("joint_torque", float("nan"))),
                    "kp_effective": float(command["kp_effective"]),
                    "kd_effective": float(command["kd_effective"]),
                    "temperature_c": temperature,
                    "fault_bits": fault_bits,
                }
                row.update({f"{name}_{key}": value for key, value in values.items()})
            logger.write(row)

            if cycle_abs >= next_status_abs:
                print(
                    f"t={elapsed:6.2f}s segment={segment:20s} "
                    f"f={frequency if math.isfinite(frequency) else 0.0:4.2f}Hz "
                    f"err_max={np.max(np.abs(errors)):.3f}rad "
                    f"vel_max={np.max(np.abs(qd_after)):.2f}rad/s"
                )
                next_status_abs = cycle_abs + 1.0 / args.status_hz

        return output_path, pace_path, sample_index + 1
    except BaseException as exc:
        safety_event = str(exc)
        print(f"\nSAFETY ABORT: {safety_event}", file=sys.stderr)
        raise
    finally:
        if buses:
            stop_motors(layer, buses)
            close_can_buses(buses)
        if logger is not None:
            logger.close()
        if pace_path is not None and pace_time:
            saved_path = save_pace_data(
                pace_path,
                pace_time,
                pace_desired,
                pace_measured,
            )
            print("PACE tensor data:", saved_path)
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if enabled:
            print("Motors stopped and disabled.")
        if safety_event and output_path:
            print("Partial log:", output_path)


def main(argv=None):
    args = parse_args(argv)
    validate_arguments(args)
    q_mean, amplitudes, directions, peak_speed = preflight(args)
    print_configuration(args, q_mean, amplitudes, directions, peak_speed)
    if not args.enable_motors:
        return 0
    output_path, pace_path, count = run_hardware(
        args, q_mean, amplitudes, directions
    )
    if output_path is not None:
        print("Chirp log:", output_path)
        print("PACE data:", pace_path)
        print("Logged samples:", count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
