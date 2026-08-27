"""Deterministic 50 Hz waypoint and excitation trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .constants import JOINT_COUNT, JOINT_ORDER


@dataclass(frozen=True)
class TrajectorySample:
    index: int
    time_s: float
    segment: str
    q_requested: np.ndarray


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def joint_vector(value, base=None, field="joint vector"):
    if base is None:
        result = np.zeros(JOINT_COUNT, dtype=np.float64)
    else:
        result = np.asarray(base, dtype=np.float64).copy()
    if value is None:
        return result
    if isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (JOINT_COUNT,):
            raise ValueError(f"{field} must contain exactly {JOINT_COUNT} values")
        result[:] = array
    elif isinstance(value, dict):
        unknown = sorted(set(value) - set(JOINT_ORDER))
        if unknown:
            raise ValueError(f"{field} contains unknown joints: {unknown}")
        for name, number in value.items():
            result[JOINT_ORDER.index(name)] = float(number)
    else:
        raise ValueError(f"{field} must be a 12-value list or joint-name mapping")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field} contains NaN or Inf")
    return result


def _chirp_phase(t, duration, f_start, f_end, law):
    if f_start <= 0.0 or f_end <= 0.0:
        raise ValueError("chirp frequencies must be positive")
    if law == "linear":
        slope = (f_end - f_start) / duration
        return 2.0 * math.pi * (f_start * t + 0.5 * slope * t * t)
    if law == "logarithmic":
        if abs(f_end - f_start) < 1.0e-12:
            return 2.0 * math.pi * f_start * t
        beta = math.log(f_end / f_start) / duration
        return 2.0 * math.pi * f_start * math.expm1(beta * t) / beta
    raise ValueError("chirp law must be linear or logarithmic")


def build_trajectory(spec, initial_q, expected_hz=50.0):
    configured_hz = float(spec.get("control_hz", expected_hz))
    if abs(configured_hz - float(expected_hz)) > 1.0e-9:
        raise ValueError(
            f"trajectory control_hz={configured_hz} does not match required {expected_hz} Hz"
        )
    segments = spec.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("trajectory needs a non-empty segments list")

    dt = 1.0 / configured_hz
    current = np.asarray(initial_q, dtype=np.float64).copy()
    if current.shape != (JOINT_COUNT,):
        raise ValueError(f"initial_q must have shape ({JOINT_COUNT},)")

    samples = []
    global_index = 0
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segment {segment_index} must be a mapping")
        kind = str(segment.get("type", "hold")).strip().lower()
        name = str(segment.get("name", f"segment_{segment_index}_{kind}"))
        duration = float(segment.get("duration_s", 0.0))
        count = int(round(duration * configured_hz))
        if duration <= 0.0 or count <= 0:
            raise ValueError(f"{name}: duration_s must produce at least one sample")

        start = current.copy()
        if kind in ("hold", "linear", "smoothstep", "minimum_jerk"):
            has_target = "target" in segment
            has_relative_target = "relative_target" in segment
            if has_target and has_relative_target:
                raise ValueError(
                    f"{name}: use either target or relative_target, not both"
                )
            if has_relative_target:
                delta = joint_vector(
                    segment.get("relative_target"),
                    field=f"{name}.relative_target",
                )
                target = start + delta
            else:
                target = joint_vector(
                    segment.get("target"), base=start, field=f"{name}.target"
                )
            for local_index in range(count):
                if kind == "hold":
                    q = target
                else:
                    alpha = float(local_index + 1) / float(count)
                    if kind == "smoothstep":
                        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                    elif kind == "minimum_jerk":
                        alpha = (
                            10.0 * alpha**3
                            - 15.0 * alpha**4
                            + 6.0 * alpha**5
                        )
                    q = start + alpha * (target - start)
                samples.append(TrajectorySample(
                    index=global_index,
                    time_s=global_index * dt,
                    segment=name,
                    q_requested=np.asarray(q, dtype=np.float64).copy(),
                ))
                global_index += 1
            current = target.copy()
            continue

        if kind in ("sine", "chirp"):
            center = joint_vector(segment.get("center"), base=start, field=f"{name}.center")
            amplitude = joint_vector(segment.get("amplitude"), field=f"{name}.amplitude")
            phase_offset = float(segment.get("phase_rad", 0.0))
            f_start = float(segment.get("frequency_hz", segment.get("f_start_hz", 0.5)))
            f_end = float(segment.get("f_end_hz", f_start))
            law = str(segment.get("law", "linear")).strip().lower()
            for local_index in range(count):
                local_time = local_index * dt
                if kind == "sine":
                    phase = 2.0 * math.pi * f_start * local_time
                else:
                    phase = _chirp_phase(local_time, duration, f_start, f_end, law)
                q = center + amplitude * math.sin(phase + phase_offset)
                samples.append(TrajectorySample(
                    index=global_index,
                    time_s=global_index * dt,
                    segment=name,
                    q_requested=np.asarray(q, dtype=np.float64).copy(),
                ))
                global_index += 1
            current = samples[-1].q_requested.copy()
            continue

        raise ValueError(f"{name}: unsupported segment type {kind!r}")

    return samples


def load_trajectory(path, initial_q, expected_hz=50.0):
    spec = load_yaml(path)
    return spec, build_trajectory(spec, initial_q, expected_hz=expected_hz)
