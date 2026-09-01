"""Time-based waypoint and excitation trajectories."""

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
    segment_time_s: float = 0.0
    instantaneous_frequency_hz: float | None = None


@dataclass(frozen=True)
class CompiledSegment:
    name: str
    kind: str
    start_time_s: float
    duration_s: float
    start: np.ndarray
    target: np.ndarray | None = None
    center: np.ndarray | None = None
    amplitude: np.ndarray | None = None
    phase_offset_rad: float = 0.0
    f_start_hz: float | None = None
    f_end_hz: float | None = None
    law: str = "linear"


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


class TrajectoryPlan:
    """Compiled trajectory evaluated from actual monotonic elapsed time."""

    def __init__(self, control_hz, segments):
        self.control_hz = float(control_hz)
        self.control_dt_s = 1.0 / self.control_hz
        self.segments = tuple(segments)
        self.duration_s = sum(segment.duration_s for segment in self.segments)

    def evaluate(self, elapsed_s, index=0):
        elapsed_s = min(max(float(elapsed_s), 0.0), self.duration_s)
        segment = self.segments[-1]
        for candidate in self.segments:
            if elapsed_s < candidate.start_time_s + candidate.duration_s:
                segment = candidate
                break
        local_time = min(
            max(elapsed_s - segment.start_time_s, 0.0), segment.duration_s
        )
        frequency = None
        if segment.kind == "hold":
            q = segment.target
        elif segment.kind in ("linear", "smoothstep", "minimum_jerk"):
            alpha = local_time / segment.duration_s
            if segment.kind == "smoothstep":
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            elif segment.kind == "minimum_jerk":
                alpha = 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5
            q = segment.start + alpha * (segment.target - segment.start)
        elif segment.kind in ("sine", "chirp"):
            if segment.kind == "sine":
                phase = 2.0 * math.pi * segment.f_start_hz * local_time
                frequency = segment.f_start_hz
            else:
                phase = _chirp_phase(
                    local_time,
                    segment.duration_s,
                    segment.f_start_hz,
                    segment.f_end_hz,
                    segment.law,
                )
                if segment.law == "linear":
                    slope = (
                        segment.f_end_hz - segment.f_start_hz
                    ) / segment.duration_s
                    frequency = segment.f_start_hz + slope * local_time
                else:
                    frequency = segment.f_start_hz * (
                        segment.f_end_hz / segment.f_start_hz
                    ) ** (local_time / segment.duration_s)
            q = segment.center + segment.amplitude * math.sin(
                phase + segment.phase_offset_rad
            )
        else:
            raise RuntimeError(f"unsupported compiled segment {segment.kind}")
        return TrajectorySample(
            index=int(index),
            time_s=elapsed_s,
            segment=segment.name,
            q_requested=np.asarray(q, dtype=np.float64).copy(),
            segment_time_s=local_time,
            instantaneous_frequency_hz=frequency,
        )


def compile_trajectory(spec, initial_q, expected_hz=None):
    configured_hz = float(spec.get("control_hz", expected_hz or 200.0))
    if configured_hz <= 0.0:
        raise ValueError("trajectory control_hz must be positive")
    if expected_hz is not None and abs(configured_hz - float(expected_hz)) > 1.0e-9:
        raise ValueError(
            f"trajectory control_hz={configured_hz} does not match configured "
            f"controller rate {expected_hz} Hz"
        )
    definitions = spec.get("segments")
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("trajectory needs a non-empty segments list")

    current = np.asarray(initial_q, dtype=np.float64).copy()
    if current.shape != (JOINT_COUNT,) or not np.all(np.isfinite(current)):
        raise ValueError(f"initial_q must be finite with shape ({JOINT_COUNT},)")
    start_time = 0.0
    compiled = []
    supported = {"hold", "linear", "smoothstep", "minimum_jerk", "sine", "chirp"}
    for segment_index, definition in enumerate(definitions):
        if not isinstance(definition, dict):
            raise ValueError(f"segment {segment_index} must be a mapping")
        kind = str(definition.get("type", "hold")).strip().lower()
        name = str(definition.get("name", f"segment_{segment_index}_{kind}"))
        duration = float(definition.get("duration_s", 0.0))
        if kind not in supported:
            raise ValueError(f"{name}: unsupported segment type {kind!r}")
        if duration <= 0.0 or not math.isfinite(duration):
            raise ValueError(f"{name}: duration_s must be positive and finite")
        kwargs = {
            "name": name,
            "kind": kind,
            "start_time_s": start_time,
            "duration_s": duration,
            "start": current.copy(),
        }
        if kind in ("hold", "linear", "smoothstep", "minimum_jerk"):
            target = joint_vector(
                definition.get("target"), base=current, field=f"{name}.target"
            )
            kwargs["target"] = target
            current = target.copy()
        else:
            center = joint_vector(
                definition.get("center"), base=current, field=f"{name}.center"
            )
            amplitude = joint_vector(
                definition.get("amplitude"), field=f"{name}.amplitude"
            )
            f_start = float(
                definition.get("frequency_hz", definition.get("f_start_hz", 0.5))
            )
            f_end = float(definition.get("f_end_hz", f_start))
            law = str(definition.get("law", "linear")).strip().lower()
            _chirp_phase(0.0, duration, f_start, f_end, law)
            phase_offset = float(definition.get("phase_rad", 0.0))
            kwargs.update(
                center=center,
                amplitude=amplitude,
                phase_offset_rad=phase_offset,
                f_start_hz=f_start,
                f_end_hz=f_end,
                law=law,
            )
            end_phase = (
                2.0 * math.pi * f_start * duration
                if kind == "sine"
                else _chirp_phase(duration, duration, f_start, f_end, law)
            )
            current = center + amplitude * math.sin(end_phase + phase_offset)
        compiled.append(CompiledSegment(**kwargs))
        start_time += duration
    return TrajectoryPlan(configured_hz, compiled)


def build_trajectory(spec, initial_q, expected_hz=None):
    plan = compile_trajectory(spec, initial_q, expected_hz=expected_hz)
    count = int(round(plan.duration_s * plan.control_hz))
    return [
        plan.evaluate(index * plan.control_dt_s, index=index)
        for index in range(count)
    ]


def load_trajectory(path, initial_q, expected_hz=None):
    spec = load_yaml(path)
    return spec, build_trajectory(spec, initial_q, expected_hz=expected_hz)
