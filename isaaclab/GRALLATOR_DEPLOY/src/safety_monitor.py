#!/usr/bin/env python3
from pathlib import Path
import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class SafetyMonitor:
    def __init__(self, policy_order):
        self.root = ROOT
        self.policy_order = policy_order
        self.limit_path = self.root / "config" / "joint_limits.yaml"
        self.control_limit_path = self.root / "config" / "control_limits.yaml"
        self.limit_mtime_ns = None
        self.control_limit_mtime_ns = None

        self.q_min = None
        self.q_max = None
        self.dq_max = None
        self.joint_position_enabled = True
        self.joint_rate_enabled = True
        self.reload_joint_limits(force=True)
        self.reload_control_limits(force=True)

        self.cfg = load_yaml(self.root / "config" / "safety_limits.yaml")
        emergency = self.cfg["emergency"]
        self.projected_gravity_gz_min = float(emergency["projected_gravity_gz_min"])
        self.max_body_ang_vel_norm = float(emergency["max_body_ang_vel_norm"])

    def _dict_to_array(self, d):
        return np.array([d[name] for name in self.policy_order], dtype=np.float32)

    def reload_joint_limits(self, force=False):
        mtime_ns = self.limit_path.stat().st_mtime_ns
        if not force and mtime_ns == self.limit_mtime_ns:
            return False

        cfg = load_yaml(self.limit_path)
        limits = cfg["joint_limits"]

        q_min = []
        q_max = []
        dq_max = []

        for joint_name in self.policy_order:
            if joint_name not in limits:
                raise KeyError(f"Missing joint limit for {joint_name} in {self.limit_path}")

            joint_limit = limits[joint_name]
            q_lo = float(joint_limit["min"])
            q_hi = float(joint_limit["max"])
            dq_step = float(joint_limit["dq_max_per_step"])

            if q_lo > q_hi:
                raise ValueError(f"{joint_name}: min {q_lo} is greater than max {q_hi}")
            if dq_step < 0.0:
                raise ValueError(f"{joint_name}: dq_max_per_step must be >= 0")

            q_min.append(q_lo)
            q_max.append(q_hi)
            dq_max.append(dq_step)

        self.q_min = np.asarray(q_min, dtype=np.float32)
        self.q_max = np.asarray(q_max, dtype=np.float32)
        self.dq_max = np.asarray(dq_max, dtype=np.float32)
        self.limit_mtime_ns = mtime_ns
        return True

    def reload_control_limits(self, force=False):
        mtime_ns = self.control_limit_path.stat().st_mtime_ns
        if not force and mtime_ns == self.control_limit_mtime_ns:
            return False

        cfg = load_yaml(self.control_limit_path)
        self.joint_position_enabled = bool(
            cfg.get("joint_position", {}).get("enabled", True)
        )
        self.joint_rate_enabled = bool(
            cfg.get("joint_rate", {}).get("enabled", True)
        )
        self.control_limit_mtime_ns = mtime_ns
        return True

    def clip_q_target(self, q_target, q_reference=None):
        q_target = np.asarray(q_target, dtype=np.float32)
        if not self.joint_position_enabled:
            return q_target

        q_clipped = np.clip(q_target, self.q_min, self.q_max)

        if q_reference is not None:
            q_reference = np.asarray(q_reference, dtype=np.float32)

            below = q_reference < self.q_min
            recovering_from_below = below & (q_target < self.q_min) & (q_target >= q_reference)
            q_clipped[recovering_from_below] = q_target[recovering_from_below]

            above = q_reference > self.q_max
            recovering_from_above = above & (q_target > self.q_max) & (q_target <= q_reference)
            q_clipped[recovering_from_above] = q_target[recovering_from_above]

        return q_clipped

    def rate_limit_q_target(self, q_desired, q_previous):
        q_desired = np.asarray(q_desired, dtype=np.float32)
        q_previous = np.asarray(q_previous, dtype=np.float32)
        if not self.joint_rate_enabled:
            return q_desired

        dq = q_desired - q_previous
        dq = np.clip(dq, -self.dq_max, self.dq_max)
        return q_previous + dq

    def safety_filter(self, q_policy_target, q_previous_target):
        self.reload_control_limits()
        self.reload_joint_limits()

        q_previous_target = np.asarray(q_previous_target, dtype=np.float32)

        q = self.clip_q_target(q_policy_target)
        q = self.rate_limit_q_target(q, q_previous_target)
        q = self.clip_q_target(q, q_reference=q_previous_target)
        return q.astype(np.float32)

    def emergency_stop_check(self, projected_gravity_b, base_ang_vel_b):
        projected_gravity_b = np.asarray(projected_gravity_b, dtype=np.float32)
        base_ang_vel_b = np.asarray(base_ang_vel_b, dtype=np.float32)

        if projected_gravity_b[2] > self.projected_gravity_gz_min:
            return True, f"bad tilt: projected_gravity={projected_gravity_b}"

        if np.linalg.norm(base_ang_vel_b) > self.max_body_ang_vel_norm:
            return True, f"high body angular velocity: {base_ang_vel_b}"

        return False, ""


if __name__ == "__main__":
    from policy_runner import PolicyRunner

    runner = PolicyRunner()
    safety = SafetyMonitor(runner.policy_order)

    print("Safety limits:")
    for i, name in enumerate(runner.policy_order):
        print(
            f"{i:02d} {name:16s} "
            f"min={safety.q_min[i]: .3f} "
            f"max={safety.q_max[i]: .3f} "
            f"dq_step={safety.dq_max[i]: .3f}"
        )
