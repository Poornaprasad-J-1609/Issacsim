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
        self.cfg = load_yaml(self.root / "config" / "safety_limits.yaml")

        self.q_min = self._dict_to_array(self.cfg["q_min"])
        self.q_max = self._dict_to_array(self.cfg["q_max"])
        self.dq_max = self._dict_to_array(self.cfg["dq_max_per_step"])

        emergency = self.cfg["emergency"]
        self.projected_gravity_gz_min = float(emergency["projected_gravity_gz_min"])
        self.max_body_ang_vel_norm = float(emergency["max_body_ang_vel_norm"])

    def _dict_to_array(self, d):
        return np.array([d[name] for name in self.policy_order], dtype=np.float32)

    def clip_q_target(self, q_target):
        q_target = np.asarray(q_target, dtype=np.float32)
        return np.clip(q_target, self.q_min, self.q_max)

    def rate_limit_q_target(self, q_desired, q_previous):
        q_desired = np.asarray(q_desired, dtype=np.float32)
        q_previous = np.asarray(q_previous, dtype=np.float32)

        dq = q_desired - q_previous
        dq = np.clip(dq, -self.dq_max, self.dq_max)
        return q_previous + dq

    def safety_filter(self, q_policy_target, q_previous_target):
        q = self.clip_q_target(q_policy_target)
        q = self.rate_limit_q_target(q, q_previous_target)
        q = self.clip_q_target(q)
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
