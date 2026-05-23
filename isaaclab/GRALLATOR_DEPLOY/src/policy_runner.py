#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import torch

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class PolicyRunner:
    def __init__(self, policy_path=None):
        self.root = ROOT

        self.joint_cfg = load_yaml(self.root / "config" / "joint_map.yaml")
        self.pose_cfg = load_yaml(self.root / "config" / "default_pose.yaml")

        self.policy_order = self.joint_cfg["policy_to_real_order"]
        self.action_scale = float(self.joint_cfg["policy_action_scale"])
        self.control_dt = float(self.joint_cfg["control_dt"])

        self.q_default = self.pose_to_array(self.pose_cfg["default_pose"])
        self.q_stand = self.pose_to_array(self.pose_cfg["stand_pose"])
        self.q_crouch = self.pose_to_array(self.pose_cfg["crouch_pose"])

        if policy_path is None:
            policy_path = self.root / "policy" / "policy.pt"

        self.policy_path = Path(policy_path)
        self.policy = torch.jit.load(str(self.policy_path), map_location="cpu")
        self.policy.eval()

    def pose_to_array(self, pose_dict):
        return np.array([pose_dict[name] for name in self.policy_order], dtype=np.float32)

    def build_observation(
        self,
        base_lin_vel_b,
        base_ang_vel_b,
        projected_gravity_b,
        command,
        q_current,
        qd_current,
        previous_action,
    ):
        obs = np.zeros(48, dtype=np.float32)

        obs[0:3] = np.asarray(base_lin_vel_b, dtype=np.float32)
        obs[3:6] = np.asarray(base_ang_vel_b, dtype=np.float32)
        obs[6:9] = np.asarray(projected_gravity_b, dtype=np.float32)
        obs[9:12] = np.asarray(command, dtype=np.float32)
        obs[12:24] = np.asarray(q_current, dtype=np.float32) - self.q_default
        obs[24:36] = np.asarray(qd_current, dtype=np.float32)
        obs[36:48] = np.asarray(previous_action, dtype=np.float32)

        return obs

    def infer_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.policy(obs_t).squeeze(0).cpu().numpy()
        return action.astype(np.float32)

    def action_to_q_target(self, action):
        action = np.asarray(action, dtype=np.float32)
        return self.q_default + self.action_scale * action

    def array_to_joint_dict(self, q):
        q = np.asarray(q, dtype=np.float32)
        return {name: float(q[i]) for i, name in enumerate(self.policy_order)}


if __name__ == "__main__":
    runner = PolicyRunner()
    print("Loaded policy:", runner.policy_path)
    print("Control dt:", runner.control_dt)
    print("Action scale:", runner.action_scale)
    print("Joint order:")
    for i, name in enumerate(runner.policy_order):
        print(f"{i:02d}: {name}")
    print("Q_DEFAULT:", runner.q_default)
    print("Q_STAND:", runner.q_stand)
    print("Q_CROUCH:", runner.q_crouch)
