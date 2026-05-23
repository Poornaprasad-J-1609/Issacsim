import numpy as np
import torch

from grallator_interface import (
    POLICY_TO_REAL_ORDER,
    Q_DEFAULT,
    Q_STAND,
    policy_action_to_q_target,
    array_to_joint_dict,
)
from safety import safety_filter

POLICY_PT = "/workspace/isaaclab/logs/rsl_rl/grallator_flat/2026-05-20_06-11-05/exported/policy.pt"

policy = torch.jit.load(POLICY_PT, map_location="cpu")
policy.eval()

# Example standing observation:
# obs = [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
#        command(3), joint_pos_rel(12), joint_vel(12), previous_action(12)]
obs = np.zeros(48, dtype=np.float32)
obs[6:9] = np.array([0.0, 0.0, -1.0], dtype=np.float32)

# Start with very low real command. Do not start with 1.4 m/s on robot.
obs[9:12] = np.array([0.05, 0.0, 0.0], dtype=np.float32)

previous_action = np.zeros(12, dtype=np.float32)
previous_q_target = Q_STAND.copy()

print("Policy:", POLICY_PT)
print("Initial command [vx, vy, yaw] =", obs[9:12])
print()

for step in range(10):
    obs[36:48] = previous_action

    with torch.no_grad():
        action = policy(torch.tensor(obs).unsqueeze(0)).squeeze(0).cpu().numpy()

    q_policy = policy_action_to_q_target(action)
    q_safe = safety_filter(q_policy, previous_q_target)

    print("=" * 80)
    print(f"step {step}")
    print("action min/max/mean_abs:",
          float(action.min()), float(action.max()), float(np.abs(action).mean()))

    for i, name in enumerate(POLICY_TO_REAL_ORDER):
        print(
            f"{i:02d} {name:16s} "
            f"act={action[i]: .3f} "
            f"q_policy={q_policy[i]: .3f} "
            f"q_safe={q_safe[i]: .3f}"
        )

    # For offline loop only:
    # assume robot reached q_safe and velocity approximately zero
    obs[12:24] = q_safe - Q_DEFAULT
    obs[24:36] = 0.0

    previous_action = action.copy()
    previous_q_target = q_safe.copy()
