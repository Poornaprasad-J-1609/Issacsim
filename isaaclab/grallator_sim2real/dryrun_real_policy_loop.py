#!/usr/bin/env python3
import time
import argparse
import numpy as np
import torch

from grallator_interface import (
    POLICY_TO_REAL_ORDER,
    Q_DEFAULT,
    Q_STAND,
    policy_action_to_q_target,
)
from safety import safety_filter

POLICY_PT = "/workspace/isaaclab/logs/rsl_rl/grallator_flat/2026-05-20_06-11-05/exported/policy.pt"

CONTROL_DT = 0.02  # 50 Hz

def build_observation(
    base_lin_vel_b,
    base_ang_vel_b,
    projected_gravity_b,
    command,
    q_current,
    qd_current,
    previous_action,
):
    obs = np.zeros(48, dtype=np.float32)

    obs[0:3] = base_lin_vel_b
    obs[3:6] = base_ang_vel_b
    obs[6:9] = projected_gravity_b
    obs[9:12] = command
    obs[12:24] = q_current - Q_DEFAULT
    obs[24:36] = qd_current
    obs[36:48] = previous_action

    return obs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vx", type=float, default=0.05)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    policy = torch.jit.load(POLICY_PT, map_location="cpu")
    policy.eval()

    q_current = Q_STAND.copy()
    qd_current = np.zeros(12, dtype=np.float32)
    q_previous_target = Q_STAND.copy()
    previous_action = np.zeros(12, dtype=np.float32)

    # For dry-run only. On real robot:
    # base_ang_vel_b and projected_gravity_b come from IMU.
    # base_lin_vel_b comes from estimator or starts as zero.
    base_lin_vel_b = np.zeros(3, dtype=np.float32)
    base_ang_vel_b = np.zeros(3, dtype=np.float32)
    projected_gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)

    print("==== DRY RUN REAL POLICY LOOP ====")
    print("Policy:", POLICY_PT)
    print("Command [vx, vy, yaw]:", command)
    print("Control dt:", CONTROL_DT)
    print()
    print("Joint order:")
    for i, name in enumerate(POLICY_TO_REAL_ORDER):
        print(f"{i:02d}: {name}")

    for step in range(args.steps):
        obs = build_observation(
            base_lin_vel_b=base_lin_vel_b,
            base_ang_vel_b=base_ang_vel_b,
            projected_gravity_b=projected_gravity_b,
            command=command,
            q_current=q_current,
            qd_current=qd_current,
            previous_action=previous_action,
        )

        with torch.no_grad():
            action = policy(torch.tensor(obs).unsqueeze(0)).squeeze(0).cpu().numpy()

        q_policy_target = policy_action_to_q_target(action)
        q_safe_target = safety_filter(q_policy_target, q_previous_target)

        if step % 20 == 0:
            print("\n" + "=" * 80)
            print(f"step {step}")
            print(
                "action min/max/mean_abs:",
                f"{action.min(): .3f}",
                f"{action.max(): .3f}",
                f"{np.abs(action).mean(): .3f}",
            )
            for i, name in enumerate(POLICY_TO_REAL_ORDER):
                print(
                    f"{i:02d} {name:16s} "
                    f"q_now={q_current[i]: .3f} "
                    f"q_policy={q_policy_target[i]: .3f} "
                    f"q_safe={q_safe_target[i]: .3f}"
                )

        # Dry-run simulation assumption:
        # pretend the robot follows the safe target.
        qd_current = (q_safe_target - q_current) / CONTROL_DT
        q_current = q_safe_target.copy()

        previous_action = action.copy()
        q_previous_target = q_safe_target.copy()

        time.sleep(CONTROL_DT)

    print("\nDry run completed. No motor commands were sent.")

if __name__ == "__main__":
    main()
