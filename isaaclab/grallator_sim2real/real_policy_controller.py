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
from motor_sender import send_motor_targets_dry

POLICY_PT = "/workspace/isaaclab/logs/rsl_rl/grallator_flat/2026-05-20_06-11-05/exported/policy.pt"

CONTROL_DT = 0.02  # 50 Hz


# ============================================================
# OBSERVATION BUILDER
# ============================================================

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


# ============================================================
# REAL ROBOT SENSOR READER PLACEHOLDER
# Replace this later with encoder + IMU reading.
# ============================================================

def read_robot_state_fake(q_previous):
    """
    Fake state for dry testing.
    Later replace with:
      - q_current from motor encoders
      - qd_current from motor velocity feedback
      - base_ang_vel_b from IMU gyro
      - projected_gravity_b from IMU orientation
      - base_lin_vel_b from estimator
    """

    q_current = q_previous.copy()
    qd_current = np.zeros(12, dtype=np.float32)

    base_lin_vel_b = np.zeros(3, dtype=np.float32)
    base_ang_vel_b = np.zeros(3, dtype=np.float32)
    projected_gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    return q_current, qd_current, base_lin_vel_b, base_ang_vel_b, projected_gravity_b


# ============================================================
# MOTOR SEND PLACEHOLDER
# Disabled unless --enable-motors is passed.
# ============================================================

def send_motor_targets(q_target, enable_motors=False):
    """
    For now:
      - enable_motors=False: return dry motor command list
      - enable_motors=True : blocked until RobStride CAN layer is added
    """
    commands = send_motor_targets_dry(q_target)

    if not enable_motors:
        return commands

    raise RuntimeError(
        "Motor sending is still blocked. "
        "RobStride CAN layer has not been connected yet."
    )


# ============================================================
# SAFETY CHECKS
# ============================================================

def emergency_stop_check(projected_gravity_b, base_ang_vel_b):
    # Basic tilt check using projected gravity.
    # Upright is [0, 0, -1].
    gx, gy, gz = projected_gravity_b

    # If body tilts too much, gz moves away from -1.
    # This is conservative.
    if gz > -0.75:
        return True, f"bad tilt: projected_gravity={projected_gravity_b}"

    # Angular velocity sanity limit.
    if np.linalg.norm(base_ang_vel_b) > 8.0:
        return True, f"high angular velocity: {base_ang_vel_b}"

    return False, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vx", type=float, default=0.05)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--enable-motors", action="store_true")
    args = parser.parse_args()

    if args.enable_motors:
        print("ERROR: --enable-motors was requested, but motor send is not implemented yet.")
        print("Exiting safely.")
        return

    policy = torch.jit.load(POLICY_PT, map_location="cpu")
    policy.eval()

    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)

    previous_action = np.zeros(12, dtype=np.float32)
    q_previous_target = Q_STAND.copy()

    print("==== GRALLATOR REAL POLICY CONTROLLER ====")
    print("MODE: DRY RUN ONLY. Motors disabled.")
    print("Policy:", POLICY_PT)
    print("Command [vx, vy, yaw]:", command)
    print("Control dt:", CONTROL_DT)
    print()
    print("Joint order:")
    for i, name in enumerate(POLICY_TO_REAL_ORDER):
        print(f"{i:02d}: {name}")

    for step in range(args.steps):
        q_current, qd_current, base_lin_vel_b, base_ang_vel_b, projected_gravity_b = read_robot_state_fake(
            q_previous_target
        )

        stop, reason = emergency_stop_check(projected_gravity_b, base_ang_vel_b)
        if stop:
            print("\nEMERGENCY STOP:", reason)
            break

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

        motor_commands = send_motor_targets(q_safe_target, enable_motors=args.enable_motors)

        if step % 25 == 0:
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

            print("\nDRY MOTOR COMMANDS:")
            for cmd in motor_commands:
                print(
                    f"motor_id={cmd['motor_id']:02d} "
                    f"{cmd['joint_name']:16s} "
                    f"q_des={cmd['q_des']: .4f} rad"
                )

        previous_action = action.copy()
        q_previous_target = q_safe_target.copy()

        time.sleep(CONTROL_DT)

    print("\nController dry run completed.")


if __name__ == "__main__":
    main()
