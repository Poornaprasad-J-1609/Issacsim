from motor_mapping import JOINT_TO_MOTOR_ID
from grallator_interface import POLICY_TO_REAL_ORDER

def send_motor_targets_dry(q_target):
    """
    Dry-run motor sender.
    This does NOT send CAN commands.
    It only converts q_target array into motor_id commands.
    """
    commands = []

    for i, joint_name in enumerate(POLICY_TO_REAL_ORDER):
        motor_id = JOINT_TO_MOTOR_ID[joint_name]
        q_des = float(q_target[i])

        commands.append({
            "joint_name": joint_name,
            "motor_id": motor_id,
            "q_des": q_des,
        })

    return commands


if __name__ == "__main__":
    import numpy as np

    q_test = np.array([
        0.1, -0.1, -0.15, 0.2,
        0.05, -0.05, 0.03, -0.04,
        -0.08, 0.09, -0.07, 0.1,
    ])

    commands = send_motor_targets_dry(q_test)

    print("DRY MOTOR COMMANDS")
    print("=" * 70)
    for cmd in commands:
        print(
            f"motor_id={cmd['motor_id']:02d} "
            f"{cmd['joint_name']:16s} "
            f"q_des={cmd['q_des']: .4f} rad"
        )
