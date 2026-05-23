from grallator_interface import POLICY_TO_REAL_ORDER

# ============================================================
# Physical RobStride motor IDs
# User-provided convention has rear legs as RL/RR.
# Our real robot policy interface uses BL/BR.
# Mapping:
#   BL = RL
#   BR = RR
# ============================================================

JOINT_TO_MOTOR_ID = {
    # Front right leg
    "FR_hip_joint": 1,
    "FR_thigh_joint": 2,
    "FR_calf_joint": 3,

    # Front left leg
    "FL_hip_joint": 4,
    "FL_thigh_joint": 5,
    "FL_calf_joint": 6,

    # Back/right rear leg
    "BR_hip_joint": 7,      # RR_hip_joint
    "BR_thigh_joint": 8,    # RR_thigh_joint
    "BR_calf_joint": 9,     # RR_calf_joint

    # Back/left rear leg
    "BL_hip_joint": 10,     # RL_hip_joint
    "BL_thigh_joint": 11,   # RL_thigh_joint
    "BL_calf_joint": 12,    # RL_calf_joint
}

def validate_motor_mapping():
    missing = []
    duplicate_ids = {}

    seen = {}
    for joint_name in POLICY_TO_REAL_ORDER:
        if joint_name not in JOINT_TO_MOTOR_ID:
            missing.append(joint_name)
        else:
            motor_id = JOINT_TO_MOTOR_ID[joint_name]
            if motor_id in seen:
                duplicate_ids[motor_id] = (seen[motor_id], joint_name)
            seen[motor_id] = joint_name

    if missing:
        raise RuntimeError(f"Missing motor IDs for joints: {missing}")

    if duplicate_ids:
        raise RuntimeError(f"Duplicate motor IDs found: {duplicate_ids}")

    ids = [JOINT_TO_MOTOR_ID[name] for name in POLICY_TO_REAL_ORDER]
    if sorted(ids) != list(range(1, 13)):
        raise RuntimeError(f"Motor IDs should be 1..12, got: {ids}")

    return True

if __name__ == "__main__":
    validate_motor_mapping()

    print("POLICY OUTPUT ORDER -> REAL JOINT -> MOTOR ID")
    print("=" * 60)
    for i, joint_name in enumerate(POLICY_TO_REAL_ORDER):
        motor_id = JOINT_TO_MOTOR_ID[joint_name]
        print(f"{i:02d} {joint_name:16s} -> motor_id={motor_id}")

    print("\nMapping validation passed.")
