import numpy as np

# ============================================================
# DO NOT CHANGE: user's real robot convention
# ============================================================

GRALLATOR_DEFAULT_JOINT_POS = {
    "BL_hip_joint": 0.0,
    "BR_hip_joint": 0.0,
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,

    "BL_thigh_joint": 0.0,
    "BR_thigh_joint": 0.0,
    "FL_thigh_joint": 0.0,
    "FR_thigh_joint": 0.0,

    "BL_calf_joint": 0.0,
    "BR_calf_joint": 0.0,
    "FL_calf_joint": 0.0,
    "FR_calf_joint": 0.0,
}

CROUCH_POSE = {
    "BL_hip_joint": 0.0,
    "BR_hip_joint": 0.0,
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,

    "BL_thigh_joint": 0.30,
    "BR_thigh_joint": -0.30,
    "FL_thigh_joint": 0.30,
    "FR_thigh_joint": -0.30,

    "BL_calf_joint": 1.00,
    "BR_calf_joint": -1.00,
    "FL_calf_joint": 1.00,
    "FR_calf_joint": -1.00,
}

STAND_POSE = {
    "BL_hip_joint": 0.0,
    "BR_hip_joint": 0.0,
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,

    "BL_thigh_joint": 0.0,
    "BR_thigh_joint": 0.0,
    "FL_thigh_joint": 0.0,
    "FR_thigh_joint": 0.0,

    "BL_calf_joint": 0.0,
    "BR_calf_joint": 0.0,
    "FL_calf_joint": 0.0,
    "FR_calf_joint": 0.0,
}

# ============================================================
# Policy order from Isaac.
# RL/RR in Isaac are mapped to BL/BR on the real robot.
# ============================================================

POLICY_TO_REAL_ORDER = [
    "FL_hip_joint",
    "FR_hip_joint",
    "BL_hip_joint",
    "BR_hip_joint",

    "FL_thigh_joint",
    "FR_thigh_joint",
    "BL_thigh_joint",
    "BR_thigh_joint",

    "FL_calf_joint",
    "FR_calf_joint",
    "BL_calf_joint",
    "BR_calf_joint",
]

ACTION_SCALE = 0.12

def pose_dict_to_policy_array(pose_dict):
    return np.array([pose_dict[name] for name in POLICY_TO_REAL_ORDER], dtype=np.float32)

Q_DEFAULT = pose_dict_to_policy_array(GRALLATOR_DEFAULT_JOINT_POS)
Q_STAND = pose_dict_to_policy_array(STAND_POSE)
Q_CROUCH = pose_dict_to_policy_array(CROUCH_POSE)

def policy_action_to_q_target(action):
    """
    Full policy action.
    No artificial action clipping.
    q_target = Q_DEFAULT + 0.12 * action
    """
    action = np.asarray(action, dtype=np.float32)
    return Q_DEFAULT + ACTION_SCALE * action

def array_to_joint_dict(q):
    q = np.asarray(q, dtype=np.float32)
    return {name: float(q[i]) for i, name in enumerate(POLICY_TO_REAL_ORDER)}

if __name__ == "__main__":
    print("POLICY_TO_REAL_ORDER:")
    for i, name in enumerate(POLICY_TO_REAL_ORDER):
        print(f"{i:02d}: {name}")

    print("\nQ_DEFAULT:")
    print(Q_DEFAULT)

    print("\nQ_STAND:")
    print(Q_STAND)

    print("\nQ_CROUCH:")
    print(Q_CROUCH)
