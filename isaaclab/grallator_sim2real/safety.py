import numpy as np

from grallator_interface import POLICY_TO_REAL_ORDER

# ============================================================
# Software limits for first real robot tests.
# These do NOT change DEFAULT, STAND, or CROUCH.
# They only protect final commanded q_target.
# ============================================================

Q_MIN_REAL = np.array([
    -0.55, -0.35, -0.55, -0.35,   # hip: FL, FR, BL, BR
    -0.75, -0.75, -0.75, -0.75,   # thigh
    -1.20, -1.20, -1.20, -1.20,   # calf
], dtype=np.float32)

Q_MAX_REAL = np.array([
     0.35,  0.35,  0.35,  0.35,   # hip
     0.75,  0.75,  0.75,  0.75,   # thigh
     1.20,  1.20,  1.20,  1.20,   # calf
], dtype=np.float32)

# Max target change per control step.
# At 50 Hz, 0.04 rad/step = 2.0 rad/s target slew rate.
DQ_MAX_PER_STEP = np.array([
    0.035, 0.035, 0.035, 0.035,   # hip
    0.045, 0.045, 0.045, 0.045,   # thigh
    0.060, 0.060, 0.060, 0.060,   # calf
], dtype=np.float32)

def clip_q_target(q_target):
    q_target = np.asarray(q_target, dtype=np.float32)
    return np.clip(q_target, Q_MIN_REAL, Q_MAX_REAL)

def rate_limit_q_target(q_desired, q_previous):
    q_desired = np.asarray(q_desired, dtype=np.float32)
    q_previous = np.asarray(q_previous, dtype=np.float32)

    dq = q_desired - q_previous
    dq = np.clip(dq, -DQ_MAX_PER_STEP, DQ_MAX_PER_STEP)
    return q_previous + dq

def safety_filter(q_policy_target, q_previous_target):
    q = clip_q_target(q_policy_target)
    q = rate_limit_q_target(q, q_previous_target)
    q = clip_q_target(q)
    return q

if __name__ == "__main__":
    print("REAL SOFTWARE LIMITS")
    for i, name in enumerate(POLICY_TO_REAL_ORDER):
        print(
            f"{i:02d} {name:16s} "
            f"min={Q_MIN_REAL[i]: .3f} "
            f"max={Q_MAX_REAL[i]: .3f} "
            f"dq_step={DQ_MAX_PER_STEP[i]: .3f}"
        )
