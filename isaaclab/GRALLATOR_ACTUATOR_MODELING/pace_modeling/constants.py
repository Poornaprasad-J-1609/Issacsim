"""Fixed PACE dataset contract."""

JOINT_ORDER = [
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

JOINT_COUNT = len(JOINT_ORDER)

# Isaac/PACE order is intentionally independent of the hardware controller's
# established internal order. Every exported array is permuted explicitly.
PACE_EXPORT_ORDER = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "BR_hip_joint",
    "BR_thigh_joint",
    "BR_calf_joint",
    "BL_hip_joint",
    "BL_thigh_joint",
    "BL_calf_joint",
]

PACE_EXPORT_INDICES = tuple(JOINT_ORDER.index(name) for name in PACE_EXPORT_ORDER)


def to_pace_order(values):
    """Return values with a final internal-joint axis in explicit PACE order."""
    return values[..., PACE_EXPORT_INDICES]
