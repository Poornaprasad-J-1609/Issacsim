from isaaclab.utils import configclass

from .rough_env_cfg import GrallatorRoughEnvCfg


@configclass
class GrallatorFlatEnvCfg(GrallatorRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------
        # Flat terrain setup
        # ---------------------------------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # No height scanner for flat training
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # No terrain curriculum on flat ground
        self.curriculum.terrain_levels = None

        # ---------------------------------------------------------
        # Reward tuning for flat walking
        # ---------------------------------------------------------

        # Stronger body-upright penalty.
        # Your old value was -1.0, which is too weak.
        self.rewards.flat_orientation_l2.weight = -5.0

        # Encourage real stepping.
        # Your old value was 0.25.
        # Anymal uses 0.5, but Grallator can start with 0.35.
        self.rewards.feet_air_time.weight = 0.5

        # Penalize excessive motor torque.
        # This helps avoid violent leg kicking.
        self.rewards.dof_torques_l2.weight = -2.5e-5


@configclass
class GrallatorFlatEnvCfg_PLAY(GrallatorFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization during play/testing
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None