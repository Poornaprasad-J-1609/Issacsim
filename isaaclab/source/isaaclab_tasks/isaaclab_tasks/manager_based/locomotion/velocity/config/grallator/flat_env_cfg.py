from isaaclab.utils import configclass

from .rough_env_cfg import GrallatorRoughEnvCfg


@configclass
class GrallatorFlatEnvCfg(GrallatorRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Flat walking first. No terrain generator and no height scan.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.contact_forces = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # Disable contact-dependent terms for first spawn test.
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.terminations.base_contact = None


        # Reward tuning for stable first walking.
        self.rewards.flat_orientation_l2.weight = -2.5
        # feet_air_time disabled for first spawn test



@configclass
class GrallatorFlatEnvCfg_PLAY(GrallatorFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
