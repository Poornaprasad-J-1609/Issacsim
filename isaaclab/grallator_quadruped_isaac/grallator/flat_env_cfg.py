from isaaclab.utils import configclass
import math
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm

from .rough_env_cfg import GrallatorRoughEnvCfg


@configclass
class GrallatorFlatEnvCfg(GrallatorRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------------
        # Flat terrain only
        # ---------------------------------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # ---------------------------------------------------------
        # Disable randomization for first flat walking stage
        # ---------------------------------------------------------
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.add_base_mass = None

        # ---------------------------------------------------------
        # Easy forward command only
        # ---------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.25)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Small residual action first.
        # Too much action makes the rear legs fold/kick.
        self.actions.joint_pos.scale = 0.12

        # ---------------------------------------------------------
        # Main walking rewards
        # ---------------------------------------------------------
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.15

        # Strong trunk upright penalty.
        # This penalizes roll/pitch continuously.
        self.rewards.flat_orientation_l2.weight = -10.0

        # Penalize vertical bouncing and body angular wobble.
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.25

        # Keep this low first.
        # If this is too high, the robot lifts feet before learning balance.
        self.rewards.feet_air_time.weight = 0.05

        # Smoothness / motor penalties
        self.rewards.dof_torques_l2.weight = -1.0e-4
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.03

        # Stronger penalty if trunk/thigh/calf touches ground.
        self.rewards.undesired_contacts.weight = -8.0

        # ---------------------------------------------------------
        # Anti rear-leg folding posture penalties
        # ---------------------------------------------------------
        # This keeps calf joints near the default standing pose.
        # It directly fights knee/calf folding.
        self.rewards.calf_posture = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.35,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_calf_joint"],
                )
            },
        )

        # This keeps thigh joints from collapsing too much.
        self.rewards.thigh_posture = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.15,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_thigh_joint"],
                )
            },
        )

        # ---------------------------------------------------------
        # Terminations
        # ---------------------------------------------------------
        # Keep robot from learning knee-walking / belly sliding.
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "trunk",
            ".*_hip",
            ".*_thigh",
            ".*_calf",
        ]
        self.terminations.base_contact.params["threshold"] = 0.04

        # If trunk goes too low, terminate.
        self.terminations.base_height.params["minimum_height"] = 0.40

        # Trunk tilt termination.
        # Start with 12 degrees. Later reduce to 8 deg, then 4 deg.
        self.terminations.bad_orientation.params["limit_angle"] = math.radians(12.0)


@configclass
class GrallatorFlatEnvCfg_PLAY(GrallatorFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_mass = None
