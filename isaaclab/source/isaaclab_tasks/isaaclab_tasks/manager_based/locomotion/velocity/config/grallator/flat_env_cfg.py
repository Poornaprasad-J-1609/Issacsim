from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

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

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # ---------------------------------------------------------
        # STANDING ONLY command
        # ---------------------------------------------------------
        # No walking now. First teach the robot to stand tall.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 1.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)

        # Small action scale = less shaking while learning stand.
        self.actions.joint_pos.scale = 0.25

        # ---------------------------------------------------------
        # Standing reward tuning
        # ---------------------------------------------------------

        # Keep body upright.
        self.rewards.flat_orientation_l2.weight = -11.0

        # Since command velocity is zero, this rewards staying still.
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 0.4

        # No stepping during standing training.
        self.rewards.feet_air_time.weight = 0.25

        # Penalize vertical bouncing, roll/pitch angular velocity, jerky actions.
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.dof_acc_l2.weight = -1.0e-6
        self.rewards.action_rate_l2.weight = -0.030

        # ---------------------------------------------------------
        # Standing posture penalties
        # ---------------------------------------------------------
        # IMPORTANT:
        # These rewards pull joints toward GRALLATOR_DEFAULT_JOINT_POS.
        # So your grallator.py default joint pose MUST be a standing pose.

        self.rewards.rear_thigh_posture = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.10,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "BL_thigh_joint",
                        "BR_thigh_joint",
                    ],
                )
            },
        )

        self.rewards.rear_calf_posture = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.20,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "BL_calf_joint",
                        "BR_calf_joint",
                    ],
                )
            },
        )

        self.rewards.front_thigh_posture = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.10,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FL_thigh_joint",
                        "FR_thigh_joint",
                    ],
                )
            },
        )

        self.rewards.front_calf_posture = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.20,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FL_calf_joint",
                        "FR_calf_joint",
                    ],
                )
            },
        )

        self.rewards.rear_hip_posture = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.08,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "BL_hip_joint",
                        "BR_hip_joint",
                    ],
                )
            },
        )

        self.rewards.front_hip_posture = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.04,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "FL_hip_joint",
                        "FR_hip_joint",
                    ],
                )
            },
        )

        # ---------------------------------------------------------
        # Disable all individual stepping rewards
        # ---------------------------------------------------------
        # No airtime reward now. We only want standing.
        foot_air_func = self.rewards.feet_air_time.func
        base_foot_air_params = dict(self.rewards.feet_air_time.params)

        fl_params = dict(base_foot_air_params)
        fl_params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names="FL_foot")

        fr_params = dict(base_foot_air_params)
        fr_params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names="FR_foot")

        bl_params = dict(base_foot_air_params)
        bl_params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names="BL_foot")

        br_params = dict(base_foot_air_params)
        br_params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names="BR_foot")

        self.rewards.fl_step_bonus = RewTerm(func=foot_air_func, weight=0.008, params=fl_params)
        self.rewards.fr_step_bonus = RewTerm(func=foot_air_func, weight=0.008, params=fr_params)
        self.rewards.bl_step_bonus = RewTerm(func=foot_air_func, weight=0.02, params=bl_params)
        self.rewards.br_step_bonus = RewTerm(func=foot_air_func, weight=0.02, params=br_params)

        # ---------------------------------------------------------
        # Base height reward: do not allow sitting
        # ---------------------------------------------------------
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight= 4.8,
            params={
                "target_height": 0.50,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ---------------------------------------------------------
        # Contact penalty
        # ---------------------------------------------------------
        if getattr(self.rewards, "undesired_contacts", None) is not None:
            self.rewards.undesired_contacts.weight = -5.5
            self.rewards.undesired_contacts.params["threshold"] = 0.03
            self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
                "trunk",
                ".*_hip",
                ".*_thigh",
                ".*_calf",
            ]

        # ---------------------------------------------------------
        # Terminations
        # ---------------------------------------------------------
        # Trunk/hip contact means failed standing.
        if getattr(self.terminations, "base_contact", None) is not None:
            self.terminations.base_contact.params["sensor_cfg"].body_names = [
                "trunk",
                ".*_hip",
            ]
            self.terminations.base_contact.params["threshold"] = 0.08

        # Sitting too low means failed standing.
        if getattr(self.terminations, "base_height", None) is None:
            self.terminations.base_height = DoneTerm(
                func=mdp.root_height_below_minimum,
                params={
                    "minimum_height": 0.44,
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            )
        else:
            self.terminations.base_height.params["minimum_height"] = 0.44

        # Kill large roll/pitch collapse.
        if getattr(self.terminations, "bad_orientation", None) is None:
            self.terminations.bad_orientation = DoneTerm(
                func=mdp.bad_orientation,
                params={
                    "limit_angle": 0.90,
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            )
        else:
            self.terminations.bad_orientation.params["limit_angle"] = 0.90


@configclass
class GrallatorFlatEnvCfg_PLAY(GrallatorFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None