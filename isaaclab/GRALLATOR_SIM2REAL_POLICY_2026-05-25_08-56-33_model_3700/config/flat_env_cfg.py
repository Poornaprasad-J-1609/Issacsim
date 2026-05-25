from isaaclab.utils import configclass

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.noise import GaussianNoiseCfg

from .rough_env_cfg import GrallatorRoughEnvCfg



def zero_base_lin_vel(env, asset_cfg=None):
    """Return zero base linear velocity with same [num_envs, 3] shape.

    This keeps old checkpoint observation size compatible,
    but prevents the policy from seeing true simulator base linear velocity.
    """
    return torch.zeros(env.num_envs, 3, device=env.device)


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

        # ---------------------------------------------------------
        # Remove true base linear velocity from policy observation
        # ---------------------------------------------------------
        # Important for sim-to-real:
        # real robot cannot measure perfect simulator base_lin_vel.
        #
        # We do NOT set base_lin_vel = None because we want to resume
        # model_3025.pt without changing observation dimension.
        # Instead, we keep the same 3 observation slots and feed zeros.
        if getattr(self.observations.policy, "base_lin_vel", None) is not None:
            self.observations.policy.base_lin_vel.func = zero_base_lin_vel
            self.observations.policy.base_lin_vel.params = {}
        self.curriculum.terrain_levels = None

        # ---------------------------------------------------------
        # Sensor noise for sim-to-real robustness
        # ---------------------------------------------------------
        # Joint encoder noise:
        #   joint_pos -> encoder angle noise
        #   joint_vel -> velocity-estimation noise
        #
        # IMU noise:
        #   projected_gravity -> orientation / gravity-vector noise
        #   base_ang_vel      -> gyro noise
        #
        # Important:
        # Observation dimension does NOT change.
        # This is safe for resuming model_3025.pt.
        self.observations.policy.enable_corruption = True

        for obs_name, noise_std in {
            "joint_pos": 0.008,             # rad
            "joint_vel": 0.05,              # rad/s
            "projected_gravity": 0.015,     # unit gravity vector noise
            "base_ang_vel": 0.02,           # rad/s gyro noise
        }.items():
            obs_term = getattr(self.observations.policy, obs_name, None)
            if obs_term is not None:
                obs_term.noise = GaussianNoiseCfg(mean=0.0, std=noise_std, operation="add")

        # ---------------------------------------------------------
        # Action latency hook
        # ---------------------------------------------------------
        # True action latency in Isaac Lab should be implemented at the
        # actuator level using DelayedPDActuatorCfg.
        #
        # This block activates latency if your robot actuator cfg already
        # supports min_delay / max_delay. If your actuator is still
        # ImplicitActuatorCfg, this block safely does nothing.
        #
        # Recommended first range:
        #   0 to 2 policy/control steps of delay
        if hasattr(self.scene.robot, "actuators"):
            for actuator_cfg in self.scene.robot.actuators.values():
                if hasattr(actuator_cfg, "min_delay") and hasattr(actuator_cfg, "max_delay"):
                    actuator_cfg.min_delay = 0
                    actuator_cfg.max_delay = 2

        # ---------------------------------------------------------
        # STANDING ONLY command
        # ---------------------------------------------------------
        # No walking now. First teach the robot to stand tall.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.2, 1.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.2, 1.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # Small action scale = less shaking while learning stand.
        self.actions.joint_pos.scale = 0.25

        # ---------------------------------------------------------
        # Domain randomization for sim-to-real robustness
        # ---------------------------------------------------------
        # These are intentionally kept mild because this config is still
        # trying to learn stable standing/walking without collapse.

        # Randomize FOOT friction only.
        # Do not randomize every body material; that is heavier and unnecessary.
        # This directly targets sim-to-real floor/contact mismatch.
        self.events.robot_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "FL_foot",
                        "FR_foot",
                        "BL_foot",
                        "BR_foot",
                    ],
                ),
                "static_friction_range": (0.60, 1.40),
                "dynamic_friction_range": (0.50, 1.20),
                "restitution_range": (0.0, 0.04),
                "num_buckets": 64,
                "make_consistent": True,
            },
        )

        # Randomize trunk mass slightly.
        # Grallator mass is around 39.4 kg, so +/-1.5 kg is a safe start.
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "mass_distribution_params": (-1.5, 1.5),
                "operation": "add",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )

        # Randomize trunk center of mass slightly.
        # Keep this very small first; large COM shifts can make early training collapse.
        self.events.randomize_base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "com_range": {
                    "x": (-0.02, 0.02),
                    "y": (-0.015, 0.015),
                    "z": (-0.01, 0.01),
                },
            },
        )

        # Randomize the initial joint state at every reset.
        # This prevents the policy from overfitting to one exact spawn posture.
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "position_range": (-0.04, 0.04),
                "velocity_range": (-0.10, 0.10),
            },
        )

        # Randomize actuator stiffness/damping to handle RobStride04 mismatch.
        # Keep this mild at first; increase later only after stable walking.
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.80, 1.20),
                "damping_distribution_params": (0.80, 1.20),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # Small random pushes during training.
        # This improves recovery, but it is deliberately weak for early training.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {
                    "x": (-0.15, 0.15),
                    "y": (-0.10, 0.10),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-0.10, 0.10),
                },
            },
        )

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

        # Keep play/evaluation deterministic.
        # Enable these again only if you want to visualize robustness testing.
        self.events.robot_physics_material = None
        self.events.add_base_mass = None
        self.events.randomize_base_com = None
        self.events.reset_robot_joints = None
        self.events.randomize_actuator_gains = None