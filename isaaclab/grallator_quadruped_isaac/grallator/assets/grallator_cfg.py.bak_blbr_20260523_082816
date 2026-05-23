import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


GRALLATOR_USD_PATH = "/workspace/isaaclab/grallator_quadruped_isaac/usd/grallator_final.usd"


GRALLATOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GRALLATOR_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos={
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
        },
    ),

    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_joint",
                ".*_thigh_joint",
                ".*_calf_joint",
            ],
            effort_limit_sim=60.0,
            velocity_limit_sim=2.0,
            stiffness=200.0,
            damping=10.0,
        ),
    },
)
