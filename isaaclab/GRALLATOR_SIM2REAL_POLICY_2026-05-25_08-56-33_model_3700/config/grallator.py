import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg


GRALLATOR_USD_PATH = os.environ.get(
    "GRALLATOR_USD_PATH",
    "/workspace/isaaclab/grallator_quadruped_isaac/usd/grallator_final.usd",
)


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


GRALLATOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GRALLATOR_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=2,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos=GRALLATOR_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),

    soft_joint_pos_limit_factor=0.9,
   actuators={
        # ---------------------------------------------------------
        # High-Power Industrial Config with Saturation Constraints (13.0 kg Robot)
        # ---------------------------------------------------------
        "front_hips": DCMotorCfg(
            joint_names_expr=["FL_hip_joint", "FR_hip_joint"],
            stiffness=80.0,            # High rigidity to maintain lateral structural integrity
            damping=5.0,               # Mathematically paired damping to avoid joint tremor
            effort_limit=60.0,         # Continuous maximum torque (Nm)
            saturation_effort=85.0,    # Peak burst torque to handle quick changes in direction
            velocity_limit=25.0,       # Speed ceiling (rad/s)
        ),

        "rear_hips": DCMotorCfg(
            joint_names_expr=["BL_hip_joint", "BR_hip_joint"],
            stiffness=80.0,            
            damping=5.0,               
            effort_limit=75.0,         
            saturation_effort=100.0,   # Stronger peak tolerance for rear abduction stability
            velocity_limit=25.0,       
        ),

        "front_thighs": DCMotorCfg(
            joint_names_expr=["FL_thigh_joint", "FR_thigh_joint"],
            stiffness=110.0,           # High tracking rigidity to force the torso upward
            damping=6.5,               
            effort_limit=90.0,         
            saturation_effort=130.0,   # Massive peak ceiling to break static sitting inertia
            velocity_limit=22.0,       
        ),

       "rear_thighs": DCMotorCfg(
            joint_names_expr=["BL_thigh_joint", "BR_thigh_joint"],
            stiffness=130.0,           # Maximum stiffness concentrated in rear driving joints
            damping=8.0,               
            effort_limit=120.0,        
            saturation_effort=160.0,   # High saturation allows dynamic push-off during gaits
            velocity_limit=22.0,       
        ),

        "front_calves": DCMotorCfg(
            joint_names_expr=["FL_calf_joint", "FR_calf_joint"],
            stiffness=110.0,           
            damping=6.5,               
            effort_limit=95.0,         
            saturation_effort=140.0,   # High peak capacity protects against hard ground impacts
            velocity_limit=22.0,       
        ),
   
        "rear_calves": DCMotorCfg(
            joint_names_expr=["BL_calf_joint", "BR_calf_joint"],
            stiffness=130.0,           
            damping=8.0,               
            effort_limit=130.0,        
            saturation_effort=170.0,   # Highest peak saturation to prevent buckling on impact
            velocity_limit=22.0,       
        ),
    },
)
