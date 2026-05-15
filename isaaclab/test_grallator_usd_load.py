from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


USD_PATH = "/workspace/isaaclab/grallator_quadruped_isaac/usd/grallator_final.usd"


GRALLATOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
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
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.60,
            "FR_calf_joint": -0.95,

            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.60,
            "FL_calf_joint": 0.95,

            "BR_hip_joint": 0.0,
            "BR_thigh_joint": 0.50,
            "BR_calf_joint": -1.05,

            "BL_hip_joint": 0.0,
            "BL_thigh_joint": 0.50,
            "BL_calf_joint": 1.05,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit_sim=80.0,
            velocity_limit_sim=20.0,
            stiffness=40.0,
            damping=2.0,
        ),
    },
)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, -2.0, 1.2], [0.0, 0.0, 0.3])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0)
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(GRALLATOR_CFG.replace(prim_path="/World/Grallator"))

    sim.reset()

    print("\n======================================")
    print("GRALLATOR USD LOAD TEST")
    print("======================================")
    print("USD path:", USD_PATH)
    print("Number of joints:", len(robot.joint_names))
    print("Number of bodies:", len(robot.body_names))

    print("\nJoint names:")
    for i, name in enumerate(robot.joint_names):
        print(i, name)

    print("\nBody names:")
    for i, name in enumerate(robot.body_names):
        print(i, name)

    print("\nRunning standing pose test...")

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    for i in range(2000):
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

        if i % 200 == 0:
            base_pos = robot.data.root_pos_w[0].cpu().numpy()
            base_quat = robot.data.root_quat_w[0].cpu().numpy()
            print(f"step={i:04d} base_pos={base_pos} base_quat={base_quat}")

    print("\nFinished test. Close viewer manually.")
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
