from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, livestream=1)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


USD_PATH = "/workspace/isaaclab/grallator_quadruped_isaac/usd/grallator_final.usd"

CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.2),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit_sim=120.0,
            velocity_limit_sim=20.0,
            stiffness=60.0,
            damping=3.0,
        ),
    },
)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view([2.5, -2.5, 1.5], [0.0, 0.0, 0.7])

    light = sim_utils.DomeLightCfg(intensity=3000.0)
    light.func("/World/Light", light)

    robot = Articulation(CFG.replace(prim_path="/World/Grallator"))
    sim.reset()

    print("\nJOINT ORDER:")
    for i, name in enumerate(robot.joint_names):
        print(i, name)

    joints_to_test = [
        "BL_calf_joint",
        "BR_calf_joint",
        "FL_calf_joint",
        "FR_calf_joint",
    ]

    base_q = robot.data.default_joint_pos.clone()
    zero_vel = torch.zeros_like(base_q)

    robot.write_joint_state_to_sim(base_q, zero_vel)
    robot.reset()

    dt = sim.get_physics_dt()

    for joint_name in joints_to_test:
        joint_id = robot.joint_names.index(joint_name)

        print("\n================================")
        print("MOVING:", joint_name)
        print("Look at viewer and note which leg moves.")
        print("================================")

        for step in range(600):
            q = base_q.clone()

            if step < 300:
                q[:, joint_id] = 0.8
            else:
                q[:, joint_id] = -0.8

            robot.set_joint_position_target(q)
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)

        # pause at zero
        for step in range(200):
            robot.set_joint_position_target(base_q)
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)

    print("\nFinished joint mapping test.")
    while simulation_app.is_running():
        robot.set_joint_position_target(base_q)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
