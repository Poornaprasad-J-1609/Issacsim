from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


USD_PATH = "/workspace/isaaclab/grallator_quadruped_isaac/usd/grallator.usd"


# Joint order from your Isaac Lab output:
# 0  FL_hip_joint
# 1  FR_hip_joint
# 2  RL_hip_joint
# 3  RR_hip_joint
# 4  FL_thigh_joint
# 5  FR_thigh_joint
# 6  RL_thigh_joint
# 7  RR_thigh_joint
# 8  FL_calf_joint
# 9  FR_calf_joint
# 10 RL_calf_joint
# 11 RR_calf_joint

STAND_POSE = {
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "RR_hip_joint": 0.0,

    "FL_thigh_joint": 0.55,
    "FR_thigh_joint": 0.55,
    "RL_thigh_joint": 0.55,
    "RR_thigh_joint": 0.55,

    "FL_calf_joint": -0.95,
    "FR_calf_joint": -0.95,
    "RL_calf_joint": -0.95,
    "RR_calf_joint": -0.95,
}


GRALLATOR_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=UsdFileCfg(
        usd_path=USD_PATH,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos=STAND_POSE,
    ),
    actuators={
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_joint"],
            stiffness={
                "FL_hip_joint": 798.75653,
                "FR_hip_joint": 803.97223,
                "RL_hip_joint": 1127.08679,
                "RR_hip_joint": 1120.34473,
            },
            damping={
                "FL_hip_joint": 25.0,
                "FR_hip_joint": 25.0,
                "RL_hip_joint": 30.0,
                "RR_hip_joint": 30.0,
            },
            effort_limit=120.0,
            velocity_limit=30.0,
        ),
        "thighs": ImplicitActuatorCfg(
            joint_names_expr=[".*thigh_joint"],
            stiffness={
                "FL_thigh_joint": 283.56577,
                "FR_thigh_joint": 285.50311,
                "RL_thigh_joint": 366.37448,
                "RR_thigh_joint": 363.53192,
            },
            damping={
                "FL_thigh_joint": 12.0,
                "FR_thigh_joint": 12.0,
                "RL_thigh_joint": 15.0,
                "RR_thigh_joint": 15.0,
            },
            effort_limit=120.0,
            velocity_limit=30.0,
        ),
        "calves": ImplicitActuatorCfg(
            joint_names_expr=[".*calf_joint"],
            stiffness={
                "FL_calf_joint": 262.29556,
                "FR_calf_joint": 261.23358,
                "RL_calf_joint": 275.05780,
                "RR_calf_joint": 278.95438,
            },
            damping={
                "FL_calf_joint": 10.0,
                "FR_calf_joint": 10.0,
                "RL_calf_joint": 10.0,
                "RR_calf_joint": 10.0,
            },
            effort_limit=120.0,
            velocity_limit=30.0,
        ),
    },
)


@configclass
class GrallatorSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    robot: ArticulationCfg = GRALLATOR_CFG


def main():
    sim_cfg = SimulationCfg(dt=0.005)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=[2.5, 2.5, 1.4],
        target=[0.0, 0.0, 0.35],
    )

    scene_cfg = GrallatorSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    robot = scene["robot"]

    print("")
    print("======================================")
    print("GRALLATOR PD STAND TEST")
    print("======================================")
    print("Joint names:", robot.joint_names)
    print("Body names:", robot.body_names)

    joint_target = robot.data.default_joint_pos.clone()

    for joint_name, angle in STAND_POSE.items():
        joint_id = robot.joint_names.index(joint_name)
        joint_target[:, joint_id] = angle

    step = 0

    while simulation_app.is_running():
        robot.set_joint_position_target(joint_target)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

        step += 1

        if step % 200 == 0:
            root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
            root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
            joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()

            print("")
            print("step:", step)
            print("base xyz:", root_pos)
            print("base quat:", root_quat)
            print("joint pos:", joint_pos)

    simulation_app.close()


if __name__ == "__main__":
    main()
