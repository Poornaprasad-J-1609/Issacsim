from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


USD_PATH = "/workspace/isaaclab/grallator_quadruped_isaac/usd/grallator.usd"


GRALLATOR_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=UsdFileCfg(
        usd_path=USD_PATH,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={".*": 300.0},
            damping={".*": 20.0},
            effort_limit=120.0,
            velocity_limit=30.0,
        ),
    },
)


@configclass
class GrallatorSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = GRALLATOR_CFG


def main():
    sim_cfg = SimulationCfg(dt=0.005)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=[2.5, 2.5, 1.5],
        target=[0.0, 0.0, 0.3],
    )

    scene_cfg = GrallatorSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    robot = scene["robot"]

    print("")
    print("======================================")
    print("GRALLATOR USD LOAD TEST")
    print("======================================")
    print("USD path:", USD_PATH)
    print("Number of joints:", robot.num_joints)
    print("Number of bodies:", robot.num_bodies)

    print("")
    print("Joint names:")
    for i, name in enumerate(robot.joint_names):
        print(i, name)

    print("")
    print("Body names:")
    for i, name in enumerate(robot.body_names):
        print(i, name)

    count = 0
    while simulation_app.is_running():
        sim.step()
        scene.update(sim.get_physics_dt())

        count += 1
        if count % 200 == 0:
            root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
            print("base position:", root_pos)

    simulation_app.close()


if __name__ == "__main__":
    main()
