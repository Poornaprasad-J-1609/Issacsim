from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, livestream=2)
simulation_app = app_launcher.app

import time
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation

from grallator.grallator import GRALLATOR_CFG


def main():
    usd_path = "/workspace/isaaclab/grallator_quadruped_isaac/grallator/assets/grallator.usd"
    print("Loading USD:", usd_path)

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=[2.5, 2.0, 1.2],
        target=[0.0, 0.0, 0.25],
    )

    # Ground
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # Light
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0)
    light_cfg.func("/World/Light", light_cfg)

    # Use original Grallator USD
    robot_cfg = GRALLATOR_CFG.replace(prim_path="/World/Grallator")
    robot_cfg.spawn.usd_path = usd_path

    # Spawn slightly above ground
    robot_cfg.init_state.pos = (0.0, 0.0, 0.45)

    robot = Articulation(cfg=robot_cfg)

    sim.reset()
    robot.update(sim.get_physics_dt())

    # Hold initial pose using active PD target
    hold_joint_pos = robot.data.joint_pos.clone()

    print("Original Grallator USD loaded with ground, light, camera, and PD hold.")
    print("Open/refresh viewer now:")
    print("https://isaac0-u2xsmfb2z.brevlab.com/viewer/")

    step = 0
    while simulation_app.is_running():
        robot.set_joint_position_target(hold_joint_pos)
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim.get_physics_dt())

        if step % 300 == 0:
            print("step:", step, "root_pos:", robot.data.root_pos_w.cpu().numpy())

        step += 1
        time.sleep(0.005)


if __name__ == "__main__":
    main()
    simulation_app.close()
