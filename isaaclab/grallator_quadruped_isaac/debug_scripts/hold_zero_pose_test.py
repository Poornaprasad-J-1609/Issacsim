from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, livestream=2)
simulation_app = app_launcher.app

import time
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation

from grallator.grallator import GRALLATOR_CFG, GRALLATOR_DEFAULT_JOINT_POS


def main():
    print("GRALLATOR_DEFAULT_JOINT_POS:")
    for k, v in GRALLATOR_DEFAULT_JOINT_POS.items():
        print(f"  {k}: {v}")

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=[2.5, 2.0, 1.0],
        target=[0.0, 0.0, 0.25],
    )

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0)
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = GRALLATOR_CFG.replace(prim_path="/World/Grallator")
    robot_cfg.init_state.pos = (0.0, 0.0, 0.45)

    robot = Articulation(cfg=robot_cfg)

    sim.reset()
    robot.update(sim.get_physics_dt())

    print("\nJOINT NAMES FROM ISAAC LAB:")
    print(robot.data.joint_names)

    print("\nDEFAULT JOINT POS FROM ISAAC LAB:")
    print(robot.data.default_joint_pos.cpu().numpy())

    hold_pos = robot.data.default_joint_pos.clone()

    print("\nHolding this pose with active joint targets.")
    print("Open/refresh viewer:")
    print("https://isaac0-u2xsmfb2z.brevlab.com/viewer/")

    step = 0
    while simulation_app.is_running():
        robot.set_joint_position_target(hold_pos)
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim.get_physics_dt())

        if step % 300 == 0:
            print("step:", step, "root_z:", robot.data.root_pos_w[:, 2].cpu().numpy())

        step += 1
        time.sleep(0.005)


if __name__ == "__main__":
    main()
    simulation_app.close()
