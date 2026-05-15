from isaaclab.app import AppLauncher

# GUI mode
app_launcher = AppLauncher(headless=True, livestream=1)
simulation_app = app_launcher.app

import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


USD_PATH = "/workspace/isaaclab/grallator_quadruped_isaac/usd/grallator_final.usd"


# ------------------------------------------------------------
# IMPORTANT NAME MAPPING
# Your notation:
#   RL = rear-left
#   RR = rear-right
#
# Isaac Lab USD detected:
#   BL = rear-left / RL
#   BR = rear-right / RR
# ------------------------------------------------------------


CROUCH_POSE = {
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
}


STAND_POSE = {
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
        # For your almost-straight standing pose, start higher.
        # If robot drops too hard, try 1.20.
        # If it floats too much, try 0.95.
        pos=(0.0, 0.0, 0.3),
        joint_pos=CROUCH_POSE,
    ),

    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_joint",
                ".*_thigh_joint",
                ".*_calf_joint",
            ],
            effort_limit_sim=60,
            velocity_limit_sim=2.0,
            stiffness=200.0,
            damping=10.0,
        ),
    },
)


def quat_to_rpy_deg(q):
    """Convert quaternion [w, x, y, z] to roll, pitch, yaw in degrees."""
    w, x, y, z = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def smoothstep(alpha):
    alpha = max(0.0, min(1.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def pose_dict_to_tensor(robot, pose_dict):
    joint_pos = robot.data.default_joint_pos.clone()

    for joint_name, joint_value in pose_dict.items():
        if joint_name not in robot.joint_names:
            raise RuntimeError(
                f"Joint name not found: {joint_name}\n"
                f"Available joints: {robot.joint_names}"
            )

        joint_id = robot.joint_names.index(joint_name)
        joint_pos[:, joint_id] = joint_value

    return joint_pos


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=[2.4, -2.4, 1.5],
        target=[0.0, 0.0, 0.55],
    )

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0)
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(GRALLATOR_CFG.replace(prim_path="/World/Grallator"))

    sim.reset()

    print("\n======================================")
    print("GRALLATOR CROUCH TO STAND PD TEST")
    print("======================================")
    print("USD path:", USD_PATH)
    print("Number of joints:", len(robot.joint_names))
    print("Number of bodies:", len(robot.body_names))

    print("\nJoint order detected by Isaac Lab:")
    for i, name in enumerate(robot.joint_names):
        print(i, name)

    crouch_q = pose_dict_to_tensor(robot, CROUCH_POSE)
    stand_q = pose_dict_to_tensor(robot, STAND_POSE)
    zero_vel = torch.zeros_like(crouch_q)

    # Force robot into crouch before starting simulation
    robot.write_joint_state_to_sim(crouch_q, zero_vel)
    robot.reset()

    dt = sim.get_physics_dt()

    crouch_hold_time = 1.0
    standup_time = 3.0
    stand_hold_time = 5.0

    crouch_hold_steps = int(crouch_hold_time / dt)
    standup_steps = int(standup_time / dt)
    stand_hold_steps = int(stand_hold_time / dt)

    total_steps = crouch_hold_steps + standup_steps + stand_hold_steps

    print("\nRunning phases:")
    print("1. Hold crouch")
    print("2. Smooth crouch to stand")
    print("3. Hold stand\n")

    for step in range(total_steps):
        if step < crouch_hold_steps:
            phase = "CROUCH"
            target_q = crouch_q

        elif step < crouch_hold_steps + standup_steps:
            phase = "STANDUP"
            k = step - crouch_hold_steps
            alpha = smoothstep(k / standup_steps)
            target_q = (1.0 - alpha) * crouch_q + alpha * stand_q

        else:
            phase = "STAND"
            target_q = stand_q

        robot.set_joint_position_target(target_q)
        robot.write_data_to_sim()

        sim.step()
        robot.update(dt)

        if step % 100 == 0:
            base_pos = robot.data.root_pos_w[0].cpu().numpy()
            base_quat = robot.data.root_quat_w[0].cpu().numpy()
            roll, pitch, yaw = quat_to_rpy_deg(base_quat)

            print(
                f"step={step:04d} "
                f"phase={phase:7s} "
                f"base_x={base_pos[0]:+.3f} "
                f"base_y={base_pos[1]:+.3f} "
                f"base_z={base_pos[2]:+.3f} "
                f"roll={roll:+.2f} "
                f"pitch={pitch:+.2f} "
                f"yaw={yaw:+.2f}"
            )

            if phase == "STAND":
                for foot_name in ["BL_foot", "BR_foot", "FL_foot", "FR_foot"]:
                    foot_id = robot.body_names.index(foot_name)
                    foot_pos = robot.data.body_pos_w[0, foot_id].cpu().numpy()
                    print(f"    {foot_name}: z={foot_pos[2]:+.4f}")

    print("\nFinished crouch-to-stand test.")
    print("Robot will keep holding STAND_POSE. Close viewer manually.")

    while simulation_app.is_running():
        robot.set_joint_position_target(stand_q)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
