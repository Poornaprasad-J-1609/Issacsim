from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, livestream=1)
simulation_app = app_launcher.app

import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


USD_PATH = "/workspace/isaaclab/grallator_quadruped_isaac/usd/grallator_final.usd"


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


# Walking base pose: rear knees slightly bent so rear calf links stay off ground.
WALK_BASE_POSE = {
    "BL_hip_joint": 0.0,
    "BR_hip_joint": 0.0,
    "FL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,

    # Rear legs: less folded than before
    "BL_thigh_joint": 0.05,
    "BR_thigh_joint": -0.05,
    "FL_thigh_joint": 0.0,
    "FR_thigh_joint": 0.0,

    "BL_calf_joint": 0.25,
    "BR_calf_joint": -0.25,
    "FL_calf_joint": 0.0,
    "FR_calf_joint": 0.0,
}


LEG_SIGN = {
    "FL": +1.0,
    "BL": +1.0,
    "FR": -1.0,
    "BR": -1.0,
}


# Trot pair:
# FL + BR together, FR + BL together
LEG_PHASE_OFFSET = {
    "FL": 0.0,
    "BR": 0.0,
    "FR": 0.5,
    "BL": 0.5,
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
        pos=(0.0, 0.0, 0.30),
        joint_pos=CROUCH_POSE,
    ),

    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_joint",
                ".*_thigh_joint",
                ".*_calf_joint",
            ],
            effort_limit_sim=80.0,
            velocity_limit_sim=20.0,
            stiffness=120.0,
            damping=5.0,
        ),
    },
)


def smoothstep(alpha):
    alpha = max(0.0, min(1.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def pose_dict_to_tensor(robot, pose_dict):
    q = robot.data.default_joint_pos.clone()

    for joint_name, joint_value in pose_dict.items():
        if joint_name not in robot.joint_names:
            raise RuntimeError(f"Joint not found: {joint_name}")
        joint_id = robot.joint_names.index(joint_name)
        q[:, joint_id] = joint_value

    return q


def make_walk_target(robot, t, stand_q):
    q = stand_q.clone()

    # Slow crawl gait: one leg swings at a time.
    # This is safer than trot for debugging.
    gait_freq = 0.35
    swing_fraction = 0.22

    # One-leg-at-a-time order
    leg_phase = {
        "FL": 0.00,
        "BR": 0.25,
        "FR": 0.50,
        "BL": 0.75,
    }

    for leg in ["FL", "FR", "BL", "BR"]:
        sign = LEG_SIGN[leg]
        phase = (t * gait_freq + leg_phase[leg]) % 1.0

        thigh_id = robot.joint_names.index(f"{leg}_thigh_joint")
        calf_id = robot.joint_names.index(f"{leg}_calf_joint")

        # Diagnostic: freeze rear legs at WALK_BASE_POSE.
        # Do not overwrite rear joints with zero.
        if leg in ["BL", "BR"]:
            continue
        else:
            thigh_amp = 0.12
            calf_lift = 0.18

        if phase < swing_fraction:
            # Swing phase: lift foot gently and move leg forward.
            u = phase / swing_fraction
            u_smooth = smoothstep(u)

            sweep = -1.0 + 2.0 * u_smooth
            lift = math.sin(math.pi * u)

            thigh_mag = thigh_amp * sweep
            calf_mag = calf_lift * lift

        else:
            # Stance phase: keep foot near ground and slowly sweep backward.
            u = (phase - swing_fraction) / (1.0 - swing_fraction)
            u_smooth = smoothstep(u)

            sweep = 1.0 - 2.0 * u_smooth

            thigh_mag = thigh_amp * sweep
            calf_mag = 0.0

        q[:, thigh_id] = sign * thigh_mag
        q[:, calf_id] = sign * calf_mag

    return q


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=[2.6, -2.4, 1.4],
        target=[0.0, 0.0, 0.45],
    )

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0)
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(GRALLATOR_CFG.replace(prim_path="/World/Grallator"))

    sim.reset()

    print("\n======================================")
    print("GRALLATOR CROUCH → STAND → WALK TEST")
    print("======================================")
    print("USD path:", USD_PATH)
    print("Number of joints:", len(robot.joint_names))
    print("Number of bodies:", len(robot.body_names))

    crouch_q = pose_dict_to_tensor(robot, CROUCH_POSE)
    stand_q = pose_dict_to_tensor(robot, STAND_POSE)
    walk_base_q = pose_dict_to_tensor(robot, WALK_BASE_POSE)
    zero_vel = torch.zeros_like(crouch_q)

    robot.write_joint_state_to_sim(crouch_q, zero_vel)
    robot.reset()

    dt = sim.get_physics_dt()

    crouch_hold_time = 1.0
    standup_time = 3.0
    stand_hold_time = 2.0
    walk_time = 25.0

    crouch_hold_steps = int(crouch_hold_time / dt)
    standup_steps = int(standup_time / dt)
    stand_hold_steps = int(stand_hold_time / dt)
    walk_steps = int(walk_time / dt)

    total_steps = crouch_hold_steps + standup_steps + stand_hold_steps + walk_steps

    print("\nPhases:")
    print("1. CROUCH")
    print("2. STANDUP")
    print("3. STAND")
    print("4. WALK\n")

    for step in range(total_steps):
        if step < crouch_hold_steps:
            phase_name = "CROUCH"
            target_q = crouch_q

        elif step < crouch_hold_steps + standup_steps:
            phase_name = "STANDUP"
            k = step - crouch_hold_steps
            alpha = smoothstep(k / standup_steps)
            target_q = (1.0 - alpha) * crouch_q + alpha * stand_q

        elif step < crouch_hold_steps + standup_steps + stand_hold_steps:
            phase_name = "STAND"
            target_q = stand_q

        else:
            phase_name = "WALK"
            k = step - crouch_hold_steps - standup_steps - stand_hold_steps
            t_walk = k * dt
            target_q = make_walk_target(robot, t_walk, walk_base_q)

        robot.set_joint_position_target(target_q)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        if step % 100 == 0:
            pos = robot.data.root_pos_w[0].detach().cpu().numpy()
            print(
                f"step={step:04d} "
                f"phase={phase_name:7s} "
                f"x={pos[0]:+.3f} "
                f"y={pos[1]:+.3f} "
                f"z={pos[2]:+.3f}"
            )

    print("\nFinished. Keeping viewer open in walking mode.")

    t_walk = 0.0
    while simulation_app.is_running():
        t_walk += dt
        target_q = make_walk_target(robot, t_walk, walk_base_q)

        robot.set_joint_position_target(target_q)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
