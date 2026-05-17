from isaaclab.app import AppLauncher

# Brev viewer mode
app_launcher = AppLauncher(headless=True, livestream=1)
simulation_app = app_launcher.app

import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


USD_PATH = "/workspace/isaaclab/grallator_quadruped_isaac/usd/grallator_final.usd"


# Standing pose: same logic as your RL standing reset
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


# Left legs use positive thigh/calf direction.
# Right legs use negative thigh/calf direction.
LEG_SIGN = {
    "FL": +1.0,
    "BL": +1.0,
    "FR": -1.0,
    "BR": -1.0,
}


# Diagonal trot:
# Pair A = FL + BR
# Pair B = FR + BL
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
        joint_pos=STAND_POSE,
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
            stiffness=100.0,
            damping=10.0,
        ),
    },
)


def pose_dict_to_tensor(robot, pose_dict):
    q = robot.data.default_joint_pos.clone()
    for name, value in pose_dict.items():
        jid = robot.joint_names.index(name)
        q[:, jid] = value
    return q


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def make_trot_target(robot, t, base_q):
    q = base_q.clone()

    gait_freq = 1.0          # Hz, slow first
    thigh_amp = 0.20         # forward/back swing amplitude
    calf_lift = 0.45         # knee bend during swing
    stance_calf = 0.05       # small bend during stance

    for leg in ["FL", "FR", "BL", "BR"]:
        sign = LEG_SIGN[leg]

        phase = (t * gait_freq + LEG_PHASE_OFFSET[leg]) % 1.0

        thigh_name = f"{leg}_thigh_joint"
        calf_name = f"{leg}_calf_joint"

        thigh_id = robot.joint_names.index(thigh_name)
        calf_id = robot.joint_names.index(calf_name)

        if phase < 0.5:
            # STANCE PHASE
            # Foot stays near ground and leg sweeps backward.
            u = phase / 0.5
            sweep = (1.0 - 2.0 * u)   # +1 to -1
            thigh_mag = thigh_amp * sweep
            calf_mag = stance_calf

        else:
            # SWING PHASE
            # Leg lifts, moves forward, then prepares to touch down.
            u = (phase - 0.5) / 0.5
            u_smooth = smoothstep(u)

            sweep = -1.0 + 2.0 * u_smooth   # -1 to +1
            lift = math.sin(math.pi * u)    # 0 to 1 to 0

            thigh_mag = thigh_amp * sweep
            calf_mag = stance_calf + calf_lift * lift

        q[:, thigh_id] = sign * thigh_mag
        q[:, calf_id] = sign * calf_mag

    return q


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view([2.5, -2.5, 1.4], [0.0, 0.0, 0.4])

    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/GroundPlane", ground)

    light = sim_utils.DomeLightCfg(intensity=3000.0)
    light.func("/World/Light", light)

    robot = Articulation(GRALLATOR_CFG.replace(prim_path="/World/Grallator"))

    sim.reset()

    print("\n====================================")
    print("GRALLATOR PD TROT WALK TEST")
    print("====================================")
    print("USD:", USD_PATH)
    print("Joints:", robot.joint_names)
    print("Bodies:", robot.body_names)

    base_q = pose_dict_to_tensor(robot, STAND_POSE)
    zero_vel = torch.zeros_like(base_q)

    robot.write_joint_state_to_sim(base_q, zero_vel)
    robot.reset()

    dt = sim.get_physics_dt()

    stand_hold_time = 2.0
    walk_time = 20.0

    stand_steps = int(stand_hold_time / dt)
    walk_steps = int(walk_time / dt)

    print("\nPhase 1: hold stand")
    print("Phase 2: open-loop PD trot")
    print("Watch whether it moves forward, jumps, or only shakes.\n")

    for step in range(stand_steps + walk_steps):
        if step < stand_steps:
            target_q = base_q
            phase_name = "STAND"
            t_walk = 0.0
        else:
            phase_name = "TROT"
            t_walk = (step - stand_steps) * dt
            target_q = make_trot_target(robot, t_walk, base_q)

        robot.set_joint_position_target(target_q)
        robot.write_data_to_sim()

        sim.step()
        robot.update(dt)

        if step % 100 == 0:
            pos = robot.data.root_pos_w[0].detach().cpu().numpy()
            print(
                f"step={step:04d} "
                f"phase={phase_name:5s} "
                f"x={pos[0]:+.3f} "
                f"y={pos[1]:+.3f} "
                f"z={pos[2]:+.3f}"
            )

    print("\nFinished PD trot test. Keeping viewer open.")

    while simulation_app.is_running():
        t_walk += dt
        target_q = make_trot_target(robot, t_walk, base_q)
        robot.set_joint_position_target(target_q)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
