#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc

from policy_runner import PolicyRunner
from safety_monitor import SafetyMonitor
from state_estimator import FakeStateEstimator
from joystick_interface import CommandSource
from robstride_can_interface import ATUsbCan
from motor_command_layer import MotorCommandLayer, print_mit_commands


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_motor_ids():
    cfg = load_yaml(ROOT / "config" / "motor_ids.yaml")
    return cfg["motor_ids"]


def smoothstep(alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def fake_start_pose_array(runner, name):
    if name == "stand":
        return runner.q_stand.copy()
    if name == "crouch":
        return runner.q_crouch.copy()
    if name == "random_small":
        rng = np.random.default_rng(7)
        return rng.uniform(-0.25, 0.25, size=12).astype(np.float32)
    raise ValueError(f"Unknown fake start pose: {name}")


def run_startup_to_stand(
    runner,
    safety,
    motor_layer,
    estimator,
    bus,
    mode,
    standup_seconds,
    log_every,
    show_hex,
):
    dt = runner.control_dt
    steps = max(1, int(standup_seconds / dt))

    q_start, _, _, _, _ = estimator.read()
    q_previous_target = q_start.copy()

    print("\n" + "#" * 80)
    print("STARTUP PHASE: current pose -> STAND / DEFAULT pose")
    print("#" * 80)
    print("standup_seconds:", standup_seconds)
    print("startup steps:", steps)
    print("mode:", mode)

    for step in range(steps):
        alpha = smoothstep((step + 1) / steps)
        q_desired = (1.0 - alpha) * q_start + alpha * runner.q_stand
        q_safe = safety.safety_filter(q_desired, q_previous_target)

        commands = motor_layer.build_mit_commands(q_safe, phase="startup")

        if mode == "signal":
            for cmd in commands:
                bus.send_signal_frame(cmd["motor_id"])
        elif mode == "mit-signal":
            motor_layer.send_signal_commands(bus, commands)

        if step % log_every == 0 or step == steps - 1:
            print("\n" + "=" * 80)
            print(f"startup step {step}/{steps - 1}, alpha={alpha:.3f}")
            print("MIT commands to STAND:")
            print_mit_commands(commands, show_hex=show_hex)

        estimator.dry_update_as_if_robot_followed(q_safe, dt)
        q_previous_target = q_safe.copy()
        time.sleep(dt)

    print("\nStartup phase completed. Robot target is STAND / DEFAULT pose.")
    return q_previous_target


def run_policy_loop(
    runner,
    safety,
    motor_layer,
    estimator,
    command_source,
    bus,
    mode,
    q_previous_target,
    steps,
    log_every,
    show_hex,
):
    dt = runner.control_dt
    previous_action = np.zeros(12, dtype=np.float32)

    control_mode = "policy"  # options: policy, stand, sit

    print("\n" + "#" * 80)
    print("POLICY / POSE PHASE")
    print("#" * 80)
    print("mode:", mode)
    print("Joystick buttons:")
    print("  stand button  -> STAND pose")
    print("  sit button    -> CROUCH/SIT pose")
    print("  policy button -> RL walking policy")

    for step in range(steps):
        (
            q_current,
            qd_current,
            base_lin_vel_b,
            base_ang_vel_b,
            projected_gravity_b,
        ) = estimator.read()

        stop, reason = safety.emergency_stop_check(
            projected_gravity_b=projected_gravity_b,
            base_ang_vel_b=base_ang_vel_b,
        )
        if stop:
            print("\nEMERGENCY STOP:", reason)
            break

        mode_request = command_source.get_mode_request()
        if mode_request is not None:
            control_mode = mode_request
            print(f"\n[MODE CHANGE] control_mode -> {control_mode}")

        command = command_source.read()

        if control_mode == "stand":
            q_policy_target = runner.q_stand.copy()
            q_safe_target = safety.safety_filter(q_policy_target, q_previous_target)
            commands = motor_layer.build_mit_commands(q_safe_target, phase="startup")
            action = np.zeros(12, dtype=np.float32)

        elif control_mode == "sit":
            q_policy_target = runner.q_crouch.copy()
            q_safe_target = safety.safety_filter(q_policy_target, q_previous_target)
            commands = motor_layer.build_mit_commands(q_safe_target, phase="startup")
            action = np.zeros(12, dtype=np.float32)

        elif control_mode == "policy":
            obs = runner.build_observation(
                base_lin_vel_b=base_lin_vel_b,
                base_ang_vel_b=base_ang_vel_b,
                projected_gravity_b=projected_gravity_b,
                command=command,
                q_current=q_current,
                qd_current=qd_current,
                previous_action=previous_action,
            )

            action = runner.infer_action(obs)
            q_policy_target = runner.action_to_q_target(action)
            q_safe_target = safety.safety_filter(q_policy_target, q_previous_target)
            commands = motor_layer.build_mit_commands(q_safe_target, phase="policy")

        else:
            raise RuntimeError(f"Unknown control_mode: {control_mode}")

        if mode == "signal":
            for cmd in commands:
                bus.send_signal_frame(cmd["motor_id"])
        elif mode == "mit-signal":
            motor_layer.send_signal_commands(bus, commands)

        if step % log_every == 0:
            print("\n" + "=" * 80)
            print(f"step {step}")
            print("control_mode:", control_mode)
            print(
                "command [vx, vy, yaw]:",
                f"{command[0]: .3f}",
                f"{command[1]: .3f}",
                f"{command[2]: .3f}",
            )
            print(
                "action min/max/mean_abs:",
                f"{action.min(): .3f}",
                f"{action.max(): .3f}",
                f"{np.abs(action).mean(): .3f}",
            )
            print("MIT commands:")
            print_mit_commands(commands, show_hex=show_hex)

        estimator.dry_update_as_if_robot_followed(q_safe_target, dt)

        if control_mode == "policy":
            previous_action = action.copy()
        else:
            previous_action = np.zeros(12, dtype=np.float32)

        q_previous_target = q_safe_target.copy()
        time.sleep(dt)

    print("\nPolicy / pose phase completed.")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["print", "signal", "mit-signal", "motors"],
        default="print",
        help="print=no serial, signal=harmless empty CAN frames, mit-signal=sends MIT packets, motors=blocked",
    )

    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=921600)

    parser.add_argument("--command-source", choices=["fixed", "joystick"], default="fixed")

    parser.add_argument("--vx", type=float, default=0.05)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)

    parser.add_argument("--max-vx", type=float, default=0.15)
    parser.add_argument("--max-vy", type=float, default=0.10)
    parser.add_argument("--max-yaw", type=float, default=0.30)

    parser.add_argument("--axis-vx", type=int, default=1)
    parser.add_argument("--axis-vy", type=int, default=0)
    parser.add_argument("--axis-yaw", type=int, default=3)

    parser.add_argument("--button-stand", type=int, default=0)
    parser.add_argument("--button-sit", type=int, default=1)
    parser.add_argument("--button-policy", type=int, default=2)

    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--expo", type=float, default=0.35)
    parser.add_argument("--smoothing", type=float, default=0.20)

    parser.add_argument("--policy-steps", type=int, default=200)
    parser.add_argument("--standup-seconds", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--show-hex", action="store_true")

    parser.add_argument(
        "--fake-start",
        choices=["stand", "crouch", "random_small"],
        default="crouch",
    )

    args = parser.parse_args()

    if args.mode == "motors":
        print("ERROR: --mode motors is intentionally blocked for now.")
        print("Reason: real encoder feedback parser is not connected yet.")
        print("Use --mode print, --mode signal, or --mode mit-signal first.")
        return

    runner = PolicyRunner()
    safety = SafetyMonitor(runner.policy_order)
    motor_ids = load_motor_ids()
    motor_layer = MotorCommandLayer(runner.policy_order, motor_ids)

    mit_cfg = load_yaml(ROOT / "config" / "mit_motor_control.yaml")
    standup_seconds = (
        float(args.standup_seconds)
        if args.standup_seconds is not None
        else float(mit_cfg["startup"]["standup_seconds"])
    )

    command_source = CommandSource(
        source=args.command_source,
        vx=args.vx,
        vy=args.vy,
        yaw=args.yaw,
        max_vx=args.max_vx,
        max_vy=args.max_vy,
        max_yaw=args.max_yaw,
        axis_vx=args.axis_vx,
        axis_vy=args.axis_vy,
        axis_yaw=args.axis_yaw,
        button_stand=args.button_stand,
        button_sit=args.button_sit,
        button_policy=args.button_policy,
        deadzone=args.deadzone,
        expo=args.expo,
        smoothing=args.smoothing,
    )

    q_fake_start = fake_start_pose_array(runner, args.fake_start)
    estimator = FakeStateEstimator(q_initial=q_fake_start)

    print("==== GRALLATOR JETSON MIT CONTROLLER ====")
    print("Mode:", args.mode)
    print("Command source:", args.command_source)
    print("Initial command:", command_source.read())
    print("Policy:", runner.policy_path)
    print("Control dt:", runner.control_dt)
    print("Port:", args.port)
    print("Baud:", args.baud)
    print("Fake start pose:", args.fake_start)
    print()

    print("Joint order and motor IDs:")
    for i, name in enumerate(runner.policy_order):
        print(f"{i:02d}: {name:16s} -> motor_id={motor_ids[name]}")

    bus = None

    if args.mode in ["signal", "mit-signal"]:
        if args.mode == "signal":
            print("\n--mode signal: sends harmless empty CAN frames only.")
            print("This tests USB-CAN transmission even if motors are absent.")
        elif args.mode == "mit-signal":
            print("\nWARNING: --mode mit-signal sends MIT control packets.")
            print("If motors are connected and powered, they may move.")
            print("For no-motor packet test, keep motors disconnected from CAN/power.")

        print("\nOpening USB-CAN serial port...")
        bus = ATUsbCan(port=args.port, baud=args.baud).open()
        print("USB-CAN opened.")

    try:
        q_previous_target = run_startup_to_stand(
            runner=runner,
            safety=safety,
            motor_layer=motor_layer,
            estimator=estimator,
            bus=bus,
            mode=args.mode,
            standup_seconds=standup_seconds,
            log_every=args.log_every,
            show_hex=args.show_hex,
        )

        run_policy_loop(
            runner=runner,
            safety=safety,
            motor_layer=motor_layer,
            estimator=estimator,
            command_source=command_source,
            bus=bus,
            mode=args.mode,
            q_previous_target=q_previous_target,
            steps=args.policy_steps,
            log_every=args.log_every,
            show_hex=args.show_hex,
        )

    finally:
        if bus is not None:
            bus.close()
            print("\nUSB-CAN closed.")

    print("\nController finished.")


if __name__ == "__main__":
    main()
