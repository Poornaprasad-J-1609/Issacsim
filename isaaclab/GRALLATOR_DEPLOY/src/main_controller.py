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
from state_estimator import FakeStateEstimator, MitFeedbackStateEstimator
from joystick_interface import CommandSource, load_joystick_defaults, load_speed_scale_defaults
from imu_interface import create_imu_sensor, load_imu_config
from robstride_can_interface import ATUsbCan
from motor_command_layer import MotorCommandLayer, print_mit_commands


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_motor_ids():
    cfg = load_yaml(ROOT / "config" / "motor_ids.yaml")
    return cfg["motor_ids"]


def load_active_joints():
    cfg = load_yaml(ROOT / "config" / "motor_ids.yaml")
    active_joints = cfg.get("active_joints", [])
    return list(active_joints or [])


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


def refresh_estimator_feedback(estimator, timeout=0.0):
    if hasattr(estimator, "refresh_from_bus"):
        return estimator.refresh_from_bus(timeout=timeout)
    return 0


def joystick_walk_requested(command, threshold):
    command = np.asarray(command, dtype=np.float32)
    return bool(np.max(np.abs(command)) > float(threshold))


def command_speed_scale(command_source):
    if hasattr(command_source, "get_speed_scale"):
        return float(command_source.get_speed_scale())
    return 1.0


def command_torque_stats(commands):
    if not commands:
        return 0.0, 0.0

    tau = np.asarray([float(cmd["tau_ff"]) for cmd in commands], dtype=np.float32)
    return float(np.abs(tau).mean()), float(np.abs(tau).max())


def feedback_torque_stats(estimator):
    feedback_by_joint = getattr(estimator, "last_feedback_by_joint", {})
    if not feedback_by_joint:
        return None

    tau = np.asarray(
        [float(feedback["torque"]) for feedback in feedback_by_joint.values()],
        dtype=np.float32,
    )
    return float(np.abs(tau).mean()), float(np.abs(tau).max())


def estimator_imu_status(estimator):
    if hasattr(estimator, "imu_status"):
        return estimator.imu_status()
    return "none"


def print_compact_telemetry(step, mode, command, command_source, commands, estimator):
    command = np.asarray(command, dtype=np.float32)
    vxy = float(np.linalg.norm(command[:2]))
    tau_cmd_mean, tau_cmd_max = command_torque_stats(commands)
    tau_fb = feedback_torque_stats(estimator)

    line = (
        f"step={step:06d} "
        f"mode={mode:6s} "
        f"vx={command[0]: .3f} "
        f"vy={command[1]: .3f} "
        f"vxy={vxy: .3f} "
        f"yaw={command[2]: .3f} "
        f"speed={command_speed_scale(command_source):.2f} "
        f"imu={estimator_imu_status(estimator)} "
        f"tau_cmd={tau_cmd_mean: .3f} "
        f"tau_cmd_max={tau_cmd_max: .3f} "
        f"cmds={len(commands):02d}"
    )

    if tau_fb is None:
        line += " tau_fb=na tau_fb_max=na"
    else:
        tau_fb_mean, tau_fb_max = tau_fb
        line += f" tau_fb={tau_fb_mean: .3f} tau_fb_max={tau_fb_max: .3f}"

    print(line)


def request_feedback_snapshot(motor_layer, bus, mode):
    if mode == "mit-signal" and bus is not None:
        motor_layer.send_raw_commands(bus, motor_layer.build_feedback_poll_commands())
        time.sleep(0.002)
        motor_layer.send_raw_commands(bus, motor_layer.build_enable_commands())


def imu_source_name(source, imu_defaults):
    if source == "auto":
        source = imu_defaults.get("source", "fake")
    return str(source).replace("-", "_").lower()


def initialize_hold_target(estimator, feedback_timeout):
    refresh_estimator_feedback(estimator, timeout=feedback_timeout)
    q_current, _, _, _, _ = estimator.read()

    print("\n" + "#" * 80)
    print("STARTUP PHASE: HOLD current pose")
    print("#" * 80)
    print("No sit, stand, or walking command is sent until joystick input.")
    return q_current.copy()


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
    feedback_timeout,
):
    dt = runner.control_dt
    steps = max(1, int(standup_seconds / dt))

    refresh_estimator_feedback(estimator, timeout=feedback_timeout)
    q_start, _, _, _, _ = estimator.read()
    q_previous_target = q_start.copy()

    print("\n" + "#" * 80)
    print("STARTUP PHASE: current pose -> STAND / DEFAULT pose")
    print("#" * 80)
    print("standup_seconds:", standup_seconds)
    print("startup steps:", steps)
    print("mode:", mode)

    for step in range(steps):
        if hasattr(estimator, "imu_stale") and estimator.imu_stale():
            print("\nEMERGENCY STOP: IMU data missing or stale during startup")
            break

        alpha = smoothstep((step + 1) / steps)
        q_desired = (1.0 - alpha) * q_start + alpha * runner.q_stand
        q_safe = safety.safety_filter(q_desired, q_previous_target)

        commands = motor_layer.build_mit_commands(q_safe, phase="startup")

        if mode == "signal":
            for cmd in commands:
                bus.send_signal_frame(cmd["motor_id"])
        elif mode == "mit-signal":
            motor_layer.send_signal_commands(bus, commands)

        refresh_estimator_feedback(estimator, timeout=feedback_timeout)

        if step % log_every == 0 or step == steps - 1:
            tau_cmd_mean, tau_cmd_max = command_torque_stats(commands)
            print(
                f"startup_step={step:06d}/{steps - 1:06d} "
                f"mode=stand "
                f"alpha={alpha:.3f} "
                f"vx={0.0: .3f} vy={0.0: .3f} "
                f"vxy={0.0: .3f} yaw={0.0: .3f} "
                f"tau_cmd={tau_cmd_mean: .3f} "
                f"tau_cmd_max={tau_cmd_max: .3f} "
                f"cmds={len(commands):02d}"
            )
            if show_hex:
                print_mit_commands(commands, show_hex=True)

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
    start_control_mode,
    feedback_timeout,
    walk_command_threshold,
):
    dt = runner.control_dt
    previous_action = np.zeros(12, dtype=np.float32)

    control_mode = start_control_mode  # options: hold, policy, stand, sit
    has_motion_target = control_mode in ("stand", "sit")

    print("\n" + "#" * 80)
    print("POLICY / POSE PHASE")
    print("#" * 80)
    print("mode:", mode)
    print("Joystick buttons:")
    print("  button 4    -> STOP walking and SIT/CROUCH pose")
    print("  button 5    -> STAND pose")
    print("  button 6    -> reduce speed scale")
    print("  button 7    -> increase speed scale")
    print("  buttons 0-3 -> EMERGENCY STOP")
    print("Joystick axes:")
    print("  left stick Y  -> forward/back vx")
    print("  left stick X  -> left/right vy")
    print("  right stick X -> turn/yaw")
    print("start_control_mode:", control_mode)
    print("walk_command_threshold:", walk_command_threshold)
    if steps is None:
        print("policy_steps: unlimited, running until emergency stop or Ctrl+C")
    else:
        print("policy_steps:", steps)

    step = 0
    last_active_control_mode = None
    while steps is None or step < steps:
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
        if hasattr(estimator, "imu_stale") and estimator.imu_stale():
            print("\nEMERGENCY STOP: IMU data missing or stale")
            break

        joystick_emergency_reason = command_source.get_emergency_stop_request()
        if joystick_emergency_reason is not None:
            print("\nEMERGENCY STOP:", joystick_emergency_reason)
            break

        mode_request = command_source.get_mode_request()
        if mode_request is not None:
            if mode_request in ("stand", "sit"):
                request_feedback_snapshot(
                    motor_layer=motor_layer,
                    bus=bus,
                    mode=mode,
                )
                feedback_count = refresh_estimator_feedback(
                    estimator,
                    timeout=feedback_timeout,
                )
                (
                    q_current,
                    qd_current,
                    base_lin_vel_b,
                    base_ang_vel_b,
                    projected_gravity_b,
                ) = estimator.read()
                q_previous_target = q_current.copy()
                if feedback_count > 0:
                    print(
                        f"\n[FEEDBACK] pose transition starts from "
                        f"{feedback_count} measured motor angle(s)"
                    )

            control_mode = mode_request
            if control_mode in ("stand", "sit"):
                has_motion_target = True
            print(f"\n[MODE CHANGE] control_mode -> {control_mode}")

        command = command_source.read()
        walk_requested = joystick_walk_requested(command, walk_command_threshold)
        if control_mode == "sit":
            active_control_mode = "sit"
        elif walk_requested:
            active_control_mode = "policy"
        elif control_mode == "policy":
            active_control_mode = "hold"
        else:
            active_control_mode = control_mode

        if walk_requested:
            has_motion_target = True
        if active_control_mode == "policy" and last_active_control_mode != "policy":
            previous_action = np.zeros(12, dtype=np.float32)

        if active_control_mode == "hold":
            q_safe_target = q_previous_target.copy()
            commands = (
                motor_layer.build_mit_commands(q_safe_target, phase="startup")
                if has_motion_target
                else []
            )
            action = np.zeros(12, dtype=np.float32)

        elif active_control_mode == "stand":
            q_policy_target = runner.q_stand.copy()
            q_safe_target = safety.safety_filter(q_policy_target, q_previous_target)
            commands = motor_layer.build_mit_commands(q_safe_target, phase="startup")
            action = np.zeros(12, dtype=np.float32)

        elif active_control_mode == "sit":
            q_policy_target = runner.q_crouch.copy()
            q_safe_target = safety.safety_filter(q_policy_target, q_previous_target)
            commands = motor_layer.build_mit_commands(q_safe_target, phase="startup")
            action = np.zeros(12, dtype=np.float32)

        elif active_control_mode == "policy":
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
            raise RuntimeError(f"Unknown control_mode: {active_control_mode}")

        if mode == "signal":
            for cmd in commands:
                bus.send_signal_frame(cmd["motor_id"])
        elif mode == "mit-signal":
            motor_layer.send_signal_commands(bus, commands)

        refresh_estimator_feedback(estimator, timeout=feedback_timeout)

        if step % log_every == 0:
            print_compact_telemetry(
                step=step,
                mode=active_control_mode,
                command=command,
                command_source=command_source,
                commands=commands,
                estimator=estimator,
            )
            if show_hex and commands:
                print_mit_commands(commands, show_hex=True)

        estimator.dry_update_as_if_robot_followed(q_safe_target, dt)

        if active_control_mode == "policy":
            previous_action = action.copy()
        else:
            previous_action = np.zeros(12, dtype=np.float32)

        q_previous_target = q_safe_target.copy()
        last_active_control_mode = active_control_mode
        step += 1
        time.sleep(dt)

    print("\nPolicy / pose phase completed.")


def main():
    joystick_defaults = load_joystick_defaults()
    speed_defaults = load_speed_scale_defaults()
    imu_defaults = load_imu_config()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["print", "signal", "mit-signal", "motors"],
        default="print",
        help="print=no serial, signal=harmless empty CAN frames, mit-signal=sends MIT packets, motors=blocked",
    )

    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument(
        "--active-joints",
        nargs="*",
        default=None,
        help="only send motor commands to these joint names; default uses config/motor_ids.yaml",
    )
    parser.add_argument("--policy-path", default=None)
    parser.add_argument(
        "--policy-activation",
        choices=["elu", "relu", "tanh", "identity", "none"],
        default="elu",
        help="activation to use when loading non-TorchScript actor checkpoints",
    )

    parser.add_argument("--command-source", choices=["fixed", "joystick"], default="joystick")

    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)

    parser.add_argument("--max-vx", type=float, default=float(joystick_defaults["speed_limits"]["max_vx"]))
    parser.add_argument("--max-vy", type=float, default=float(joystick_defaults["speed_limits"]["max_vy"]))
    parser.add_argument("--max-yaw", type=float, default=float(joystick_defaults["speed_limits"]["max_yaw"]))

    parser.add_argument("--axis-vx", type=int, default=int(joystick_defaults["axes"]["vx_axis"]))
    parser.add_argument("--axis-vy", type=int, default=int(joystick_defaults["axes"]["vy_axis"]))
    parser.add_argument("--axis-yaw", type=int, default=int(joystick_defaults["axes"]["yaw_axis"]))
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument(
        "--joystick-wait-seconds",
        type=float,
        default=float(joystick_defaults.get("connection", {}).get("wait_seconds", 5.0)),
    )

    parser.add_argument("--invert-vx", action=argparse.BooleanOptionalAction, default=bool(joystick_defaults["invert"]["vx"]))
    parser.add_argument("--invert-vy", action=argparse.BooleanOptionalAction, default=bool(joystick_defaults["invert"]["vy"]))
    parser.add_argument("--invert-yaw", action=argparse.BooleanOptionalAction, default=bool(joystick_defaults["invert"]["yaw"]))

    parser.add_argument("--button-stand", type=int, default=int(joystick_defaults["buttons"]["stand"]))
    parser.add_argument("--button-sit", "--button-sit-stop", type=int, default=int(joystick_defaults["buttons"]["sit"]))
    parser.add_argument("--button-policy", type=int, default=int(joystick_defaults["buttons"]["policy"]))
    parser.add_argument(
        "--button-speed-down",
        type=int,
        default=int(joystick_defaults["buttons"].get("speed_down", 6)),
    )
    parser.add_argument(
        "--button-speed-up",
        type=int,
        default=int(joystick_defaults["buttons"].get("speed_up", 7)),
    )
    parser.add_argument(
        "--button-emergency-stop",
        type=int,
        nargs="+",
        default=[
            int(button_id)
            for button_id in joystick_defaults["buttons"].get("emergency_stop", [0, 1, 2, 3])
        ],
    )

    parser.add_argument("--deadzone", type=float, default=float(joystick_defaults["filter"]["deadzone"]))
    parser.add_argument("--expo", type=float, default=float(joystick_defaults["filter"]["expo"]))
    parser.add_argument("--smoothing", type=float, default=float(joystick_defaults["filter"]["smoothing"]))
    parser.add_argument("--speed-scale-initial", type=float, default=speed_defaults["initial"])
    parser.add_argument("--speed-scale-min", type=float, default=speed_defaults["min"])
    parser.add_argument("--speed-scale-max", type=float, default=speed_defaults["max"])
    parser.add_argument("--speed-scale-step", type=float, default=speed_defaults["step"])

    parser.add_argument(
        "--policy-steps",
        type=int,
        default=0,
        help="policy loop steps; 0 or negative runs until emergency stop",
    )
    parser.add_argument("--standup-seconds", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--show-hex", action="store_true")
    parser.add_argument("--start-control-mode", choices=["hold", "stand", "sit", "policy"], default="hold")
    parser.add_argument("--startup-action", choices=["hold", "stand"], default="hold")
    parser.add_argument(
        "--walk-command-threshold",
        type=float,
        default=1e-4,
        help="minimum absolute vx/vy/yaw command needed to run walking policy",
    )
    parser.add_argument(
        "--feedback-source",
        choices=["auto", "fake", "mit"],
        default="auto",
        help="auto uses MIT motor feedback in --mode mit-signal, otherwise fake feedback",
    )
    parser.add_argument(
        "--feedback-timeout",
        type=float,
        default=0.05,
        help="seconds to wait for MIT feedback after sending commands",
    )
    parser.add_argument(
        "--imu-source",
        choices=["auto", "fake", "none", "xsens", "serial-json", "serial-csv"],
        default="auto",
        help="auto uses config/imu.yaml; serial sources feed gyro/gravity into policy observation",
    )
    parser.add_argument("--imu-port", default=None)
    parser.add_argument("--imu-baud", type=int, default=None)

    parser.add_argument(
        "--fake-start",
        choices=["stand", "crouch", "random_small"],
        default="crouch",
    )

    args = parser.parse_args()
    args.log_every = max(1, args.log_every)
    args.policy_steps = None if args.policy_steps <= 0 else args.policy_steps
    args.walk_command_threshold = max(0.0, args.walk_command_threshold)

    if args.mode == "motors":
        print("ERROR: --mode motors is intentionally blocked for now.")
        print("Use --mode mit-signal for the real RobStride MIT CAN path.")
        return 1

    active_imu_source = imu_source_name(args.imu_source, imu_defaults)
    active_imu_port = args.imu_port if args.imu_port is not None else imu_defaults.get("port")
    if (
        args.mode in ("signal", "mit-signal")
        and active_imu_source in ("xsens", "xsens_binary", "mtdata2", "serial_json", "serial_csv")
        and active_imu_port == args.port
    ):
        print("ERROR: CAN port and IMU port are both set to:", args.port)
        print("Use separate devices, for example --port /dev/ttyUSB0 --imu-port /dev/ttyUSB1")
        return 1

    runner = PolicyRunner(
        policy_path=args.policy_path,
        policy_activation=args.policy_activation,
    )
    safety = SafetyMonitor(runner.policy_order)
    motor_ids = load_motor_ids()
    active_joints = args.active_joints if args.active_joints is not None else load_active_joints()
    motor_layer = MotorCommandLayer(
        runner.policy_order,
        motor_ids,
        active_joints=active_joints,
    )

    mit_cfg = load_yaml(ROOT / "config" / "mit_motor_control.yaml")
    standup_seconds = (
        float(args.standup_seconds)
        if args.standup_seconds is not None
        else float(mit_cfg["startup"]["standup_seconds"])
    )

    try:
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
            invert_vx=args.invert_vx,
            invert_vy=args.invert_vy,
            invert_yaw=args.invert_yaw,
            joystick_index=args.joystick_index,
            joystick_wait_seconds=args.joystick_wait_seconds,
            button_stand=args.button_stand,
            button_sit=args.button_sit,
            button_policy=args.button_policy,
            button_speed_down=args.button_speed_down,
            button_speed_up=args.button_speed_up,
            emergency_stop_buttons=args.button_emergency_stop,
            speed_scale_initial=args.speed_scale_initial,
            speed_scale_min=args.speed_scale_min,
            speed_scale_max=args.speed_scale_max,
            speed_scale_step=args.speed_scale_step,
            deadzone=args.deadzone,
            expo=args.expo,
            smoothing=args.smoothing,
        )
    except (ImportError, RuntimeError) as exc:
        print("ERROR:", exc)
        return 1

    q_fake_start = fake_start_pose_array(runner, args.fake_start)
    feedback_source = args.feedback_source
    if feedback_source == "auto":
        feedback_source = "mit" if args.mode == "mit-signal" else "fake"

    print("==== GRALLATOR JETSON MIT CONTROLLER ====")
    print("Mode:", args.mode)
    print("Command source:", args.command_source)
    print("Initial command:", command_source.read())
    print("Initial speed scale:", f"{command_speed_scale(command_source):.2f}")
    print("Start control mode:", args.start_control_mode)
    print("Startup action:", args.startup_action)
    print("Policy:", runner.policy_path)
    print("Policy format:", runner.policy_format)
    print("Policy obs/actions:", runner.observation_dim, runner.action_dim)
    print("Control dt:", runner.control_dt)
    print("Port:", args.port)
    print("Baud:", args.baud)
    print("Feedback source:", feedback_source)
    print("IMU source:", active_imu_source)
    print(
        "Active joints:",
        ", ".join(motor_layer.active_joints)
        if len(motor_layer.active_joints) < len(runner.policy_order)
        else "all",
    )
    print("Fallback fake start pose:", args.fake_start)
    print()

    print("Joint order and motor IDs:")
    for i, name in enumerate(runner.policy_order):
        print(f"{i:02d}: {name:16s} -> motor_id=0x{int(motor_ids[name]):02X}")

    bus = None
    imu_sensor = None

    try:
        imu_sensor = create_imu_sensor(
            source=args.imu_source,
            port=args.imu_port,
            baud=args.imu_baud,
        )
        if imu_sensor is None:
            print("\nIMU source disabled. Policy IMU fields will stay at fallback values.")
        else:
            print(
                "\nIMU opened:",
                getattr(imu_sensor, "source_name", "imu"),
                "port:",
                getattr(imu_sensor, "port", "none"),
            )

        if args.mode in ["signal", "mit-signal"]:
            if args.mode == "signal":
                print("\n--mode signal: sends harmless empty CAN frames only.")
                print("This tests USB-CAN transmission even if motors are absent.")
            elif args.mode == "mit-signal":
                print("\nWARNING: --mode mit-signal sends MIT control packets.")
                print("MIT motor feedback is read when the motors reply.")
                print("Use only with the robot secured or suspended for first tests.")

            print("\nOpening USB-CAN serial port...")
            bus = ATUsbCan(port=args.port, baud=args.baud).open()
            print("USB-CAN opened.")

        if feedback_source == "mit":
            if args.mode != "mit-signal" or bus is None:
                print("ERROR: --feedback-source mit requires --mode mit-signal.")
                return 1

            estimator = MitFeedbackStateEstimator(
                q_initial=q_fake_start,
                policy_order=runner.policy_order,
                motor_ids=motor_ids,
                motor_layer=motor_layer,
                bus=bus,
                imu_sensor=imu_sensor,
            )
            print("Polling initial motor encoder feedback...")
            motor_layer.send_raw_commands(bus, motor_layer.build_feedback_poll_commands())
            initial_feedback = estimator.refresh_from_bus(timeout=0.10)
            if initial_feedback <= 0:
                print(
                    "\nMIT feedback: no initial frames yet. "
                    "The estimator will update from motor replies after MIT commands."
                )
        else:
            estimator = FakeStateEstimator(q_initial=q_fake_start, imu_sensor=imu_sensor)

        if args.mode == "mit-signal":
            print("Sending motor enable frames...")
            motor_layer.send_raw_commands(bus, motor_layer.build_enable_commands())
            print("Motor enable frames sent.")

        if args.startup_action == "stand":
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
                feedback_timeout=args.feedback_timeout,
            )
        else:
            q_previous_target = initialize_hold_target(
                estimator=estimator,
                feedback_timeout=args.feedback_timeout,
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
            start_control_mode=args.start_control_mode,
            feedback_timeout=args.feedback_timeout,
            walk_command_threshold=args.walk_command_threshold,
        )

    except KeyboardInterrupt:
        print("\nKeyboard interrupt: stopping controller.")

    finally:
        command_source.close()
        if imu_sensor is not None:
            imu_sensor.close()
        if bus is not None:
            if args.mode == "mit-signal":
                try:
                    print("\nSending motor stop frames...")
                    motor_layer.send_raw_commands(bus, motor_layer.build_stop_commands())
                    print("Motor stop frames sent.")
                except Exception as exc:
                    print("\nWARNING: failed to send motor stop frames:", exc)
            bus.close()
            print("\nUSB-CAN closed.")

    print("\nController finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
