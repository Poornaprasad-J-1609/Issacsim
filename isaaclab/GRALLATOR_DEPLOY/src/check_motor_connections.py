#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc

from motor_command_layer import MotorCommandLayer, decode_mit_feedback_frame
from robstride_can_interface import ATUsbCan


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config():
    joint_cfg = load_yaml(ROOT / "config" / "joint_map.yaml")
    motor_cfg = load_yaml(ROOT / "config" / "motor_ids.yaml")
    return joint_cfg, motor_cfg


def resolve_joints(policy_order, motor_cfg, active_joints_arg):
    if active_joints_arg is not None:
        joints = list(active_joints_arg)
    else:
        joints = list(motor_cfg.get("active_joints", []) or policy_order)

    unknown = [name for name in joints if name not in policy_order]
    if unknown:
        raise KeyError(f"Unknown joint name(s): {unknown}")

    missing_ids = [name for name in joints if name not in motor_cfg["motor_ids"]]
    if missing_ids:
        raise KeyError(f"Missing motor ID(s) in motor_ids.yaml: {missing_ids}")

    return joints


def feedback_status(feedback, now, stale_seconds):
    if feedback is None:
        return "NOT_CONNECTED"
    age = now - feedback["timestamp"]
    if age > stale_seconds:
        return "STALE"
    if int(feedback.get("fault_bits", 0)) != 0:
        return "FAULT"
    return "CONNECTED"


def print_table(joints, motor_ids, feedback_by_motor_id, stale_seconds):
    now = time.monotonic()
    print("\033[2J\033[H", end="")
    print("=" * 132)
    print("GRALLATOR ROBSTRIDE MOTOR CONNECTION CHECK")
    print("=" * 132)
    print("This sends RobStride comm-type 4 stop/poll frames only. It does not enable MIT torque control.")
    print("-" * 132)
    print(
        f"{'Joint':20s} | {'Motor':>7s} | {'State':>13s} | {'Age ms':>8s} | "
        f"{'Pos rad':>10s} | {'Vel rad/s':>10s} | {'Torque':>9s} | {'Temp C':>7s} | Fault"
    )
    print("-" * 132)

    connected = 0
    for joint_name in joints:
        motor_id = int(motor_ids[joint_name])
        feedback = feedback_by_motor_id.get(motor_id)
        state = feedback_status(feedback, now, stale_seconds)
        if state == "CONNECTED":
            connected += 1

        if feedback is None:
            print(
                f"{joint_name:20s} | 0x{motor_id:02X}    | {state:>13s} | "
                f"{'-':>8s} | {'-':>10s} | {'-':>10s} | {'-':>9s} | {'-':>7s} | -"
            )
            continue

        age_ms = 1000.0 * (now - feedback["timestamp"])
        print(
            f"{joint_name:20s} | 0x{motor_id:02X}    | {state:>13s} | "
            f"{age_ms:8.1f} | "
            f"{feedback['position']:+10.4f} | "
            f"{feedback['velocity']:+10.4f} | "
            f"{feedback['torque']:+9.4f} | "
            f"{feedback['temperature_c']:7.1f} | "
            f"0x{int(feedback['fault_bits']):02X}"
        )

    print("-" * 132)
    print(f"Connected: {connected}/{len(joints)}")
    print("=" * 132)


def main():
    joint_cfg, motor_cfg = load_config()
    policy_order = list(joint_cfg["policy_to_real_order"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--active-joints", nargs="*", default=None)
    parser.add_argument("--all", action="store_true", help="ignore config active_joints and check all configured joints")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--seconds", type=float, default=0.0, help="0 means run until Ctrl+C")
    parser.add_argument("--stale-seconds", type=float, default=0.5)
    parser.add_argument("--print-every", type=float, default=0.25)
    args = parser.parse_args()

    if args.all:
        joints = policy_order
    else:
        joints = resolve_joints(policy_order, motor_cfg, args.active_joints)

    motor_ids = motor_cfg["motor_ids"]
    layer = MotorCommandLayer(
        policy_order=policy_order,
        motor_ids=motor_ids,
        active_joints=joints,
    )
    poll_commands = layer.build_feedback_poll_commands()

    print("Opening RobStride AT USB-CAN adapter...")
    print(f"port={args.port}, baud={args.baud}")
    print("Checking joints:")
    for joint_name in layer.active_joints:
        print(f"  {joint_name:20s} -> 0x{int(motor_ids[joint_name]):02X}")

    bus = ATUsbCan(port=args.port, baud=args.baud, timeout=0.002).open()
    feedback_by_motor_id = {}
    active_motor_ids = {int(motor_ids[name]) for name in layer.active_joints}
    dt = 1.0 / max(1e-6, args.rate)
    deadline = None if args.seconds <= 0.0 else time.monotonic() + args.seconds
    next_print = 0.0

    try:
        while deadline is None or time.monotonic() < deadline:
            for cmd in poll_commands:
                bus.send_raw(cmd["can_id"], cmd["data"])
                time.sleep(0.001)

            frames = bus.read_available_frames(timeout=min(0.05, dt))
            now = time.monotonic()
            for frame in frames:
                feedback = decode_mit_feedback_frame(
                    frame.can_id,
                    frame.data,
                    layer.proto,
                )
                if feedback is None:
                    continue

                motor_id = int(feedback["motor_id"])
                if motor_id not in active_motor_ids:
                    continue
                feedback["timestamp"] = frame.timestamp
                feedback_by_motor_id[motor_id] = feedback

            if now >= next_print:
                print_table(
                    joints=layer.active_joints,
                    motor_ids=motor_ids,
                    feedback_by_motor_id=feedback_by_motor_id,
                    stale_seconds=args.stale_seconds,
                )
                next_print = now + args.print_every

            time.sleep(max(0.0, dt - 0.001 * len(poll_commands)))

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        bus.close()
        print("Serial closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
