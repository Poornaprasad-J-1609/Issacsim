#!/usr/bin/env python3
from pathlib import Path
import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def float_to_uint(x, x_min, x_max, bits):
    x = float(np.clip(x, x_min, x_max))
    span = x_max - x_min
    return int((x - x_min) * ((1 << bits) - 1) / span)


def pack_mit_command(p_des, v_des, kp, kd, tau_ff, proto):
    """
    Pack MIT control command into 8 bytes:

    p_des  : 16 bit
    v_des  : 12 bit
    kp     : 12 bit
    kd     : 12 bit
    tau_ff : 12 bit
    """
    p_int = float_to_uint(p_des, proto["p_min"], proto["p_max"], 16)
    v_int = float_to_uint(v_des, proto["v_min"], proto["v_max"], 12)
    kp_int = float_to_uint(kp, proto["kp_min"], proto["kp_max"], 12)
    kd_int = float_to_uint(kd, proto["kd_min"], proto["kd_max"], 12)
    t_int = float_to_uint(tau_ff, proto["tau_min"], proto["tau_max"], 12)

    data = bytes([
        (p_int >> 8) & 0xFF,
        p_int & 0xFF,
        (v_int >> 4) & 0xFF,
        ((v_int & 0xF) << 4) | ((kp_int >> 8) & 0xF),
        kp_int & 0xFF,
        (kd_int >> 4) & 0xFF,
        ((kd_int & 0xF) << 4) | ((t_int >> 8) & 0xF),
        t_int & 0xFF,
    ])
    return data


def mit_can_id(motor_id, proto):
    comm_type = int(proto["comm_type_mit_control"])
    master_id = int(proto["master_id"])
    motor_id = int(motor_id)

    # RobStride/CyberGear-style extended ID layout:
    # comm_type in high byte, master_id in middle, motor_id low.
    return (comm_type << 24) | (master_id << 8) | motor_id


def joint_group(joint_name):
    if "hip" in joint_name:
        return "hip"
    if "thigh" in joint_name:
        return "thigh"
    if "calf" in joint_name:
        return "calf"
    raise ValueError(f"Cannot infer joint group from {joint_name}")


class MotorCommandLayer:
    def __init__(self, policy_order, motor_ids):
        self.policy_order = policy_order
        self.motor_ids = motor_ids

        self.cfg = load_yaml(ROOT / "config" / "mit_motor_control.yaml")
        self.proto = self.cfg["mit_protocol"]
        self.gains = self.cfg["gains"]
        self.feedforward = self.cfg["feedforward"]

    def build_mit_commands(self, q_target, phase="policy"):
        q_target = np.asarray(q_target, dtype=np.float32)
        commands = []

        if phase not in self.gains:
            raise ValueError(f"Unknown phase {phase}. Expected one of {list(self.gains.keys())}")

        for i, joint_name in enumerate(self.policy_order):
            motor_id = int(self.motor_ids[joint_name])
            group = joint_group(joint_name)

            kp = float(self.gains[phase][group]["kp"])
            kd = float(self.gains[phase][group]["kd"])
            v_des = float(self.feedforward["v_des"])
            tau_ff = float(self.feedforward["tau_ff"])
            p_des = float(q_target[i])

            can_id = mit_can_id(motor_id, self.proto)
            data = pack_mit_command(
                p_des=p_des,
                v_des=v_des,
                kp=kp,
                kd=kd,
                tau_ff=tau_ff,
                proto=self.proto,
            )

            commands.append({
                "joint_name": joint_name,
                "motor_id": motor_id,
                "phase": phase,
                "p_des": p_des,
                "v_des": v_des,
                "kp": kp,
                "kd": kd,
                "tau_ff": tau_ff,
                "can_id": can_id,
                "data": data,
            })

        return commands

    def send_signal_commands(self, bus, commands):
        """
        Sends MIT packets through USB-CAN.

        Motors do not have to be connected for the serial adapter to transmit.
        If motors ARE connected and powered, these packets can move them.
        """
        sent = []
        for cmd in commands:
            pkt = bus.send_raw(cmd["can_id"], cmd["data"])
            sent.append(pkt)
        return sent


def print_mit_commands(commands, show_hex=False):
    for cmd in commands:
        line = (
            f"motor_id={cmd['motor_id']:02d} "
            f"{cmd['joint_name']:16s} "
            f"phase={cmd['phase']:7s} "
            f"p={cmd['p_des']: .4f} "
            f"kp={cmd['kp']: .2f} "
            f"kd={cmd['kd']: .2f} "
            f"tau={cmd['tau_ff']: .2f}"
        )
        if show_hex:
            line += f" can_id=0x{cmd['can_id']:08X} data={cmd['data'].hex()}"
        print(line)


if __name__ == "__main__":
    from policy_runner import PolicyRunner

    runner = PolicyRunner()
    motor_cfg = load_yaml(ROOT / "config" / "motor_ids.yaml")
    layer = MotorCommandLayer(runner.policy_order, motor_cfg["motor_ids"])

    cmds = layer.build_mit_commands(runner.q_stand, phase="startup")
    print("MIT command example for STAND pose:")
    print_mit_commands(cmds, show_hex=True)
