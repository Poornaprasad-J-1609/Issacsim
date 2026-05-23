#!/usr/bin/env python3
from pathlib import Path
import argparse
import time
import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("Install PyYAML first: pip3 install pyyaml") from exc


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_deadzone(x, deadzone):
    x = float(x)
    if abs(x) < deadzone:
        return 0.0
    sign = 1.0 if x >= 0.0 else -1.0
    return sign * (abs(x) - deadzone) / (1.0 - deadzone)


def expo_curve(x, expo):
    return (1.0 - expo) * x + expo * (x ** 3)


class FixedCommandSource:
    def __init__(self, vx=0.05, vy=0.0, yaw=0.0):
        cfg = load_yaml(ROOT / "config" / "joint_map.yaml")
        lim = cfg["command_limits"]

        self.vx_min = float(lim["vx_min"])
        self.vx_max = float(lim["vx_max"])
        self.vy_min = float(lim["vy_min"])
        self.vy_max = float(lim["vy_max"])
        self.yaw_min = float(lim["yaw_min"])
        self.yaw_max = float(lim["yaw_max"])

        self.command = np.array([vx, vy, yaw], dtype=np.float32)
        self.command = self.clip_command(self.command)

    def clip_command(self, command):
        command = np.asarray(command, dtype=np.float32).copy()
        command[0] = np.clip(command[0], self.vx_min, self.vx_max)
        command[1] = np.clip(command[1], self.vy_min, self.vy_max)
        command[2] = np.clip(command[2], self.yaw_min, self.yaw_max)
        return command

    def read(self):
        return self.command.copy()

    def get_mode_request(self):
        return None


class JoystickCommandSource:
    """
    Default Xbox / Logitech-like mapping:
      left stick Y  -> vx
      left stick X  -> vy
      right stick X -> yaw

    Buttons:
      button 0 -> stand
      button 1 -> sit/crouch
      button 2 -> policy walking
    """

    def __init__(
        self,
        max_vx=0.15,
        max_vy=0.10,
        max_yaw=0.30,
        axis_vx=1,
        axis_vy=0,
        axis_yaw=3,
        invert_vx=True,
        invert_vy=False,
        invert_yaw=False,
        deadzone=0.08,
        expo=0.35,
        smoothing=0.20,
        joystick_index=0,
        button_stand=0,
        button_sit=1,
        button_policy=2,
    ):
        self.max_vx = float(max_vx)
        self.max_vy = float(max_vy)
        self.max_yaw = float(max_yaw)

        self.axis_vx = int(axis_vx)
        self.axis_vy = int(axis_vy)
        self.axis_yaw = int(axis_yaw)

        self.invert_vx = bool(invert_vx)
        self.invert_vy = bool(invert_vy)
        self.invert_yaw = bool(invert_yaw)

        self.deadzone = float(deadzone)
        self.expo = float(expo)
        self.smoothing = float(smoothing)

        self.button_stand = int(button_stand)
        self.button_sit = int(button_sit)
        self.button_policy = int(button_policy)

        self.command = np.zeros(3, dtype=np.float32)
        self.prev_buttons = {}

        try:
            import os
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            import pygame
        except ImportError as exc:
            raise ImportError("Install pygame first: pip3 install pygame") from exc

        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count <= 0:
            raise RuntimeError(
                "No joystick found. On Jetson, check: ls /dev/input/js* /dev/input/event*"
            )

        if joystick_index >= count:
            raise RuntimeError(f"joystick_index={joystick_index} but only {count} joystick(s) found")

        self.joy = pygame.joystick.Joystick(joystick_index)
        self.joy.init()

        print("Joystick connected:")
        print("  name:", self.joy.get_name())
        print("  axes:", self.joy.get_numaxes())
        print("  buttons:", self.joy.get_numbuttons())
        print("  hats:", self.joy.get_numhats())
        print("Button mapping:")
        print(f"  stand  button: {self.button_stand}")
        print(f"  sit    button: {self.button_sit}")
        print(f"  policy button: {self.button_policy}")

    def _axis(self, axis_id):
        if axis_id < 0 or axis_id >= self.joy.get_numaxes():
            return 0.0
        return float(self.joy.get_axis(axis_id))

    def _button(self, button_id):
        if button_id < 0 or button_id >= self.joy.get_numbuttons():
            return False
        return bool(self.joy.get_button(button_id))

    def _button_rising_edge(self, button_id):
        now = self._button(button_id)
        prev = self.prev_buttons.get(button_id, False)
        self.prev_buttons[button_id] = now
        return now and not prev

    def read(self):
        self.pygame.event.pump()

        raw_vx = self._axis(self.axis_vx)
        raw_vy = self._axis(self.axis_vy)
        raw_yaw = self._axis(self.axis_yaw)

        if self.invert_vx:
            raw_vx = -raw_vx
        if self.invert_vy:
            raw_vy = -raw_vy
        if self.invert_yaw:
            raw_yaw = -raw_yaw

        vx = expo_curve(apply_deadzone(raw_vx, self.deadzone), self.expo) * self.max_vx
        vy = expo_curve(apply_deadzone(raw_vy, self.deadzone), self.expo) * self.max_vy
        yaw = expo_curve(apply_deadzone(raw_yaw, self.deadzone), self.expo) * self.max_yaw

        target = np.array([vx, vy, yaw], dtype=np.float32)
        self.command = (1.0 - self.smoothing) * self.command + self.smoothing * target

        return self.command.copy()

    def get_mode_request(self):
        self.pygame.event.pump()

        if self._button_rising_edge(self.button_stand):
            return "stand"

        if self._button_rising_edge(self.button_sit):
            return "sit"

        if self._button_rising_edge(self.button_policy):
            return "policy"

        return None


class CommandSource:
    def __init__(self, source="fixed", **kwargs):
        if source == "fixed":
            self.impl = FixedCommandSource(
                vx=kwargs.get("vx", 0.05),
                vy=kwargs.get("vy", 0.0),
                yaw=kwargs.get("yaw", 0.0),
            )
        elif source == "joystick":
            self.impl = JoystickCommandSource(
                max_vx=kwargs.get("max_vx", 0.15),
                max_vy=kwargs.get("max_vy", 0.10),
                max_yaw=kwargs.get("max_yaw", 0.30),
                axis_vx=kwargs.get("axis_vx", 1),
                axis_vy=kwargs.get("axis_vy", 0),
                axis_yaw=kwargs.get("axis_yaw", 3),
                deadzone=kwargs.get("deadzone", 0.08),
                expo=kwargs.get("expo", 0.35),
                smoothing=kwargs.get("smoothing", 0.20),
                joystick_index=kwargs.get("joystick_index", 0),
                button_stand=kwargs.get("button_stand", 0),
                button_sit=kwargs.get("button_sit", 1),
                button_policy=kwargs.get("button_policy", 2),
            )
        else:
            raise ValueError(f"Unknown command source: {source}")

    def read(self):
        return self.impl.read()

    def get_mode_request(self):
        return self.impl.get_mode_request()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["fixed", "joystick"], default="fixed")

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
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=20.0)

    args = parser.parse_args()

    source = CommandSource(
        source=args.source,
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

    dt = 1.0 / args.hz
    steps = int(args.seconds * args.hz)

    print("Testing command source:", args.source)
    print("Press stand/sit/policy buttons if using joystick.")
    for i in range(steps):
        cmd = source.read()
        mode_req = source.get_mode_request()
        print(
            f"step={i:04d} "
            f"vx={cmd[0]: .3f} vy={cmd[1]: .3f} yaw={cmd[2]: .3f} "
            f"mode_request={mode_req}"
        )
        time.sleep(dt)


if __name__ == "__main__":
    main()
