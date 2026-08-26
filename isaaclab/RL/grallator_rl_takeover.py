#!/usr/bin/env python3
"""
Grallator 12x RS04: working SIT/STAND path + automatic RL takeover.

Sequence:
    1. Put/support robot in the verified STAND pose.
    2. Press E -> capture raw stand reference and enable MIT.
    3. Press S -> SIT using the verified sit limits.
    4. Press W -> STAND.
    5. When STAND finishes, the RL policy automatically takes over.

Motor-side MIT law:
    tau = Kp*(q_des - q) + Kd*(v_des - qdot) + tau_ff

This program always sends:
    v_des  = 0.0
    tau_ff = 0.0

Therefore:
    tau = Kp*(q_des - q) - Kd*qdot

"Any policy" means any feed-forward 12-action TorchScript policy for the same
Grallator joint convention whose exact observation/action contract is described
by the JSON policy spec.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from robstride_dynamics import Motor, ParameterType, RobstrideBus


CAN0 = "slcan0"
CAN1 = "slcan1"
MOTOR_MODEL = "rs-04"

JOINTS = {
    "FR_hip_joint":   {"id": 1,  "bus": CAN0, "group": "hip"},
    "FR_thigh_joint": {"id": 2,  "bus": CAN0, "group": "thigh"},
    "FR_calf_joint":  {"id": 3,  "bus": CAN0, "group": "calf"},

    "FL_hip_joint":   {"id": 4,  "bus": CAN0, "group": "hip"},
    "FL_thigh_joint": {"id": 5,  "bus": CAN0, "group": "thigh"},
    "FL_calf_joint":  {"id": 6,  "bus": CAN0, "group": "calf"},

    "BR_hip_joint":   {"id": 7,  "bus": CAN1, "group": "hip"},
    "BR_thigh_joint": {"id": 8,  "bus": CAN1, "group": "thigh"},
    "BR_calf_joint":  {"id": 9,  "bus": CAN1, "group": "calf"},

    "BL_hip_joint":   {"id": 10, "bus": CAN1, "group": "hip"},
    "BL_thigh_joint": {"id": 11, "bus": CAN1, "group": "thigh"},
    "BL_calf_joint":  {"id": 12, "bus": CAN1, "group": "calf"},
}

ALL_JOINT_NAMES = tuple(JOINTS.keys())

# Verified SIT/STAND limits.
STAND_TARGET = {name: 0.0 for name in ALL_JOINT_NAMES}

SIT_TARGET = {
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": -0.65,
    "FR_calf_joint": -1.35,

    "FL_hip_joint": 0.0,
    "FL_thigh_joint": +0.65,
    "FL_calf_joint": +1.35,

    "BR_hip_joint": 0.0,
    "BR_thigh_joint": -0.65,
    "BR_calf_joint": -1.35,

    "BL_hip_joint": 0.0,
    "BL_thigh_joint": +0.65,
    "BL_calf_joint": +1.35,
}

# KEEP THESE IDENTICAL TO THE VALUES THAT WORKED IN YOUR SIT/STAND TEST.
#
# q_raw_target = q_raw_stand
#              + MOTOR_DIRECTION[joint] * (q_joint_target - q_joint_stand)
MOTOR_DIRECTION = {
    "FR_hip_joint": +1,
    "FR_thigh_joint": +1,
    "FR_calf_joint": +1,

    "FL_hip_joint": +1,
    "FL_thigh_joint": +1,
    "FL_calf_joint": +1,

    "BR_hip_joint": +1,
    "BR_thigh_joint": +1,
    "BR_calf_joint": +1,

    "BL_hip_joint": +1,
    "BL_thigh_joint": +1,
    "BL_calf_joint": +1,
}

# Working SIT/STAND gains. Override via CLI if your tested values differ.
DEFAULT_POSE_GAINS = {
    "hip":   {"kp": 5.0, "kd": 0.35},
    "thigh": {"kp": 6.0, "kd": 0.45},
    "calf":  {"kp": 6.0, "kd": 0.45},
}

DEFAULT_MOTOR_HZ = 200.0
DEFAULT_TRANSITION_SECONDS = 4.0
DEFAULT_TAKEOVER_DELAY_SECONDS = 0.25


def resolve_parameter(*names):
    for name in names:
        if hasattr(ParameterType, name):
            return getattr(ParameterType, name), name

    raise RuntimeError(
        "None of these ParameterType entries exist in this SDK: "
        + ", ".join(names)
    )


POSITION_PARAMETER, POSITION_PARAMETER_NAME = resolve_parameter(
    "MECHANICAL_POSITION",
    "MEASURED_POSITION",
)

VELOCITY_PARAMETER, VELOCITY_PARAMETER_NAME = resolve_parameter(
    "MECHANICAL_VELOCITY",
    "MEASURED_VELOCITY",
)


def smooth_alpha(s: float) -> float:
    s = max(0.0, min(1.0, float(s)))
    return 0.5 - 0.5 * math.cos(math.pi * s)


def require_finite_vector(value, size, name):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)

    if arr.size != size:
        raise ValueError(
            f"{name} must contain {size} values; got {arr.size}"
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")

    return arr


# ============================================================================
# Policy
# ============================================================================

SUPPORTED_OBS_TERMS = {
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "command": 3,
    "joint_pos": 12,
    "joint_pos_rel": 12,
    "joint_vel": 12,
    "previous_action": 12,
}


class TorchPolicy:
    def __init__(self, spec_path, model_path_override=None):
        self.spec_path = Path(spec_path).expanduser().resolve()

        with self.spec_path.open("r") as f:
            self.spec = json.load(f)

        self._validate_spec()

        model_path = Path(
            model_path_override
            if model_path_override is not None
            else self.spec["policy_path"]
        )
        if not model_path.is_absolute():
            model_path = self.spec_path.parent / model_path

        self.model_path = model_path.resolve()
        self._verify_artifact()
        self.policy, self.policy_format = self._load_policy()
        self.policy.eval()

        self.joint_order = list(self.spec["joint_order"])

        self.default_joint_pos = np.array(
            [
                float(self.spec["default_joint_pos"][name])
                for name in self.joint_order
            ],
            dtype=np.float32,
        )

        self.stand_joint_pos = np.array(
            [
                float(self.spec["stand_joint_pos"][name])
                for name in self.joint_order
            ],
            dtype=np.float32,
        )

        self.action_scale = float(self.spec["action_scale"])
        action_clip = self.spec.get("action_clip", None)
        self.action_clip = (
            None
            if action_clip is None
            else float(action_clip)
        )

        obs_clip = self.spec.get("obs_clip", None)
        self.obs_clip = (
            None
            if obs_clip is None
            else float(obs_clip)
        )

        self.policy_hz = float(self.spec["policy_hz"])
        self.policy_dt = 1.0 / self.policy_hz

        self.obs_terms = list(
            self.spec["observation_terms"]
        )

        self.obs_dim = sum(
            SUPPORTED_OBS_TERMS[item["name"]]
            for item in self.obs_terms
        )

        self.policy_gains = self.spec["policy_gains"]
        self.policy_gains_by_joint = self.spec.get(
            "policy_gains_by_joint"
        )

    def _verify_artifact(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)

        expected = self.spec.get("expected_sha256")
        if expected is None:
            return

        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if digest.lower() != str(expected).lower():
            raise RuntimeError(
                "Policy SHA256 mismatch: "
                f"expected {expected}, got {digest}"
            )

    def _load_policy(self):
        try:
            return (
                torch.jit.load(str(self.model_path), map_location="cpu"),
                "torchscript",
            )
        except (RuntimeError, ValueError):
            pass

        try:
            checkpoint = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location="cpu")

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise RuntimeError(
                "Policy is neither TorchScript nor an RSL-RL checkpoint "
                "containing model_state_dict"
            )

        actor_state = {
            key[len("actor."):]: value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("actor.")
        }
        expected_shapes = {
            "0.weight": (512, 48),
            "0.bias": (512,),
            "2.weight": (256, 512),
            "2.bias": (256,),
            "4.weight": (128, 256),
            "4.bias": (128,),
            "6.weight": (12, 128),
            "6.bias": (12,),
        }
        actual_shapes = {
            key: tuple(value.shape) for key, value in actor_state.items()
        }
        if actual_shapes != expected_shapes:
            raise RuntimeError(
                "Unsupported RSL-RL actor tensors: "
                f"expected {expected_shapes}, got {actual_shapes}"
            )

        actor = nn.Sequential(
            nn.Linear(48, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 12),
        )
        actor.load_state_dict(actor_state, strict=True)
        return actor, "rsl_rl_checkpoint"

    def _validate_spec(self):
        required = (
            "policy_path",
            "policy_hz",
            "action_scale",
            "joint_order",
            "default_joint_pos",
            "stand_joint_pos",
            "observation_terms",
            "policy_gains",
            "joint_limits",
        )

        missing = [
            key
            for key in required
            if key not in self.spec
        ]

        if missing:
            raise ValueError(
                f"Policy spec missing: {missing}"
            )

        if self.spec["action_scale"] is None:
            raise ValueError(
                "action_scale is null. Set the EXACT "
                "action scale used during training."
            )

        order = list(self.spec["joint_order"])

        if (
            len(order) != 12
            or set(order) != set(ALL_JOINT_NAMES)
        ):
            raise ValueError(
                "joint_order must contain each of the "
                "12 Grallator joints exactly once."
            )

        for key in (
            "default_joint_pos",
            "stand_joint_pos",
        ):
            if (
                set(self.spec[key].keys())
                != set(ALL_JOINT_NAMES)
            ):
                raise ValueError(
                    f"{key} must define all 12 joints"
                )

        policy_hz = float(self.spec["policy_hz"])
        if not 1.0 <= policy_hz <= 200.0:
            raise ValueError(
                "policy_hz must be within 1..200"
            )

        scale = float(self.spec["action_scale"])
        if not math.isfinite(scale):
            raise ValueError(
                "action_scale must be finite"
            )

        for item in self.spec["observation_terms"]:
            if "name" not in item:
                raise ValueError(
                    "Each observation term needs a name"
                )

            name = item["name"]

            if name not in SUPPORTED_OBS_TERMS:
                raise ValueError(
                    f"Unsupported observation term {name!r}. "
                    f"Supported: {sorted(SUPPORTED_OBS_TERMS)}"
                )

            if item.get("scale", None) is None:
                raise ValueError(
                    f"Observation scale for {name} is null. "
                    "Set it to the EXACT scale used during training."
                )

            obs_scale = float(item["scale"])

            if not math.isfinite(obs_scale):
                raise ValueError(
                    f"Invalid scale for {name}"
                )

        if (
            set(self.spec["joint_limits"].keys())
            != set(ALL_JOINT_NAMES)
        ):
            raise ValueError(
                "joint_limits must define all 12 joints"
            )

        for joint_name, limits in self.spec["joint_limits"].items():
            if (
                not isinstance(limits, list)
                or len(limits) != 2
            ):
                raise ValueError(
                    f"joint_limits.{joint_name} must be [min, max]"
                )

            q_min = float(limits[0])
            q_max = float(limits[1])

            if (
                not math.isfinite(q_min)
                or not math.isfinite(q_max)
                or q_min >= q_max
            ):
                raise ValueError(
                    f"Invalid joint limits for {joint_name}: {limits}"
                )

        for group in (
            "hip",
            "thigh",
            "calf",
        ):
            if group not in self.spec["policy_gains"]:
                raise ValueError(
                    f"policy_gains missing {group}"
                )

            for term in ("kp", "kd"):
                value = (
                    self.spec["policy_gains"]
                    [group]
                    .get(term)
                )

                if value is None:
                    raise ValueError(
                        f"policy_gains.{group}.{term} "
                        "is null. Set the EXACT "
                        "deployment gain for this policy."
                    )

                value = float(value)

                if (
                    not math.isfinite(value)
                    or value < 0.0
                ):
                    raise ValueError(
                        f"Invalid {group} {term}: {value}"
                    )

        gains_by_joint = self.spec.get("policy_gains_by_joint")
        if gains_by_joint is not None:
            if set(gains_by_joint) != set(ALL_JOINT_NAMES):
                raise ValueError(
                    "policy_gains_by_joint must define all 12 joints"
                )
            for name, gains in gains_by_joint.items():
                for term in ("kp", "kd"):
                    value = float(gains[term])
                    if not math.isfinite(value) or value < 0.0:
                        raise ValueError(
                            f"Invalid {name} {term}: {value}"
                        )

    def build_observation(
        self,
        *,
        base_lin_vel,
        base_ang_vel,
        projected_gravity,
        command,
        q_joint,
        qd_joint,
        previous_action,
    ):
        q_joint = require_finite_vector(
            q_joint,
            12,
            "q_joint",
        )

        qd_joint = require_finite_vector(
            qd_joint,
            12,
            "qd_joint",
        )

        previous_action = require_finite_vector(
            previous_action,
            12,
            "previous_action",
        )

        values = {
            "base_lin_vel": require_finite_vector(
                base_lin_vel,
                3,
                "base_lin_vel",
            ),
            "base_ang_vel": require_finite_vector(
                base_ang_vel,
                3,
                "base_ang_vel",
            ),
            "projected_gravity": require_finite_vector(
                projected_gravity,
                3,
                "projected_gravity",
            ),
            "command": require_finite_vector(
                command,
                3,
                "command",
            ),
            "joint_pos": q_joint,
            "joint_pos_rel": (
                q_joint
                - self.default_joint_pos
            ),
            "joint_vel": qd_joint,
            "previous_action": previous_action,
        }

        pieces = []

        for item in self.obs_terms:
            name = item["name"]
            scale = float(
                item.get("scale", 1.0)
            )

            pieces.append(
                values[name] * scale
            )

        obs = np.concatenate(
            pieces
        ).astype(np.float32)

        if self.obs_clip is not None:
            obs = np.clip(
                obs,
                -self.obs_clip,
                self.obs_clip,
            )

        return obs

    def infer(self, obs):
        obs = require_finite_vector(
            obs,
            self.obs_dim,
            "policy observation",
        )

        obs_t = (
            torch.from_numpy(obs)
            .to(dtype=torch.float32)
            .unsqueeze(0)
        )

        with torch.no_grad():
            output = self.policy(obs_t)

        if isinstance(
            output,
            (tuple, list),
        ):
            output = output[0]

        if not torch.is_tensor(output):
            raise RuntimeError(
                "Policy returned unsupported type: "
                f"{type(output).__name__}"
            )

        action = (
            output
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(-1)
        )

        if action.size != 12:
            raise RuntimeError(
                "Policy must output 12 actions; "
                f"got {action.size}"
            )

        if not np.all(
            np.isfinite(action)
        ):
            raise RuntimeError(
                "Policy returned non-finite actions"
            )

        if self.action_clip is not None:
            action = np.clip(
                action,
                -self.action_clip,
                self.action_clip,
            )

        return action

    def action_to_joint_target(
        self,
        action,
    ):
        action = require_finite_vector(
            action,
            12,
            "action",
        )

        q_target = (
            self.default_joint_pos
            + self.action_scale * action
        )

        result = {}

        for i, name in enumerate(self.joint_order):
            q_min, q_max = self.spec["joint_limits"][name]
            result[name] = float(
                np.clip(
                    q_target[i],
                    float(q_min),
                    float(q_max),
                )
            )

        return result

    def ordered_joint_state(
        self,
        q_dict,
    ):
        return np.array(
            [
                float(q_dict[name])
                for name in self.joint_order
            ],
            dtype=np.float32,
        )


# ============================================================================
# IMU plugin
# ============================================================================

class ExternalIMU:
    """
    Loads a Python file containing:

        class IMUProvider:
            def read(self):
                return {
                    "base_ang_vel": [wx, wy, wz],
                    "projected_gravity": [gx, gy, gz],
                    "base_lin_vel": [vx, vy, vz],  # optional
                }
    """

    def __init__(
        self,
        module_path,
    ):
        module_path = (
            Path(module_path)
            .expanduser()
            .resolve()
        )

        if not module_path.exists():
            raise FileNotFoundError(
                module_path
            )

        module_spec = (
            importlib.util
            .spec_from_file_location(
                "grallator_external_imu",
                str(module_path),
            )
        )

        if (
            module_spec is None
            or module_spec.loader is None
        ):
            raise RuntimeError(
                "Could not load IMU provider: "
                f"{module_path}"
            )

        module = (
            importlib.util
            .module_from_spec(module_spec)
        )

        module_spec.loader.exec_module(
            module
        )

        if not hasattr(
            module,
            "IMUProvider",
        ):
            raise RuntimeError(
                f"{module_path} must define "
                "class IMUProvider"
            )

        self.provider = module.IMUProvider()

    def read(self):
        data = self.provider.read()

        if not isinstance(
            data,
            dict,
        ):
            raise RuntimeError(
                "IMUProvider.read() must return a dict"
            )

        base_ang_vel = require_finite_vector(
            data["base_ang_vel"],
            3,
            "base_ang_vel",
        )

        projected_gravity = require_finite_vector(
            data["projected_gravity"],
            3,
            "projected_gravity",
        )

        base_lin_vel = require_finite_vector(
            data.get(
                "base_lin_vel",
                [0.0, 0.0, 0.0],
            ),
            3,
            "base_lin_vel",
        )

        return (
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
        )


@dataclass
class Transition:
    start_time: float
    duration: float
    q_start_raw: dict[str, float]
    q_goal_raw: dict[str, float]
    destination: str
    auto_policy_after: bool


class GrallatorRLController:
    def __init__(
        self,
        *,
        can0,
        can1,
        motor_hz,
        transition_seconds,
        takeover_delay,
        pose_gains,
        policy,
        imu,
    ):
        self.can0_name = can0
        self.can1_name = can1
        self.motor_hz = float(motor_hz)
        self.transition_seconds = float(
            transition_seconds
        )
        self.takeover_delay = float(
            takeover_delay
        )
        self.pose_gains = pose_gains
        self.policy = policy
        self.imu = imu

        self.lock = threading.RLock()
        self.stop_event = threading.Event()

        self.connected = False
        self.enabled = False
        self.mode = "DISABLED"

        self.stand_reference_raw = None
        self.hold_target_raw = None
        self.transition = None

        self.policy_pending_at = None
        self.policy_target_raw = None
        self.previous_action = np.zeros(
            12,
            dtype=np.float32,
        )
        self.next_policy_tick = None

        self.command = np.zeros(
            3,
            dtype=np.float32,
        )

        motors0 = {
            name: Motor(
                id=cfg["id"],
                model=MOTOR_MODEL,
            )
            for name, cfg in JOINTS.items()
            if cfg["bus"] == CAN0
        }

        motors1 = {
            name: Motor(
                id=cfg["id"],
                model=MOTOR_MODEL,
            )
            for name, cfg in JOINTS.items()
            if cfg["bus"] == CAN1
        }

        self.bus0 = RobstrideBus(
            channel=can0,
            motors=motors0,
        )

        self.bus1 = RobstrideBus(
            channel=can1,
            motors=motors1,
        )

        self.bus_locks = {
            CAN0: threading.Lock(),
            CAN1: threading.Lock(),
        }

        self.executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="grallator_can",
        )

        self.control_thread = None

    def connect(self):
        f0 = self.executor.submit(
            self.bus0.connect
        )
        f1 = self.executor.submit(
            self.bus1.connect
        )

        f0.result()
        f1.result()

        self.connected = True

    # ------------------------------------------------------------------
    # Real q / qdot
    # ------------------------------------------------------------------

    def _read_bus_state(
        self,
        bus_name,
    ):
        bus = (
            self.bus0
            if bus_name == CAN0
            else self.bus1
        )

        result = {}

        with self.bus_locks[bus_name]:
            for name, cfg in JOINTS.items():
                if cfg["bus"] != bus_name:
                    continue

                q_raw = float(
                    bus.read(
                        name,
                        POSITION_PARAMETER,
                    )
                )

                qd_raw = float(
                    bus.read(
                        name,
                        VELOCITY_PARAMETER,
                    )
                )

                if (
                    not math.isfinite(q_raw)
                    or not math.isfinite(qd_raw)
                ):
                    raise RuntimeError(
                        f"Bad feedback from {name}: "
                        f"q={q_raw}, qd={qd_raw}"
                    )

                result[name] = (
                    q_raw,
                    qd_raw,
                )

        return result

    def read_all_raw_state(self):
        f0 = self.executor.submit(
            self._read_bus_state,
            CAN0,
        )

        f1 = self.executor.submit(
            self._read_bus_state,
            CAN1,
        )

        result = {}
        result.update(f0.result())
        result.update(f1.result())

        return result

    def _stand_pose_dict(self):
        return {
            name: float(
                self.policy.stand_joint_pos[
                    self.policy.joint_order.index(
                        name
                    )
                ]
            )
            for name in ALL_JOINT_NAMES
        }

    def _raw_to_joint_state(
        self,
        raw_state,
    ):
        if self.stand_reference_raw is None:
            raise RuntimeError(
                "Stand reference is not captured"
            )

        stand_pose = (
            self._stand_pose_dict()
        )

        q_joint = {}
        qd_joint = {}

        for name in ALL_JOINT_NAMES:
            q_raw, qd_raw = raw_state[name]
            direction = float(
                MOTOR_DIRECTION[name]
            )

            q_joint[name] = (
                stand_pose[name]
                + direction
                * (
                    q_raw
                    - self.stand_reference_raw[name]
                )
            )

            qd_joint[name] = (
                direction * qd_raw
            )

        return q_joint, qd_joint

    def _joint_target_to_raw(
        self,
        joint_target,
    ):
        if self.stand_reference_raw is None:
            raise RuntimeError(
                "Stand reference is not captured"
            )

        stand_pose = (
            self._stand_pose_dict()
        )

        return {
            name: (
                self.stand_reference_raw[name]
                + MOTOR_DIRECTION[name]
                * (
                    float(joint_target[name])
                    - stand_pose[name]
                )
            )
            for name in ALL_JOINT_NAMES
        }

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def _enable_bus(
        self,
        bus_name,
    ):
        bus = (
            self.bus0
            if bus_name == CAN0
            else self.bus1
        )

        with self.bus_locks[bus_name]:
            for name, cfg in JOINTS.items():
                if cfg["bus"] != bus_name:
                    continue

                bus.disable(name)
                bus.set_run_mode(
                    name,
                    0,
                )
                bus.enable(name)

    def enable_and_capture_stand(self):
        if not self.connected:
            raise RuntimeError(
                "CAN buses are not connected"
            )

        raw_state = (
            self.read_all_raw_state()
        )

        q_now = {
            name: value[0]
            for name, value
            in raw_state.items()
        }

        with self.lock:
            self.stand_reference_raw = dict(
                q_now
            )
            self.hold_target_raw = dict(
                q_now
            )
            self.policy_target_raw = dict(
                q_now
            )

            self.transition = None
            self.policy_pending_at = None
            self.previous_action[:] = 0.0
            self.next_policy_tick = None
            self.mode = "HOLD"

        f0 = self.executor.submit(
            self._enable_bus,
            CAN0,
        )

        f1 = self.executor.submit(
            self._enable_bus,
            CAN1,
        )

        f0.result()
        f1.result()

        with self.lock:
            self.enabled = True

        if (
            self.control_thread is None
            or not self.control_thread.is_alive()
        ):
            self.control_thread = threading.Thread(
                target=self._control_loop,
                name="grallator_rl_control",
                daemon=True,
            )
            self.control_thread.start()

    def _disable_bus(
        self,
        bus_name,
    ):
        bus = (
            self.bus0
            if bus_name == CAN0
            else self.bus1
        )

        with self.bus_locks[bus_name]:
            for name, cfg in JOINTS.items():
                if cfg["bus"] == bus_name:
                    try:
                        bus.disable(name)
                    except Exception:
                        pass

    def disable_all(self):
        with self.lock:
            self.enabled = False
            self.mode = "DISABLED"
            self.transition = None
            self.policy_pending_at = None

        f0 = self.executor.submit(
            self._disable_bus,
            CAN0,
        )

        f1 = self.executor.submit(
            self._disable_bus,
            CAN1,
        )

        f0.result()
        f1.result()

    # ------------------------------------------------------------------
    # SIT / STAND
    # ------------------------------------------------------------------

    def _pose_goal_raw(
        self,
        pose,
    ):
        if pose == "SIT":
            target = SIT_TARGET

        elif pose == "STAND":
            # The raw encoder pose captured with E is defined to correspond
            # exactly to policy_spec["stand_joint_pos"].
            target = {
                name: float(
                    self.policy.stand_joint_pos[
                        self.policy.joint_order.index(name)
                    ]
                )
                for name in ALL_JOINT_NAMES
            }

        else:
            raise ValueError(pose)

        return self._joint_target_to_raw(
            target
        )

    def _current_pose_target(
        self,
        now,
    ):
        with self.lock:
            if self.hold_target_raw is None:
                raise RuntimeError(
                    "No hold target available"
                )

            transition = self.transition

            if transition is None:
                return dict(
                    self.hold_target_raw
                )

            elapsed = (
                now
                - transition.start_time
            )

            if elapsed >= transition.duration:
                self.hold_target_raw = dict(
                    transition.q_goal_raw
                )

                self.transition = None
                self.mode = transition.destination

                if transition.auto_policy_after:
                    self.policy_pending_at = (
                        now
                        + self.takeover_delay
                    )

                return dict(
                    self.hold_target_raw
                )

            alpha = smooth_alpha(
                elapsed
                / transition.duration
            )

            return {
                name: (
                    transition.q_start_raw[name]
                    + alpha
                    * (
                        transition.q_goal_raw[name]
                        - transition.q_start_raw[name]
                    )
                )
                for name in ALL_JOINT_NAMES
            }

    def _current_commanded_target(
        self,
        now,
    ):
        with self.lock:
            if self.mode == "POLICY":
                if self.policy_target_raw is None:
                    return dict(
                        self.hold_target_raw
                    )

                return dict(
                    self.policy_target_raw
                )

        return self._current_pose_target(
            now
        )

    def command_sit(self):
        with self.lock:
            if not self.enabled:
                raise RuntimeError(
                    "Motors are disabled"
                )

            self.policy_pending_at = None

            q_start = (
                self._current_commanded_target(
                    time.perf_counter()
                )
            )

            q_goal = (
                self._pose_goal_raw(
                    "SIT"
                )
            )

            self.mode = "SIT"

            self.transition = Transition(
                start_time=time.perf_counter(),
                duration=self.transition_seconds,
                q_start_raw=q_start,
                q_goal_raw=q_goal,
                destination="SIT",
                auto_policy_after=False,
            )

    def command_stand_then_policy(self):
        if self.imu is None:
            raise RuntimeError(
                "Automatic policy takeover needs "
                "--imu-provider"
            )

        with self.lock:
            if not self.enabled:
                raise RuntimeError(
                    "Motors are disabled"
                )

            self.policy_pending_at = None

            q_start = (
                self._current_commanded_target(
                    time.perf_counter()
                )
            )

            q_goal = (
                self._pose_goal_raw(
                    "STAND"
                )
            )

            self.mode = "STAND"

            self.transition = Transition(
                start_time=time.perf_counter(),
                duration=self.transition_seconds,
                q_start_raw=q_start,
                q_goal_raw=q_goal,
                destination="STAND",
                auto_policy_after=True,
            )

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def set_command(
        self,
        vx,
        vy,
        yaw,
    ):
        command = np.array(
            [vx, vy, yaw],
            dtype=np.float32,
        )

        if not np.all(
            np.isfinite(command)
        ):
            raise ValueError(
                "Command contains non-finite values"
            )

        with self.lock:
            self.command = command

    def _enter_policy(
        self,
        now,
    ):
        with self.lock:
            if not self.enabled:
                return

            self.mode = "POLICY"
            self.policy_pending_at = None
            self.previous_action[:] = 0.0
            self.next_policy_tick = now

            if self.hold_target_raw is not None:
                self.policy_target_raw = dict(
                    self.hold_target_raw
                )

        print("RL POLICY TAKEOVER")

    def _update_policy_target(
        self,
    ):
        if self.imu is None:
            raise RuntimeError(
                "Policy mode needs real IMU data"
            )

        raw_state = (
            self.read_all_raw_state()
        )

        (
            q_joint_dict,
            qd_joint_dict,
        ) = self._raw_to_joint_state(
            raw_state
        )

        q_joint = (
            self.policy
            .ordered_joint_state(
                q_joint_dict
            )
        )

        qd_joint = (
            self.policy
            .ordered_joint_state(
                qd_joint_dict
            )
        )

        (
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
        ) = self.imu.read()

        with self.lock:
            command = self.command.copy()
            previous_action = (
                self.previous_action.copy()
            )

        obs = (
            self.policy
            .build_observation(
                base_lin_vel=base_lin_vel,
                base_ang_vel=base_ang_vel,
                projected_gravity=projected_gravity,
                command=command,
                q_joint=q_joint,
                qd_joint=qd_joint,
                previous_action=previous_action,
            )
        )

        action = (
            self.policy.infer(obs)
        )

        q_target_joint = (
            self.policy
            .action_to_joint_target(
                action
            )
        )

        q_target_raw = (
            self._joint_target_to_raw(
                q_target_joint
            )
        )

        with self.lock:
            self.previous_action = (
                action.copy()
            )

            self.policy_target_raw = (
                q_target_raw
            )

    # ------------------------------------------------------------------
    # MIT send
    # ------------------------------------------------------------------

    def _active_gains(self):
        with self.lock:
            mode = self.mode

        if mode == "POLICY":
            return (
                self.policy.policy_gains_by_joint
                or self.policy.policy_gains
            )

        return self.pose_gains

    def _send_bus(
        self,
        bus_name,
        q_target_raw,
        gains,
    ):
        bus = (
            self.bus0
            if bus_name == CAN0
            else self.bus1
        )

        with self.bus_locks[bus_name]:
            for name, cfg in JOINTS.items():
                if cfg["bus"] != bus_name:
                    continue

                group = cfg["group"]
                joint_gains = (
                    gains[name]
                    if name in gains
                    else gains[group]
                )

                bus.write_operation_frame(
                    motor=name,
                    position=float(
                        q_target_raw[name]
                    ),
                    velocity=0.0,
                    kp=float(
                        joint_gains["kp"]
                    ),
                    kd=float(
                        joint_gains["kd"]
                    ),
                    torque=0.0,
                )

    # ------------------------------------------------------------------
    # Main motor loop
    # ------------------------------------------------------------------

    def _control_loop(self):
        period = (
            1.0
            / self.motor_hz
        )

        next_motor_tick = (
            time.perf_counter()
        )

        try:
            while not self.stop_event.is_set():
                now = time.perf_counter()

                with self.lock:
                    enabled = self.enabled
                    mode = self.mode

                if enabled:
                    if mode != "POLICY":
                        q_target_raw = (
                            self._current_pose_target(
                                now
                            )
                        )

                        with self.lock:
                            pending_at = (
                                self.policy_pending_at
                            )

                        if (
                            pending_at is not None
                            and now >= pending_at
                        ):
                            self._enter_policy(
                                now
                            )

                    with self.lock:
                        mode = self.mode
                        next_policy_tick = (
                            self.next_policy_tick
                        )

                    if mode == "POLICY":
                        if (
                            next_policy_tick is None
                            or now >= next_policy_tick
                        ):
                            self._update_policy_target()

                            with self.lock:
                                self.next_policy_tick = (
                                    now
                                    + self.policy.policy_dt
                                )

                        with self.lock:
                            q_target_raw = dict(
                                self.policy_target_raw
                            )

                    gains = self._active_gains()

                    f0 = self.executor.submit(
                        self._send_bus,
                        CAN0,
                        q_target_raw,
                        gains,
                    )

                    f1 = self.executor.submit(
                        self._send_bus,
                        CAN1,
                        q_target_raw,
                        gains,
                    )

                    f0.result()
                    f1.result()

                next_motor_tick += period

                wait_time = (
                    next_motor_tick
                    - time.perf_counter()
                )

                if wait_time > 0.0:
                    self.stop_event.wait(
                        wait_time
                    )
                else:
                    next_motor_tick = (
                        time.perf_counter()
                    )

        except Exception as exc:
            print(
                f"\nCONTROL ERROR: {exc}",
                file=sys.stderr,
            )

            try:
                self.disable_all()
            except Exception:
                pass

            self.stop_event.set()

    def close(self):
        self.stop_event.set()

        try:
            self.disable_all()
        except Exception:
            pass

        if (
            self.control_thread is not None
            and self.control_thread.is_alive()
            and threading.current_thread()
            is not self.control_thread
        ):
            self.control_thread.join(
                timeout=1.0
            )

        if self.connected:
            try:
                self.bus0.disconnect(
                    disable_torque=False
                )
            except Exception:
                pass

            try:
                self.bus1.disconnect(
                    disable_torque=False
                )
            except Exception:
                pass

        self.executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

        self.connected = False


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Grallator SIT -> STAND -> "
            "automatic RL policy takeover"
        )
    )

    parser.add_argument(
        "--policy-spec",
        required=True,
    )

    parser.add_argument(
        "--policy-path",
        default=None,
        help="Override policy_path from the JSON spec.",
    )

    parser.add_argument(
        "--policy-check-only",
        action="store_true",
        help="Exercise the actor without opening CAN or enabling motors.",
    )

    parser.add_argument(
        "--imu-provider",
        default=None,
    )

    parser.add_argument(
        "--can0",
        default="slcan0",
    )

    parser.add_argument(
        "--can1",
        default="slcan1",
    )

    parser.add_argument(
        "--motor-hz",
        type=float,
        default=DEFAULT_MOTOR_HZ,
    )

    parser.add_argument(
        "--transition",
        type=float,
        default=DEFAULT_TRANSITION_SECONDS,
    )

    parser.add_argument(
        "--takeover-delay",
        type=float,
        default=DEFAULT_TAKEOVER_DELAY_SECONDS,
    )

    parser.add_argument(
        "--hip-kp",
        type=float,
        default=DEFAULT_POSE_GAINS["hip"]["kp"],
    )

    parser.add_argument(
        "--hip-kd",
        type=float,
        default=DEFAULT_POSE_GAINS["hip"]["kd"],
    )

    parser.add_argument(
        "--thigh-kp",
        type=float,
        default=DEFAULT_POSE_GAINS["thigh"]["kp"],
    )

    parser.add_argument(
        "--thigh-kd",
        type=float,
        default=DEFAULT_POSE_GAINS["thigh"]["kd"],
    )

    parser.add_argument(
        "--calf-kp",
        type=float,
        default=DEFAULT_POSE_GAINS["calf"]["kp"],
    )

    parser.add_argument(
        "--calf-kd",
        type=float,
        default=DEFAULT_POSE_GAINS["calf"]["kd"],
    )

    parser.add_argument(
        "--vx",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--vy",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--yaw",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not 1.0 <= args.motor_hz <= 200.0:
        raise SystemExit(
            "--motor-hz must be within 1..200"
        )

    if not 0.5 <= args.transition <= 30.0:
        raise SystemExit(
            "--transition must be within 0.5..30"
        )

    if not 0.0 <= args.takeover_delay <= 10.0:
        raise SystemExit(
            "--takeover-delay must be within 0..10"
        )

    policy = TorchPolicy(
        args.policy_spec,
        model_path_override=args.policy_path,
    )

    if args.policy_check_only:
        nominal_kwargs = {
            "base_lin_vel": [0.0, 0.0, 0.0],
            "base_ang_vel": [0.0, 0.0, 0.0],
            "projected_gravity": [0.0, 0.0, -1.0],
            "q_joint": np.zeros(12, dtype=np.float32),
            "qd_joint": np.zeros(12, dtype=np.float32),
            "previous_action": np.zeros(12, dtype=np.float32),
        }
        stand_obs = policy.build_observation(
            command=[0.0, 0.0, 0.0],
            **nominal_kwargs,
        )
        command_obs = policy.build_observation(
            command=[args.vx, args.vy, args.yaw],
            **nominal_kwargs,
        )
        stand_action = policy.infer(stand_obs)
        command_action = policy.infer(command_obs)
        print(f"Policy: {policy.model_path}")
        print(f"Format: {policy.policy_format}")
        print(f"Observation/action: {policy.obs_dim} -> {stand_action.size}")
        print("Joint order:", ", ".join(policy.joint_order))
        print("Nominal level-stand action:", stand_action)
        print(
            f"Command [{args.vx:+.3f}, {args.vy:+.3f}, {args.yaw:+.3f}] action:",
            command_action,
        )
        print("POLICY CHECK PASSED; no CAN interface was opened.")
        return

    if policy.spec.get("hardware_test_permitted", False) is not True:
        raise SystemExit(
            "Hardware motion is blocked by this policy spec. Run "
            "--policy-check-only; verify motor directions, offsets, physical "
            "limits, real Xsens provider, and safety watchdogs first."
        )

    imu = (
        ExternalIMU(
            args.imu_provider
        )
        if args.imu_provider
        else None
    )

    pose_gains = {
        "hip": {
            "kp": args.hip_kp,
            "kd": args.hip_kd,
        },
        "thigh": {
            "kp": args.thigh_kp,
            "kd": args.thigh_kd,
        },
        "calf": {
            "kp": args.calf_kp,
            "kd": args.calf_kd,
        },
    }

    controller = GrallatorRLController(
        can0=args.can0,
        can1=args.can1,
        motor_hz=args.motor_hz,
        transition_seconds=args.transition,
        takeover_delay=args.takeover_delay,
        pose_gains=pose_gains,
        policy=policy,
        imu=imu,
    )

    controller.set_command(
        args.vx,
        args.vy,
        args.yaw,
    )

    def stop_handler(
        _sig,
        _frame,
    ):
        controller.close()
        raise SystemExit(0)

    signal.signal(
        signal.SIGINT,
        stop_handler,
    )

    signal.signal(
        signal.SIGTERM,
        stop_handler,
    )

    try:
        print(
            "Policy:",
            policy.model_path,
        )

        print(
            "Observation dimension:",
            policy.obs_dim,
        )

        print(
            "Policy rate:",
            policy.policy_hz,
            "Hz",
        )

        print(
            "Motor rate:",
            args.motor_hz,
            "Hz",
        )

        print(
            "Position feedback:",
            POSITION_PARAMETER_NAME,
        )

        print(
            "Velocity feedback:",
            VELOCITY_PARAMETER_NAME,
        )

        print()

        print(
            "Connecting slcan0 + slcan1..."
        )

        controller.connect()

        controller.read_all_raw_state()

        print(
            "All 12 RS04 q/qdot reads succeeded."
        )

        print()
        print("Commands:")
        print(
            "  e              "
            "capture CURRENT stand pose + enable"
        )
        print(
            "  s              SIT"
        )
        print(
            "  w              "
            "STAND -> automatic RL takeover"
        )
        print(
            "  v VX VY YAW    "
            "change policy command"
        )
        print(
            "  d              disable"
        )
        print(
            "  q              disable + quit"
        )
        print()

        if imu is None:
            print(
                "NOTE: SIT/HOLD can run, but W will "
                "refuse automatic RL takeover until "
                "--imu-provider is supplied."
            )
            print()

        while True:
            text = input("> ").strip()

            if not text:
                continue

            parts = text.split()
            command = parts[0].lower()

            try:
                if command == "e":
                    controller.enable_and_capture_stand()
                    print(
                        "MIT enabled; stand reference captured."
                    )

                elif command == "s":
                    controller.command_sit()
                    print(
                        "SIT commanded."
                    )

                elif command == "w":
                    controller.command_stand_then_policy()
                    print(
                        "STAND commanded; RL will take over "
                        "automatically after stand completes."
                    )

                elif command == "v":
                    if len(parts) != 4:
                        print(
                            "Usage: v VX VY YAW"
                        )
                        continue

                    vx, vy, yaw = map(
                        float,
                        parts[1:],
                    )

                    controller.set_command(
                        vx,
                        vy,
                        yaw,
                    )

                    print(
                        "command = "
                        f"[{vx:+.3f}, {vy:+.3f}, {yaw:+.3f}]"
                    )

                elif command == "d":
                    controller.disable_all()
                    print(
                        "All motors disabled."
                    )

                elif command == "q":
                    break

                else:
                    print(
                        "Use e / s / w / "
                        "v VX VY YAW / d / q"
                    )

            except Exception as exc:
                print(
                    f"Command failed: {exc}"
                )

    finally:
        controller.close()

        print(
            "All motors disabled. "
            "CAN connections closed."
        )


if __name__ == "__main__":
    main()
