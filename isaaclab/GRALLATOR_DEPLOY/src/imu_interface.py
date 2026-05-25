#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import time
from dataclasses import dataclass
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


def load_imu_config():
    return load_yaml(ROOT / "config" / "imu.yaml")["imu"]


def load_external_module(module_path, module_name):
    module_path = Path(module_path)
    if not module_path.exists():
        raise FileNotFoundError(f"IMU helper module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import IMU helper module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class ImuReading:
    base_ang_vel_b: np.ndarray
    projected_gravity_b: np.ndarray
    base_lin_vel_b: np.ndarray
    timestamp: float


def as_vector3(value, name):
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have exactly 3 values")
    return arr


def normalize_vector(vec, fallback):
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def reorder_quaternion(quat, order):
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape != (4,):
        raise ValueError("quaternion must have exactly 4 values")

    order = str(order).lower()
    if order == "wxyz":
        w, x, y, z = quat
    elif order == "xyzw":
        x, y, z, w = quat
    else:
        raise ValueError(f"Unsupported quaternion_order: {order}")

    norm = math.sqrt(float(w * w + x * x + y * y + z * z))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([w / norm, x / norm, y / norm, z / norm], dtype=np.float32)


def quat_to_rotation_matrix(quat_wxyz):
    w, x, y, z = [float(v) for v in quat_wxyz]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def projected_gravity_from_quaternion(quat, order="wxyz", frame="world_from_body"):
    quat_wxyz = reorder_quaternion(quat, order)
    rotation = quat_to_rotation_matrix(quat_wxyz)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    frame = str(frame).lower()
    if frame == "world_from_body":
        gravity_body = rotation.T @ gravity_world
    elif frame == "body_from_world":
        gravity_body = rotation @ gravity_world
    else:
        raise ValueError(f"Unsupported quaternion_frame: {frame}")

    return normalize_vector(gravity_body, fallback=[0.0, 0.0, -1.0])


def first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


class FakeImuSensor:
    source_name = "fake"
    required = False

    def __init__(self, cfg=None):
        self.stale_timeout = float((cfg or {}).get("stale_timeout", 0.25))

    def open(self):
        return self

    def close(self):
        return None

    def read(self):
        return ImuReading(
            base_ang_vel_b=np.zeros(3, dtype=np.float32),
            projected_gravity_b=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            base_lin_vel_b=np.zeros(3, dtype=np.float32),
            timestamp=time.monotonic(),
        )


class XsensBinaryImuSensor:
    source_name = "xsens"

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.required = bool(self.cfg.get("require_live", True))
        self.port = self.cfg.get("port", "/dev/ttyUSB0")
        self.baud = int(self.cfg.get("baud", 115200))
        self.timeout = float(self.cfg.get("timeout", 0.001))
        self.stale_timeout = float(self.cfg.get("stale_timeout", 0.25))
        self.force_zero_base_linear_velocity = bool(
            self.cfg.get("force_zero_base_linear_velocity", True)
        )
        self.reader_path = self.cfg.get(
            "xsens_reader_path",
            "/home/poornaprasad/Work/sensor/IMU/get_data.py",
        )
        self.sensor_to_base_rotation = np.asarray(
            self.cfg.get("sensor_to_base_rotation", np.eye(3)),
            dtype=np.float32,
        )
        if self.sensor_to_base_rotation.shape != (3, 3):
            raise ValueError("sensor_to_base_rotation must be a 3x3 matrix")
        self.flip_gravity = bool(self.cfg.get("flip_gravity", True))

        self.helper = None
        self.ser = None
        self.buffer = bytearray()
        self.latest_quaternion_ws = None
        self.latest_gyro_sensor = None
        self.last_packet_time = None

    def open(self):
        import serial

        self.helper = load_external_module(self.reader_path, "grallator_external_imu_get_data")
        if hasattr(self.helper, "is_port_available") and not self.helper.is_port_available(self.port):
            raise FileNotFoundError(f"IMU port not found: {self.port}")

        self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        time.sleep(0.1)
        return self

    def close(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def _read_packets(self):
        waiting = int(getattr(self.ser, "in_waiting", 0))
        read_size = max(512, waiting) if waiting > 0 else 512
        data = self.ser.read(read_size)
        if data:
            self.buffer.extend(data)

        updated = False
        for _, mid, payload in self.helper.extract_xbus_packets(self.buffer):
            if mid != self.helper.MID_MTDATA2:
                continue

            parsed = self.helper.parse_mtdata2(payload)
            if "quaternion_ws" in parsed:
                self.latest_quaternion_ws = self.helper.normalize_quat(parsed["quaternion_ws"])
                updated = True
            if "gyro_sensor" in parsed:
                self.latest_gyro_sensor = np.asarray(parsed["gyro_sensor"], dtype=np.float32)
                updated = True

        if updated:
            self.last_packet_time = time.monotonic()

    def read(self):
        if self.ser is None:
            raise RuntimeError("IMU serial port is not open")

        self._read_packets()
        if self.latest_quaternion_ws is None or self.latest_gyro_sensor is None:
            return None

        projected_gravity = self.helper.compute_projected_gravity_from_quaternion(
            quaternion_ws=self.latest_quaternion_ws,
            sensor_to_base_rotation=self.sensor_to_base_rotation,
            flip_gravity=self.flip_gravity,
        )
        base_ang_vel = self.sensor_to_base_rotation @ self.latest_gyro_sensor

        # Training used base linear velocity [0, 0, 0]. Keep it fixed.
        base_lin_vel = np.zeros(3, dtype=np.float32)

        return ImuReading(
            base_ang_vel_b=np.asarray(base_ang_vel, dtype=np.float32),
            projected_gravity_b=normalize_vector(
                projected_gravity,
                fallback=[0.0, 0.0, -1.0],
            ),
            base_lin_vel_b=base_lin_vel,
            timestamp=self.last_packet_time or time.monotonic(),
        )


class SerialImuSensor:
    def __init__(self, cfg, source_name):
        self.cfg = dict(cfg)
        self.source_name = source_name
        self.required = bool(self.cfg.get("require_live", True))
        self.port = self.cfg.get("port", "/dev/ttyUSB1")
        self.baud = int(self.cfg.get("baud", 115200))
        self.timeout = float(self.cfg.get("timeout", 0.001))
        self.stale_timeout = float(self.cfg.get("stale_timeout", 0.25))
        self.gyro_units = str(self.cfg.get("gyro_units", "rad_s")).lower()
        self.gyro_axis_signs = as_vector3(self.cfg.get("gyro_axis_signs", [1.0, 1.0, 1.0]), "gyro_axis_signs")
        self.gravity_axis_signs = as_vector3(
            self.cfg.get("gravity_axis_signs", [1.0, 1.0, 1.0]),
            "gravity_axis_signs",
        )
        self.csv_fields = list(self.cfg.get("csv_fields", ["gx", "gy", "gz", "qw", "qx", "qy", "qz"]))
        self.quaternion_order = self.cfg.get("quaternion_order", "wxyz")
        self.quaternion_frame = self.cfg.get("quaternion_frame", "world_from_body")
        self.force_zero_base_linear_velocity = bool(
            self.cfg.get("force_zero_base_linear_velocity", True)
        )
        self.ser = None

    def open(self):
        import serial

        self.ser = serial.Serial(
            self.port,
            self.baud,
            timeout=self.timeout,
            write_timeout=self.timeout,
        )
        time.sleep(0.05)
        return self

    def close(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def scale_gyro(self, gyro):
        gyro = as_vector3(gyro, "gyro") * self.gyro_axis_signs
        if self.gyro_units in ("deg_s", "deg/s", "dps"):
            gyro = np.deg2rad(gyro)
        elif self.gyro_units not in ("rad_s", "rad/s", "rps"):
            raise ValueError(f"Unsupported gyro_units: {self.gyro_units}")
        return gyro.astype(np.float32)

    def parse_json_line(self, line):
        msg = json.loads(line)
        gyro = first_present(msg, ["base_ang_vel", "ang_vel", "angular_velocity", "gyro", "gyr"])
        if gyro is None:
            raise ValueError("JSON IMU line missing gyro/angular velocity")

        projected_gravity = first_present(
            msg,
            ["projected_gravity", "projected_gravity_b", "gravity", "gravity_b", "grav"],
        )
        if projected_gravity is not None:
            projected_gravity = normalize_vector(
                as_vector3(projected_gravity, "projected_gravity") * self.gravity_axis_signs,
                fallback=[0.0, 0.0, -1.0],
            )
        else:
            quat = first_present(msg, ["quat", "quaternion", "orientation"])
            if quat is None:
                raise ValueError("JSON IMU line needs projected_gravity or quaternion")
            projected_gravity = projected_gravity_from_quaternion(
                quat,
                order=self.quaternion_order,
                frame=self.quaternion_frame,
            )
            projected_gravity = normalize_vector(
                projected_gravity * self.gravity_axis_signs,
                fallback=[0.0, 0.0, -1.0],
            )

        base_lin_vel = first_present(msg, ["base_lin_vel", "lin_vel", "linear_velocity"])
        if self.force_zero_base_linear_velocity or base_lin_vel is None:
            base_lin_vel = np.zeros(3, dtype=np.float32)
        else:
            base_lin_vel = as_vector3(base_lin_vel, "base_lin_vel")

        return ImuReading(
            base_ang_vel_b=self.scale_gyro(gyro),
            projected_gravity_b=projected_gravity,
            base_lin_vel_b=base_lin_vel.astype(np.float32),
            timestamp=time.monotonic(),
        )

    def parse_csv_line(self, line):
        values = [float(part.strip()) for part in line.split(",")]
        if len(values) != len(self.csv_fields):
            raise ValueError(
                f"CSV IMU line has {len(values)} values, expected {len(self.csv_fields)}"
            )
        msg = dict(zip(self.csv_fields, values))

        gyro = [msg["gx"], msg["gy"], msg["gz"]]
        if all(key in msg for key in ("grav_x", "grav_y", "grav_z")):
            projected_gravity = normalize_vector(
                np.array([msg["grav_x"], msg["grav_y"], msg["grav_z"]], dtype=np.float32)
                * self.gravity_axis_signs,
                fallback=[0.0, 0.0, -1.0],
            )
        else:
            quat_keys = ("qw", "qx", "qy", "qz")
            if not all(key in msg for key in quat_keys):
                raise ValueError("CSV IMU line needs qw,qx,qy,qz or grav_x,grav_y,grav_z")
            projected_gravity = projected_gravity_from_quaternion(
                [msg["qw"], msg["qx"], msg["qy"], msg["qz"]],
                order="wxyz",
                frame=self.quaternion_frame,
            )
            projected_gravity = normalize_vector(
                projected_gravity * self.gravity_axis_signs,
                fallback=[0.0, 0.0, -1.0],
            )

        base_lin_vel = np.zeros(3, dtype=np.float32)
        if not self.force_zero_base_linear_velocity and all(key in msg for key in ("vx", "vy", "vz")):
            base_lin_vel = np.array([msg["vx"], msg["vy"], msg["vz"]], dtype=np.float32)

        return ImuReading(
            base_ang_vel_b=self.scale_gyro(gyro),
            projected_gravity_b=projected_gravity,
            base_lin_vel_b=base_lin_vel,
            timestamp=time.monotonic(),
        )

    def parse_line(self, line):
        line = line.strip()
        if not line:
            return None
        if self.source_name == "serial_json":
            return self.parse_json_line(line)
        if self.source_name == "serial_csv":
            return self.parse_csv_line(line)
        raise ValueError(f"Unsupported serial IMU source: {self.source_name}")

    def read(self):
        if self.ser is None:
            raise RuntimeError("IMU serial port is not open")

        latest = None
        reads = max(1, int(getattr(self.ser, "in_waiting", 0) > 0) * 32)
        for _ in range(reads):
            raw = self.ser.readline()
            if not raw:
                break
            try:
                line = raw.decode("utf-8", errors="ignore")
                reading = self.parse_line(line)
            except Exception:
                continue
            if reading is not None:
                latest = reading

        return latest


def create_imu_sensor(source="auto", port=None, baud=None):
    cfg = load_imu_config()
    if source == "auto":
        source = cfg.get("source", "fake")

    source = str(source).replace("-", "_").lower()
    if port is not None:
        cfg["port"] = port
    if baud is not None:
        cfg["baud"] = int(baud)

    if source in ("none", "off"):
        return None
    if source == "fake":
        return FakeImuSensor(cfg).open()
    if source in ("xsens", "xsens_binary", "mtdata2"):
        return XsensBinaryImuSensor(cfg).open()
    if source in ("serial_json", "serial_csv"):
        return SerialImuSensor(cfg, source_name=source).open()

    raise ValueError(f"Unknown IMU source: {source}")


def main():
    cfg = load_imu_config()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["auto", "fake", "none", "xsens", "serial-json", "serial-csv"],
        default="auto",
    )
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--hz", "--rate", type=float, default=20.0)
    args = parser.parse_args()

    sensor = create_imu_sensor(source=args.source, port=args.port, baud=args.baud)
    if sensor is None:
        print("IMU source disabled.")
        return 0

    dt = 1.0 / max(1e-6, args.hz)
    steps = int(args.seconds * args.hz)
    print("IMU source:", sensor.source_name)
    print("IMU port:", getattr(sensor, "port", "none"))
    print("IMU baud:", getattr(sensor, "baud", "none"))
    print("Config source:", cfg.get("source", "fake"))

    try:
        for step in range(steps):
            reading = sensor.read()
            if reading is None:
                print(f"step={step:04d} imu=no_new_data")
            else:
                print(
                    f"step={step:04d} "
                    f"gyro={reading.base_ang_vel_b} "
                    f"gravity={reading.projected_gravity_b} "
                    f"lin_vel={reading.base_lin_vel_b}"
                )
            time.sleep(dt)
    finally:
        sensor.close()

    return 0


if __name__ == "__main__":
    main()
