#!/usr/bin/env python3
import time
import numpy as np

from motor_command_layer import decode_mit_feedback_frame


class FakeStateEstimator:
    """
    Dry-run estimator.

    Replace this later with:
      - q_current from RobStride encoder feedback
      - qd_current from RobStride velocity feedback
      - base_ang_vel_b from IMU gyro
      - projected_gravity_b from IMU orientation
      - base_lin_vel_b from velocity estimator
    """

    def __init__(self, q_initial, imu_sensor=None):
        self.q_current = np.asarray(q_initial, dtype=np.float32).copy()
        self.qd_current = np.zeros(12, dtype=np.float32)

        self.base_lin_vel_b = np.zeros(3, dtype=np.float32)
        self.base_ang_vel_b = np.zeros(3, dtype=np.float32)
        self.projected_gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.imu_sensor = imu_sensor
        self.last_imu_timestamp = None
        self.last_imu_update_time = None

    def refresh_imu(self):
        if self.imu_sensor is None:
            return False

        reading = self.imu_sensor.read()
        if reading is None:
            return False

        # The deployed policy was trained with base linear velocity fixed at zero.
        # Keep this observation term identical to training even if an IMU helper
        # can estimate velocity from acceleration.
        self.base_lin_vel_b = np.zeros(3, dtype=np.float32)
        self.base_ang_vel_b = np.asarray(reading.base_ang_vel_b, dtype=np.float32).copy()
        self.projected_gravity_b = np.asarray(reading.projected_gravity_b, dtype=np.float32).copy()
        self.last_imu_timestamp = float(reading.timestamp)
        self.last_imu_update_time = time.monotonic()
        return True

    def imu_required(self):
        return bool(getattr(self.imu_sensor, "required", False))

    def imu_age(self):
        if self.last_imu_update_time is None:
            return None
        return time.monotonic() - self.last_imu_update_time

    def imu_stale(self):
        if not self.imu_required():
            return False
        age = self.imu_age()
        stale_timeout = float(getattr(self.imu_sensor, "stale_timeout", 0.25))
        return age is None or age > stale_timeout

    def imu_status(self):
        if self.imu_sensor is None:
            return "none"
        if not self.imu_required():
            return getattr(self.imu_sensor, "source_name", "imu")
        age = self.imu_age()
        if age is None:
            return f"{getattr(self.imu_sensor, 'source_name', 'imu')}:missing"
        if self.imu_stale():
            return f"{getattr(self.imu_sensor, 'source_name', 'imu')}:stale"
        return f"{getattr(self.imu_sensor, 'source_name', 'imu')}:live"

    def read(self):
        self.refresh_imu()
        return (
            self.q_current.copy(),
            self.qd_current.copy(),
            self.base_lin_vel_b.copy(),
            self.base_ang_vel_b.copy(),
            self.projected_gravity_b.copy(),
        )

    def dry_update_as_if_robot_followed(self, q_target, dt):
        q_target = np.asarray(q_target, dtype=np.float32)
        self.qd_current = (q_target - self.q_current) / float(dt)
        self.q_current = q_target.copy()


class MitFeedbackStateEstimator(FakeStateEstimator):
    """
    State estimator backed by RobStride/CyberGear MIT feedback frames.

    Motor feedback position is converted back into policy joint coordinates by
    subtracting the configured joint offset used when commands are sent.
    """

    def __init__(self, q_initial, policy_order, motor_ids, motor_layer, bus, imu_sensor=None):
        super().__init__(q_initial=q_initial, imu_sensor=imu_sensor)

        self.policy_order = list(policy_order)
        self.motor_layer = motor_layer
        self.bus = bus

        self.joint_index_by_name = {
            name: index for index, name in enumerate(self.policy_order)
        }
        self.joint_name_by_motor_id = {
            int(motor_id): joint_name
            for joint_name, motor_id in motor_ids.items()
        }
        self.last_feedback_by_joint = {}
        self.last_feedback_count = 0

    def update_from_frames(self, frames):
        count = 0

        for frame in frames:
            feedback = decode_mit_feedback_frame(
                frame.can_id,
                frame.data,
                self.motor_layer.proto,
            )
            if feedback is None:
                continue

            joint_name = self.joint_name_by_motor_id.get(int(feedback["motor_id"]))
            if joint_name is None:
                continue

            index = self.joint_index_by_name[joint_name]
            offset = float(self.motor_layer.joint_offsets[joint_name])

            self.q_current[index] = float(feedback["position"]) - offset
            self.qd_current[index] = float(feedback["velocity"])
            self.last_feedback_by_joint[joint_name] = feedback
            count += 1

        self.last_feedback_count = count
        return count

    def refresh_from_bus(self, timeout=0.0):
        frames = self.bus.read_available_frames(timeout=timeout)
        return self.update_from_frames(frames)

    def read(self):
        self.refresh_from_bus(timeout=0.0)
        return super().read()

    def dry_update_as_if_robot_followed(self, q_target, dt):
        # Live feedback owns q_current/qd_current. Keep the method for the
        # controller loop interface, but do not overwrite measured state.
        return None

    def has_all_joint_feedback(self):
        return len(self.last_feedback_by_joint) >= len(self.policy_order)
