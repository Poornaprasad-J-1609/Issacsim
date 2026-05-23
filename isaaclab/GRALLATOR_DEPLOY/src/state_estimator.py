#!/usr/bin/env python3
import numpy as np


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

    def __init__(self, q_initial):
        self.q_current = np.asarray(q_initial, dtype=np.float32).copy()
        self.qd_current = np.zeros(12, dtype=np.float32)

        self.base_lin_vel_b = np.zeros(3, dtype=np.float32)
        self.base_ang_vel_b = np.zeros(3, dtype=np.float32)
        self.projected_gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def read(self):
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
