#!/usr/bin/env python3
"""
Template adapter for your existing Xsens MTi-630 reader.

grallator_rl_takeover.py expects:

    class IMUProvider:
        def read(self):
            return {
                "base_ang_vel": [wx, wy, wz],
                "projected_gravity": [gx, gy, gz],
                "base_lin_vel": [vx, vy, vz],  # optional
            }

IMPORTANT:
The axis/sign/frame convention must be EXACTLY the same as the RL simulation.
"""

import numpy as np


class IMUProvider:
    def __init__(self):
        # Initialize your existing Xsens UDP receiver here.
        raise NotImplementedError(
            "Connect this class to the existing Xsens MTi-630 code."
        )

    def read(self):
        # Replace this with your actual sensor data.
        #
        # return {
        #     "base_ang_vel": np.array(
        #         [wx, wy, wz],
        #         dtype=np.float32,
        #     ),
        #     "projected_gravity": np.array(
        #         [gx, gy, gz],
        #         dtype=np.float32,
        #     ),
        #     "base_lin_vel": np.zeros(
        #         3,
        #         dtype=np.float32,
        #     ),
        # }
        raise NotImplementedError
