import unittest
from pathlib import Path

import numpy as np

from pace_modeling.constants import JOINT_ORDER
from pace_modeling.controller import (
    final_command_target,
    load_yaml,
    verify_deployment_contract,
)
from pace_modeling.trajectory import build_trajectory


ROOT = Path(__file__).resolve().parents[1]


class TrajectoryTests(unittest.TestCase):
    def test_exact_20_second_sample_count(self):
        spec = load_yaml(ROOT / "trajectories" / "dataset_A_chirp_20s.yaml")
        samples = build_trajectory(spec, np.zeros(12), expected_hz=50.0)
        self.assertEqual(len(samples), 1000)
        self.assertAlmostEqual(samples[1].time_s - samples[0].time_s, 0.02)

    def test_required_joint_order(self):
        self.assertEqual(JOINT_ORDER[:4], [
            "FL_hip_joint", "FR_hip_joint", "BL_hip_joint", "BR_hip_joint"
        ])
        self.assertEqual(JOINT_ORDER[-4:], [
            "FL_calf_joint", "FR_calf_joint", "BL_calf_joint", "BR_calf_joint"
        ])

    def test_final_target_is_hard_and_rate_limited(self):
        config = load_yaml(ROOT / "config" / "pace_config.yaml")
        requested = np.full(12, 10.0)
        previous = np.zeros(12)
        actual = np.zeros(12)
        velocity = np.zeros(12)
        sent = final_command_target(
            requested, previous, actual, velocity, config, 0.02
        )
        self.assertAlmostEqual(sent[0], 0.008, places=7)
        self.assertAlmostEqual(sent[4], 0.012, places=7)
        self.assertAlmostEqual(sent[8], 0.012, places=7)
        self.assertEqual(sent[9], 0.0)  # right calf positive request hits max=0

    def test_pace_calibration_matches_deployment(self):
        config = load_yaml(ROOT / "config" / "pace_config.yaml")
        verify_deployment_contract(config, ROOT.parent / "GRALLATOR_DEPLOY")


if __name__ == "__main__":
    unittest.main()
