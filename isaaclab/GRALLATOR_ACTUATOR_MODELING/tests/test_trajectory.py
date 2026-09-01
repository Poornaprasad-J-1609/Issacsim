import unittest
from pathlib import Path

import numpy as np

from pace_modeling.constants import (
    JOINT_ORDER,
    PACE_EXPORT_ORDER,
    PACE_EXPORT_INDICES,
    to_pace_order,
)
from pace_modeling.controller import (
    final_command_target,
    load_yaml,
    validate_requested_trajectory,
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

    def test_slow_chirp_contract_and_safe_ranges(self):
        config = load_yaml(ROOT / "config" / "pace_config.yaml")
        spec = load_yaml(
            ROOT / "trajectories" / "grallator_all_joints_slow_chirp.yaml"
        )
        initial = np.array([
            float(spec["dry_run_initial_q"][name]) for name in JOINT_ORDER
        ])
        samples = build_trajectory(spec, initial, expected_hz=50.0)
        self.assertEqual(len(samples), 1550)
        chirp = [sample for sample in samples
                 if sample.segment == "all_joints_slow_chirp"]
        self.assertEqual(len(chirp), 1000)
        self.assertAlmostEqual(chirp[0].instantaneous_frequency_hz, 0.1)
        self.assertGreater(chirp[-1].instantaneous_frequency_hz, 0.499)
        validate_requested_trajectory(config, spec, samples)
        requested = np.stack([sample.q_requested for sample in chirp])
        self.assertTrue(np.all(np.ptp(requested, axis=0) > 0.49))

    def test_minimum_jerk_reaches_target(self):
        target = {name: 0.1 for name in JOINT_ORDER}
        spec = {
            "control_hz": 50.0,
            "segments": [{
                "name": "move", "type": "minimum_jerk",
                "duration_s": 1.0, "target": target,
            }],
        }
        samples = build_trajectory(spec, np.zeros(12), expected_hz=50.0)
        self.assertEqual(len(samples), 50)
        np.testing.assert_allclose(samples[-1].q_requested, 0.1, atol=1.0e-12)
        expected_first_alpha = 10 * 0.02**3 - 15 * 0.02**4 + 6 * 0.02**5
        self.assertAlmostEqual(samples[0].q_requested[0], 0.1 * expected_first_alpha)

    def test_limiter_diagnostics_identify_each_stage(self):
        config = load_yaml(ROOT / "config" / "pace_config.yaml")
        requested = np.full(12, 10.0)
        sent, diagnostics = final_command_target(
            requested,
            np.zeros(12),
            np.zeros(12),
            np.zeros(12),
            config,
            0.02,
            return_diagnostics=True,
        )
        self.assertTrue(np.all(diagnostics["hard_limit_active"]))
        self.assertTrue(np.any(diagnostics["rate_limit_active"]))
        np.testing.assert_allclose(diagnostics["limiter_delta"], sent - requested)

    def test_explicit_pace_export_order(self):
        self.assertEqual(PACE_EXPORT_ORDER[:3], [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"
        ])
        internal = np.arange(12)
        exported = to_pace_order(internal)
        np.testing.assert_array_equal(
            exported,
            np.asarray([JOINT_ORDER.index(name) for name in PACE_EXPORT_ORDER]),
        )
        self.assertEqual(tuple(exported), PACE_EXPORT_INDICES)


if __name__ == "__main__":
    unittest.main()
