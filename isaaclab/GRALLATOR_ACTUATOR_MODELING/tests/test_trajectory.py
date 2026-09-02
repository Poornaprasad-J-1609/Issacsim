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
    next_control_deadline,
    validate_requested_trajectory,
    verify_deployment_contract,
)
from pace_modeling.trajectory import build_trajectory, compile_trajectory


ROOT = Path(__file__).resolve().parents[1]


class TrajectoryTests(unittest.TestCase):
    def test_exact_20_second_sample_count(self):
        spec = load_yaml(ROOT / "trajectories" / "dataset_A_chirp_20s.yaml")
        samples = build_trajectory(spec, np.zeros(12), expected_hz=200.0)
        self.assertEqual(len(samples), 4000)
        self.assertAlmostEqual(samples[1].time_s - samples[0].time_s, 0.005)

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
            requested, previous, actual, velocity, config, 0.005
        )
        self.assertAlmostEqual(sent[0], 0.002, places=7)
        self.assertAlmostEqual(sent[4], 0.003, places=7)
        self.assertAlmostEqual(sent[8], 0.003, places=7)
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
        samples = build_trajectory(spec, initial, expected_hz=200.0)
        self.assertEqual(len(samples), 7400)
        chirp = [sample for sample in samples
                 if sample.segment == "all_joints_slow_chirp"]
        self.assertEqual(len(chirp), 4000)
        self.assertAlmostEqual(chirp[0].instantaneous_frequency_hz, 0.1)
        self.assertGreater(chirp[-1].instantaneous_frequency_hz, 0.499)
        validate_requested_trajectory(config, spec, samples)
        requested = np.stack([sample.q_requested for sample in chirp])
        self.assertTrue(np.all(np.ptp(requested, axis=0) > 0.49))

    def test_stage0_chirp_fits_configured_command_rates(self):
        config = load_yaml(ROOT / "config" / "pace_config.yaml")
        spec = load_yaml(
            ROOT / "trajectories" / "grallator_all_joints_stage0_chirp.yaml"
        )
        initial = np.array([
            float(spec["dry_run_initial_q"][name]) for name in JOINT_ORDER
        ])
        samples = build_trajectory(spec, initial, expected_hz=200.0)
        self.assertEqual(len(samples), 7400)
        validate_requested_trajectory(config, spec, samples)

        previous = initial.copy()
        maximum_delta = 0.0
        for sample in samples:
            sent = final_command_target(
                sample.q_requested,
                previous,
                previous,
                np.zeros(12),
                config,
                0.005,
            )
            maximum_delta = max(
                maximum_delta,
                float(np.max(np.abs(sent - sample.q_requested))),
            )
            previous = sent
        self.assertLess(maximum_delta, 1.0e-9)

    def test_entry_move_accepts_only_small_inward_initial_limit_violation(self):
        config = load_yaml(ROOT / "config" / "pace_config.yaml")
        spec = load_yaml(
            ROOT / "trajectories" / "grallator_all_joints_stage0_chirp.yaml"
        )
        initial = np.array([
            float(spec["dry_run_initial_q"][name]) for name in JOINT_ORDER
        ])
        fl_hip = JOINT_ORDER.index("FL_hip_joint")
        hard_min = float(config["joint_limits"]["FL_hip_joint"]["min"])
        tolerance = float(config["initial_pose_hard_limit_tolerance_rad"])
        initial[fl_hip] = hard_min - 0.006996
        samples = build_trajectory(spec, initial, expected_hz=200.0)
        validate_requested_trajectory(config, spec, samples)

        initial[fl_hip] = hard_min - tolerance - 0.001
        samples = build_trajectory(spec, initial, expected_hz=200.0)
        with self.assertRaisesRegex(ValueError, "hard limit plus tolerance"):
            validate_requested_trajectory(config, spec, samples)

    def test_minimum_jerk_reaches_target(self):
        target = {name: 0.1 for name in JOINT_ORDER}
        spec = {
            "control_hz": 200.0,
            "segments": [{
                "name": "move", "type": "minimum_jerk",
                "duration_s": 1.0, "target": target,
            }],
        }
        plan = compile_trajectory(spec, np.zeros(12), expected_hz=200.0)
        samples = build_trajectory(spec, np.zeros(12), expected_hz=200.0)
        self.assertEqual(len(samples), 200)
        np.testing.assert_allclose(samples[0].q_requested, 0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            plan.evaluate(1.0).q_requested, 0.1, atol=1.0e-12
        )

    def test_limiter_diagnostics_identify_each_stage(self):
        config = load_yaml(ROOT / "config" / "pace_config.yaml")
        requested = np.full(12, 10.0)
        sent, diagnostics = final_command_target(
            requested,
            np.zeros(12),
            np.zeros(12),
            np.zeros(12),
            config,
            0.005,
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

    def test_linear_chirp_uses_actual_elapsed_time_formula(self):
        spec = load_yaml(
            ROOT / "trajectories" / "grallator_all_joints_slow_chirp.yaml"
        )
        initial = np.array([
            float(spec["dry_run_initial_q"][name]) for name in JOINT_ORDER
        ])
        plan = compile_trajectory(spec, initial, expected_hz=200.0)
        elapsed = 12.0 + 7.123
        sample = plan.evaluate(elapsed)
        local_time = 7.123
        f_start, f_end, duration = 0.1, 0.5, 20.0
        slope = (f_end - f_start) / duration
        phase = 2.0 * np.pi * (
            f_start * local_time + 0.5 * slope * local_time**2
        )
        center = initial
        amplitude = np.array([
            float(spec["segments"][2]["amplitude"][name]) for name in JOINT_ORDER
        ])
        np.testing.assert_allclose(
            sample.q_requested, center + amplitude * np.sin(phase), atol=1.0e-12
        )
        self.assertAlmostEqual(
            sample.instantaneous_frequency_hz,
            f_start + slope * local_time,
        )

    def test_deadline_resynchronizes_after_one_long_pause(self):
        next_deadline, resynchronized = next_control_deadline(
            deadline=10.000,
            wake_time=10.020,
            dt=0.005,
            severe_lateness=0.005,
        )
        self.assertTrue(resynchronized)
        self.assertAlmostEqual(next_deadline, 10.025)

        following, resynchronized = next_control_deadline(
            deadline=next_deadline,
            wake_time=10.0251,
            dt=0.005,
            severe_lateness=0.005,
        )
        self.assertFalse(resynchronized)
        self.assertAlmostEqual(following, 10.030)


if __name__ == "__main__":
    unittest.main()
