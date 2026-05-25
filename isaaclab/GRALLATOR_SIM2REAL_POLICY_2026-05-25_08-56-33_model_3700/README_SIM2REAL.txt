GRALLATOR SIM-TO-REAL POLICY EXPORT
====================================

Export date:
Mon May 25 08:56:54 UTC 2026

Checkpoint source:
logs/rsl_rl/grallator_flat/2026-05-25_08-21-10/model_3700.pt

Deployment checkpoint:
checkpoint/grallator_flat_sim2real_latest.pt

Task:
Isaac-Velocity-Flat-Grallator-v0

Policy input:
48-dimensional actor input, according to latest training log.

Important sim-to-real notes:
- base_lin_vel is zero-masked in flat_env_cfg.py.
- Policy should not receive true simulator base linear velocity.
- Policy expects observation structure from the saved flat_env_cfg.py.
- Action scale used in training must match deployment.
- Domain randomization included:
  foot friction
  base/trunk mass
  base COM
  joint reset noise
  actuator gain / motor strength
  pushes
  joint encoder noise
  projected gravity / IMU noise if patch was applied

Use this folder as the frozen policy package for deployment.
Do not modify config files inside this export folder.
