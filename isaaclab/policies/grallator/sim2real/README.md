# Grallator Sim-to-Real Policy

Policy file:

policy.pt

Description:

- Exported TorchScript policy for real-time deployment
- Source checkpoint: best_zero_base_vel_extreme_dr.pt
- Trained with extreme domain randomization
- Uses 48 observations
- Base linear velocity is NOT required from real robot
- First 3 observation values must be set to [0, 0, 0]

Observation order:

1. base_lin_vel      3  -> always [0, 0, 0]
2. base_ang_vel      3  -> from IMU gyro
3. projected_gravity 3  -> from IMU orientation
4. command           3  -> [vx_cmd, vy_cmd, yaw_rate_cmd]
5. joint_pos         12 -> joint position relative/default-scaled same as training
6. joint_vel         12 -> joint velocity
7. previous_action   12 -> previous policy action

Total observation dimension: 48

Action dimension: 12

Use case:

Best current Grallator flat-walking sim-to-real policy without real base linear velocity sensor.
