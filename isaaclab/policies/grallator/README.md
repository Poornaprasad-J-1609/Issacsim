# Grallator Best Policy

Policy file:

best_zero_base_vel_extreme_dr.pt

Description:

- 48-observation policy
- base linear velocity observation is kept, but replaced with [0, 0, 0]
- trained with extreme domain randomization
- selected after visual play test looked perfect
- task: Isaac-Velocity-Flat-Grallator-v0
- use case: sim-to-real flat walking without real base linear velocity sensor

Observation structure:

base_lin_vel      3  -> zeroed
base_ang_vel      3
projected_gravity 3
command           3
joint_pos         12
joint_vel         12
previous_action   12

Total: 48 observations
