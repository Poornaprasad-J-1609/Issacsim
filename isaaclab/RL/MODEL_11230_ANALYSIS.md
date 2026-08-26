# Model 11230 integration analysis

## Artifact

- Repository file: `RL/policy/model_11230_omni_stand50_best.pt`
- SHA256: `52105ae8c6572ca379cbc196510fc5498a2875bac57f545fc05de2cff39db03c`
- Format: RSL-RL checkpoint dictionary, not TorchScript
- Training iteration: 11230
- Actor: `48 -> 512 -> 256 -> 128 -> 12`, ELU activations

`grallator_rl_takeover.py` now supports this checkpoint format and verifies the
artifact hash from `policy_spec_11230_48d.json`.

## Contract represented by the supplied companion evaluator

- Policy rate: 50 Hz
- Low-level rate: 200 Hz
- Actor order: BL, BR, FL, FR for hips, then thighs, then calves
- Action target: `q_target = 0.25 * raw_action`
- No routine action or observation clipping in the evaluator
- Observation:
  - `[0:3]` base linear velocity (the hardware fallback is zero)
  - `[3:6]` body angular velocity
  - `[6:9]` projected gravity
  - `[9:12]` velocity command
  - `[12:24]` joint position relative to zero default
  - `[24:36]` joint velocity
  - `[36:48]` previous raw actor action

The policy spec uses unit observation scales because the supplied evaluator
concatenates these values without additional scaling.

## Hardware blockers in this prototype

Hardware motion remains disabled in `policy_spec_11230_48d.json` because the
standalone RL controller is not yet equivalent to the verified deployment
controller:

1. Its hardcoded `MOTOR_DIRECTION` values are all `+1`; the deployed mapping
   uses `-1` for every hip/thigh and `+1` for every calf.
2. It captures raw encoders as a temporary stand reference instead of using
   the calibrated offset/direction mapping.
3. The included IMU provider is a template, not an Xsens implementation.
4. It has no feedback-age, tilt, temperature, current, torque, or loop-deadline
   watchdog comparable to the deployment controller.
5. It does not log the full observation/action/target/feedback path.
6. Its 200 Hz loop sends sequential motor commands and does not qualify or
   report the actual CAN deadline.

The actor can therefore be tested offline, but this file should not command the
robot until those integrations are replaced with the verified deployment
layers.

## Motors-disabled check

```bash
cd /home/poornaprasad/Work/WORKSPACE/Issacsim/isaaclab

python3 RL/grallator_rl_takeover.py \
  --policy-spec RL/policy_spec_11230_48d.json \
  --policy-check-only \
  --vx 0.10 \
  --vy 0.00 \
  --yaw 0.00
```

This validates the hash, checkpoint tensors, 48-to-12 dimensions, nominal
level observation, and command-conditioned inference without opening CAN.
