# Grallator PACE Actuator Modeling

This is a separate 12-actuator experiment controller and logger. It does not
modify `GRALLATOR_DEPLOY`. It reuses that repository's proven RobStride packet,
SocketCAN, routing, and feedback decoder as a runtime dependency.

## Fixed contract

- Control and logging: 50 Hz (`dt=0.02 s`)
- Two CAN adapters: front legs on `slcan0`, back legs on `slcan1`
- `v_des=0`, `tau_ff=0`
- `q_des`: final logical position actually put into the MIT command after hard,
  rate, and estimated-torque limits
- `q_actual`: real encoder feedback converted to logical Isaac coordinates
- `q_raw/qd_raw`: real mechanical position/velocity from MIT status feedback
- Incomplete current-cycle feedback aborts the experiment

The joint order is always:

```text
FL_hip, FR_hip, BL_hip, BR_hip,
FL_thigh, FR_thigh, BL_thigh, BR_thigh,
FL_calf, FR_calf, BL_calf, BR_calf
```

## Trajectory format

Edit `trajectories/custom_angles_template.yaml`. Targets use logical
Isaac/URDF radians and may be supplied by joint name. Supported segment types:

- `hold`
- `linear`
- `smoothstep`
- `minimum_jerk` (`10u^3 - 15u^4 + 6u^5`)
- `sine`
- `chirp` with `linear` or `logarithmic` frequency law

Unspecified joints retain the preceding segment's target. The runtime always
logs both the requested trajectory and the final transmitted target.
Use `relative_target` instead of `target` to specify an offset from the target
at the beginning of that segment.

## Kp/Kd testing phase

Before collecting the full PACE datasets, run the one-joint suspended test:

```bash
python3 -m pace_modeling \
  --config config/testing_phase_kp250_kd4.yaml \
  --deploy-root ~/JetsonNanoDeploy \
  --dataset testing_phase \
  --trajectory trajectories/testing_phase_small_movement.yaml \
  --can-front slcan0 \
  --can-back slcan1
```

This uses `Kp=250`, `Kd=4` on all twelve enabled actuators but moves only the
FL hip by `+0.05 rad` relative to its measured start and then returns it. The
robot must be fully suspended. Use the same gains when replaying this dataset
in Isaac; otherwise PACE can mistake controller mismatch for plant dynamics.

## Dry run

From this directory:

```bash
export PYTHONPATH="$PWD"

python3 -m pace_modeling \
  --dry-run \
  --dataset dataset_A \
  --trajectory trajectories/dataset_A_chirp_20s.yaml
```

The 20-second example must produce exactly 1000 rows beneath:

```text
actuator_modeling_logs/dataset_A/<timestamp>_dry/
```

## Jetson installation

Place this folder inside the existing repository without replacing deployment
code:

```text
~/JetsonNanoDeploy/actuator_modeling/
```

Then:

```bash
cd ~/JetsonNanoDeploy/actuator_modeling
export PYTHONPATH="$PWD"
sudo ip link set slcan0 txqueuelen 32
sudo ip link set slcan1 txqueuelen 32
```

## Real suspended test

Review every requested angle, physically suspend the robot, clear the work
area, and keep emergency power removal accessible. Start Dataset A with:

```bash
python3 -m pace_modeling \
  --deploy-root ~/JetsonNanoDeploy \
  --dataset dataset_A \
  --trajectory trajectories/dataset_A_chirp_20s.yaml \
  --can-front slcan0 \
  --can-back slcan1
```

The program first polls all 12 encoders while disabled, prints the measured
starting pose, and requires the exact phrase `ENABLE PACE` before enabling.
Press `Ctrl+C` to stop; stop frames are sent in `finally` on every normal Python
exit or detected safety violation.

Run Dataset B separately:

```bash
python3 -m pace_modeling \
  --deploy-root ~/JetsonNanoDeploy \
  --dataset dataset_B \
  --trajectory trajectories/dataset_B_validation_20s.yaml \
  --can-front slcan0 \
  --can-back slcan1
```

Never fit PACE on Dataset B. It is reserved for validation.

## Logs

Every run creates a timestamped directory under `actuator_modeling_logs/` with:

- `pace_*.csv`: all 50 Hz samples
- `pace_*_metadata.json`: gains, limits, routing, signs, offsets, trajectory,
  source commit, and feedback-source definitions

The main CSV includes command time, feedback time, per-joint feedback age,
requested and sent targets, logical and raw feedback, effective gains, torque,
temperature, fault bits, motor IDs, and buses.

## PACE export

After checking that every row has complete feedback and no safety event:

```bash
python3 export_pace.py \
  actuator_modeling_logs/dataset_A/<run>/pace_dataset_A_*.csv \
  --output chirp_data.pt
```

The output contains:

```python
{
    "time":        [N],
    "dof_pos":     [N, 12],
    "des_dof_pos": [N, 12],
    "joint_order": [...],
}
```

## Feedback-source precision

The active experiment uses operation-status replies (`comm_type=2`). These
contain the real RS04 mechanical position and velocity and are converted using
the configured direction and offset. Separate `0x7019/0x701B` parameter reads
are deliberately not interleaved with MIT streaming because doing so can make
the SDK consume a status frame as a parameter reply. The metadata records this
distinction explicitly.
