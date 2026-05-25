#!/usr/bin/env python3
"""
============================================================
LOGITECH LEFT-STICK CONTROLLED THREE-POINT GAIT
6 MOTOR / 2 LEG ROBSTRIDE04 VERSION
============================================================

Each leg has:
    COLLAR + HIP + KNEE

Left stick UP:
    forward gait:
    P1 -> P2 -> P3 -> P1

Left stick DOWN:
    reverse gait:
    P1 <- P2 <- P3 <- P1

Left stick CENTER:
    hold current phase / current pose

Collar joints:
    always held fixed at COLLAR_HOME.

WARNING:
    Test unloaded / on stand first.
    Keep emergency power-off ready.
============================================================
"""

from collections import deque
import signal
import time

try:
    import pygame
except ImportError:
    raise SystemExit(
        "pygame is not installed.\n"
        "Install it using:\n"
        "  sudo apt install python3-pygame\n"
        "or:\n"
        "  python3 -m pip install pygame"
    )

from robstride_at_adapter import RobStrideATAdapter

# ============================================================
# MOTOR MAP — CHANGE IDS IF NEEDED
# ============================================================

LEG_A_NAME = "FRONT_LEFT"

LEG_A_COLLAR_MOTOR = 0x0A
LEG_A_HIP_MOTOR    = 0x0B
LEG_A_KNEE_MOTOR   = 0x0C


MOTOR_IDS = {
    "A_COLLAR": LEG_A_COLLAR_MOTOR,
    "A_HIP":    LEG_A_HIP_MOTOR,
    "A_KNEE":   LEG_A_KNEE_MOTOR,
}

ALL_MOTOR_IDS = list(MOTOR_IDS.values())

# ============================================================
# FIXED COLLAR POSITIONS
# ============================================================

LEG_A_COLLAR_HOME = 1.39

# ============================================================
# THREE GAIT POINTS FOR HIP/KNEE
# ============================================================

P1_HIP = 5.36
P1_KNEE = 4.57

P2_HIP = 6.5
P2_KNEE = 2.95

P3_HIP = 5.87
P3_KNEE = 2.95
POSES = [
    ("P1", P1_HIP, P1_KNEE),
    ("P2", P2_HIP, P2_KNEE),
    ("P3", P3_HIP, P3_KNEE),
]

# ============================================================
# GAIT TIMING
# ============================================================

TARGET_CYCLE_TIME = 7.0

_BASE_P1_TO_P2 = 3.0
_BASE_P2_TO_P3 = 3.0
_BASE_P3_TO_P1 = 3.0
_BASE_CYCLE = _BASE_P1_TO_P2 + _BASE_P2_TO_P3 + _BASE_P3_TO_P1
_TIME_SCALE = TARGET_CYCLE_TIME / _BASE_CYCLE

P1_TO_P2_TIME = _BASE_P1_TO_P2 * _TIME_SCALE
P2_TO_P3_TIME = _BASE_P2_TO_P3 * _TIME_SCALE
P3_TO_P1_TIME = _BASE_P3_TO_P1 * _TIME_SCALE

SEGMENT_TIMES = [
    P1_TO_P2_TIME,
    P2_TO_P3_TIME,
    P3_TO_P1_TIME,
]

CYCLE_TIME = P1_TO_P2_TIME + P2_TO_P3_TIME + P3_TO_P1_TIME
MAX_PHASE_SPEED = 1.0 / CYCLE_TIME

RATE_HZ = 100.0
PRINT_HZ = 10.0

# ============================================================
# LOGITECH SETTINGS
# ============================================================

LEFT_STICK_Y_AXIS = 1
STICK_UP_IS_NEGATIVE = True
STICK_DEADZONE = 0.15
STOP_BUTTON = 1

# ============================================================
# MIT GAINS
# ============================================================

KP_COLLAR = 35.0
KD_COLLAR = 3.0

KP_HIP = 20.0
KD_HIP = 2.0

KP_KNEE = 70.0
KD_KNEE = 6.0

KP_LOW = 10.0
KD_LOW = 0.6

HOME_ARM_TIME = 4.0
HOME_RETURN_TIME = 4.0
HOME_KP = 12.0
HOME_KD = 1.2

FRAME_GAP_S = 0.0015

# ============================================================
# SAFETY COMMAND LIMITS
# ============================================================

COLLAR_MIN_CMD = -1.5
COLLAR_MAX_CMD = +1.5

HIP_MIN_CMD = -1.0
HIP_MAX_CMD = 1.0

KNEE_MIN_CMD = 0.0
KNEE_MAX_CMD = 4.5


running = True

HOME_POSE = {
    "A_COLLAR": LEG_A_COLLAR_HOME,
    "A_HIP": P1_HIP,
    "A_KNEE": P1_KNEE,
}
LAST_CMD = dict(HOME_POSE)


def handle_sigint(sig, frame):
    global running
    running = False


signal.signal(signal.SIGINT, handle_sigint)


def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def apply_deadzone(x, deadzone):
    if abs(x) < deadzone:
        return 0.0

    sign = 1.0 if x > 0.0 else -1.0
    mag = (abs(x) - deadzone) / (1.0 - deadzone)
    return sign * clamp(mag, 0.0, 1.0)


def smoothstep(s):
    s = clamp(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def joint_type_from_name(name):
    if name.endswith("_COLLAR"):
        return "COLLAR"
    if name.endswith("_HIP"):
        return "HIP"
    if name.endswith("_KNEE"):
        return "KNEE"
    raise RuntimeError(f"Unknown joint name: {name}")


def gains_for_joint(name):
    jt = joint_type_from_name(name)

    if jt == "COLLAR":
        return KP_COLLAR, KD_COLLAR
    if jt == "HIP":
        return KP_HIP, KD_HIP
    if jt == "KNEE":
        return KP_KNEE, KD_KNEE

    raise RuntimeError(f"Unknown joint type for {name}")


def check_command_limit(name, pos):
    jt = joint_type_from_name(name)

    if jt == "COLLAR":
        lo, hi = COLLAR_MIN_CMD, COLLAR_MAX_CMD
    elif jt == "HIP":
        lo, hi = HIP_MIN_CMD, HIP_MAX_CMD
    elif jt == "KNEE":
        lo, hi = KNEE_MIN_CMD, KNEE_MAX_CMD
    else:
        raise RuntimeError(f"Unknown joint type for {name}")

    if not (lo <= pos <= hi):
        raise RuntimeError(
            f"{name} command {pos:+.4f} outside [{lo:+.4f}, {hi:+.4f}]"
        )


def validate_motor_ids():
    ids = list(MOTOR_IDS.values())
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate motor IDs found. Each motor must have unique CAN ID.")

    print("MOTOR ID MAP:")
    print(f"  {LEG_A_NAME:>10s} collar = 0x{LEG_A_COLLAR_MOTOR:02X}")
    print(f"  {LEG_A_NAME:>10s} hip    = 0x{LEG_A_HIP_MOTOR:02X}")
    print(f"  {LEG_A_NAME:>10s} knee   = 0x{LEG_A_KNEE_MOTOR:02X}")
    print()

def send_pose_mit(bus, pose, velocity=None, gains_override=None, torque=None):
    global LAST_CMD

    if velocity is None:
        velocity = {}

    if torque is None:
        torque = {}

    for name in MOTOR_IDS:
        check_command_limit(name, pose[name])

    for name, motor_id in MOTOR_IDS.items():
        pos = pose[name]
        vel = velocity.get(name, 0.0)
        tau = torque.get(name, 0.0)

        if gains_override is not None and name in gains_override:
            kp, kd = gains_override[name]
        else:
            kp, kd = gains_for_joint(name)

        bus.mit_control(
            motor_id,
            position_rad=pos,
            velocity_rad_s=vel,
            kp=kp,
            kd=kd,
            torque_nm=tau,
            verbose=False,
        )

        LAST_CMD[name] = pos
        time.sleep(FRAME_GAP_S)


def constant_gains(kp, kd):
    return {name: (kp, kd) for name in MOTOR_IDS}


def soft_arm_to_home(ser):
    print()
    print(f"SOFT ARM TO HOME: {HOME_ARM_TIME:.2f}s")
    print(f"  {LEG_A_NAME} collar={LEG_A_COLLAR_HOME:+.4f}, hip={P1_HIP:+.4f}, knee={P1_KNEE:+.4f}")

    dt = 1.0 / RATE_HZ
    n_steps = max(1, int(HOME_ARM_TIME * RATE_HZ))

    for i in range(n_steps):
        if not running:
            break

        a = smoothstep(i / n_steps)

        kp = KP_LOW + a * (HOME_KP - KP_LOW)
        kd = KD_LOW + a * (HOME_KD - KD_LOW)

        send_pose_mit(
            ser,
            HOME_POSE,
            velocity={},
            gains_override=constant_gains(kp, kd),
        )

        if i % max(1, int(RATE_HZ / PRINT_HZ)) == 0:
            print(
                f"soft home | kp={kp:.2f} kd={kd:.2f} | "
                f"A=[{LEG_A_COLLAR_HOME:+.3f}, {P1_HIP:+.3f}, {P1_KNEE:+.3f}]"
            )

        time.sleep(dt)

def slow_return_to_home(ser, pose_from):
    print()
    print(f"SLOW RETURN TO HOME: {HOME_RETURN_TIME:.2f}s")

    dt = 1.0 / RATE_HZ
    n_steps = max(1, int(HOME_RETURN_TIME * RATE_HZ))

    last = dict(pose_from)

    for i in range(n_steps + 1):
        a = smoothstep(i / n_steps)

        pose = {}
        vel = {}

        for name in MOTOR_IDS:
            start_pos = pose_from[name]
            end_pos = HOME_POSE[name]

            pos = start_pos + a * (end_pos - start_pos)
            pose[name] = pos
            vel[name] = (pos - last[name]) / dt
            last[name] = pos

        send_pose_mit(
            ser,
            pose,
            velocity=vel,
            gains_override=constant_gains(HOME_KP, HOME_KD),
        )

        if i % max(1, int(RATE_HZ / PRINT_HZ)) == 0:
            print(
                f"slow home | "
                f"A=[{pose['A_COLLAR']:+.3f}, {pose['A_HIP']:+.3f}, {pose['A_KNEE']:+.3f}]"
            )

        time.sleep(dt)

    for _ in range(30):
        send_pose_mit(
            ser,
            HOME_POSE,
            gains_override=constant_gains(KP_LOW, KD_LOW),
        )
        time.sleep(0.01)

def cubic_hermite(p0, p1, v0, v1, duration, s):
    s = clamp(s, 0.0, 1.0)

    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2

    pos = h00 * p0 + h10 * duration * v0 + h01 * p1 + h11 * duration * v1

    dh00 = 6.0 * s**2 - 6.0 * s
    dh10 = 3.0 * s**2 - 4.0 * s + 1.0
    dh01 = -6.0 * s**2 + 6.0 * s
    dh11 = 3.0 * s**2 - 2.0 * s

    vel = (
        dh00 * p0
        + dh10 * duration * v0
        + dh01 * p1
        + dh11 * duration * v1
    ) / duration

    return pos, vel


def compute_cyclic_point_velocities():
    hip = [P1_HIP, P2_HIP, P3_HIP]
    knee = [P1_KNEE, P2_KNEE, P3_KNEE]
    times = SEGMENT_TIMES

    hip_vel = []
    knee_vel = []

    for i in range(3):
        prev_i = (i - 1) % 3
        next_i = (i + 1) % 3

        t_prev_to_i = times[prev_i]
        t_i_to_next = times[i]

        hip_slope_in = (hip[i] - hip[prev_i]) / t_prev_to_i
        hip_slope_out = (hip[next_i] - hip[i]) / t_i_to_next

        knee_slope_in = (knee[i] - knee[prev_i]) / t_prev_to_i
        knee_slope_out = (knee[next_i] - knee[i]) / t_i_to_next

        hip_vel.append(0.5 * (hip_slope_in + hip_slope_out))
        knee_vel.append(0.5 * (knee_slope_in + knee_slope_out))

    return hip_vel, knee_vel


HIP_POINT_VELS, KNEE_POINT_VELS = compute_cyclic_point_velocities()


def gait_at_phase(phase):
    phase = phase % 1.0
    t = phase * CYCLE_TIME

    if t < P1_TO_P2_TIME:
        start_idx = 0
        end_idx = 1
        local_t = t
        duration = P1_TO_P2_TIME
    elif t < P1_TO_P2_TIME + P2_TO_P3_TIME:
        start_idx = 1
        end_idx = 2
        local_t = t - P1_TO_P2_TIME
        duration = P2_TO_P3_TIME
    else:
        start_idx = 2
        end_idx = 0
        local_t = t - P1_TO_P2_TIME - P2_TO_P3_TIME
        duration = P3_TO_P1_TIME

    _, hip0, knee0 = POSES[start_idx]
    _, hip1, knee1 = POSES[end_idx]

    s = local_t / duration

    hip_pos, hip_vel = cubic_hermite(
        hip0,
        hip1,
        HIP_POINT_VELS[start_idx],
        HIP_POINT_VELS[end_idx],
        duration,
        s,
    )

    knee_pos, knee_vel = cubic_hermite(
        knee0,
        knee1,
        KNEE_POINT_VELS[start_idx],
        KNEE_POINT_VELS[end_idx],
        duration,
        s,
    )

    return hip_pos, knee_pos, hip_vel, knee_vel


def init_joystick():
    pygame.init()
    pygame.joystick.init()

    count = pygame.joystick.get_count()

    if count == 0:
        raise RuntimeError("No joystick found. Check Logitech receiver / USB connection.")

    joy = pygame.joystick.Joystick(0)
    joy.init()

    print("LOGITECH CONTROLLER FOUND")
    print(f"  name    = {joy.get_name()}")
    print(f"  axes    = {joy.get_numaxes()}")
    print(f"  buttons = {joy.get_numbuttons()}")
    print()

    return joy


def read_left_stick_speed(joy):
    pygame.event.pump()

    raw_y = joy.get_axis(LEFT_STICK_Y_AXIS)

    if STICK_UP_IS_NEGATIVE:
        cmd = -raw_y
    else:
        cmd = raw_y

    cmd = apply_deadzone(cmd, STICK_DEADZONE)

    return raw_y, cmd


def check_stop_button(joy):
    global running

    pygame.event.pump()

    if joy.get_numbuttons() > STOP_BUTTON:
        if joy.get_button(STOP_BUTTON):
            print("\nStop button pressed.")
            running = False


def print_summary():
    print("=" * 74)
    print("LOGITECH LEFT-STICK THREE-POINT GAIT — FRONT_LEFT 3 MOTOR ROBSTRIDE04")
    print("=" * 74)

    validate_motor_ids()

    print("GAIT POINTS:")
    for name, h, k in POSES:
        print(f"  {name}: HIP={h:+.4f}, KNEE={k:+.4f}")
    print()

    print("COLLAR HOLD:")
    print(f"  {LEG_A_NAME} collar fixed at {LEG_A_COLLAR_HOME:+.4f} rad")
    print()

    print("JOYSTICK:")
    print("  left stick UP     -> FRONT_LEFT forward gait")
    print("  left stick DOWN   -> FRONT_LEFT reverse gait")
    print("  left stick CENTER -> hold current pose")
    print(f"  stop button       -> button {STOP_BUTTON}")
    print()

    print("TIMING:")
    print(f"  P1 -> P2 = {P1_TO_P2_TIME:.2f}s")
    print(f"  P2 -> P3 = {P2_TO_P3_TIME:.2f}s")
    print(f"  P3 -> P1 = {P3_TO_P1_TIME:.2f}s")
    print(f"  cycle time = {CYCLE_TIME:.2f}s")
    print()

    print("GAINS:")
    print(f"  COLLAR KP={KP_COLLAR:.1f}, KD={KD_COLLAR:.1f}")
    print(f"  HIP    KP={KP_HIP:.1f}, KD={KD_HIP:.1f}")
    print(f"  KNEE   KP={KP_KNEE:.1f}, KD={KD_KNEE:.1f}")
    print()

    print("OPEN-LOOP MODE: no feedback required.")
    print("WARNING: Test unloaded / on stand first.")
    print()

def confirm_before_enable():
    print("SAFETY CHECK:")
    print("  1. Robot/leg is unloaded or on a stand.")
    print("  2. Emergency power-off is ready.")
    print("  3. Motor IDs and collar home angles above are correct.")
    ans = input("Type YES to enable motors and continue: ").strip()

    if ans != "YES":
        raise RuntimeError("User did not type YES. Exiting before enabling motors.")



def build_leg_a_pose_and_velocity(phase_a, speed_a):
    a_hip, a_knee, a_hip_vel_nom, a_knee_vel_nom = gait_at_phase(phase_a)

    pose = {
        "A_COLLAR": LEG_A_COLLAR_HOME,
        "A_HIP": a_hip,
        "A_KNEE": a_knee,
    }

    vel = {
        "A_COLLAR": 0.0,
        "A_HIP": a_hip_vel_nom * speed_a,
        "A_KNEE": a_knee_vel_nom * speed_a,
    }

    return pose, vel


def run_joystick_gait(ser, joy):
    global running

    dt = 1.0 / RATE_HZ
    print_interval = max(1, int(RATE_HZ / PRINT_HZ))

    phase_a = 0.0

    print("\nStarting FRONT_LEFT 3-motor joystick gait control.")
    print("Keep left stick centered to hold current pose.")
    print("Move left stick UP/DOWN to move gait forward/reverse.\n")

    i = 0

    while running:
        check_stop_button(joy)

        raw_y, speed_a = read_left_stick_speed(joy)

        phase_a = (phase_a + speed_a * MAX_PHASE_SPEED * dt) % 1.0

        pose, vel = build_leg_a_pose_and_velocity(
            phase_a=phase_a,
            speed_a=speed_a,
        )

        send_pose_mit(ser, pose, velocity=vel)

        if i % print_interval == 0:
            if speed_a > 0.05:
                direction = "FORWARD"
            elif speed_a < -0.05:
                direction = "REVERSE"
            else:
                direction = "HOLD"

            print(
                f"{direction:7s} | "
                f"raw_y={raw_y:+.3f} speed_A={speed_a:+.3f} | "
                f"phase_A={phase_a:.3f} | "
                f"A=[{pose['A_COLLAR']:+.3f}, {pose['A_HIP']:+.3f}, {pose['A_KNEE']:+.3f}]"
            )

        i += 1
        time.sleep(dt)

    return dict(LAST_CMD)


def main():
    global running

    print_summary()

    bus = None

    try:
        joy = init_joystick()

        confirm_before_enable()

        bus = RobStrideATAdapter(port="/dev/ttyUSB0", baud=921600, host_id=0xFD).open()

        print("Adapter opened.")

        print("Stopping motors...")
        for motor_id in ALL_MOTOR_IDS:
            bus.stop(motor_id, verbose=True)
            time.sleep(0.05)

        time.sleep(0.2)

        print("Setting MIT mode...")
        for motor_id in ALL_MOTOR_IDS:
            print(f"  motor 0x{motor_id:02X} -> MIT mode")
            bus.set_mit_mode(motor_id, verbose=True)
            time.sleep(0.05)

        time.sleep(0.2)

        print("Enabling motors...")
        for motor_id in ALL_MOTOR_IDS:
            bus.enable(motor_id, verbose=True)
            time.sleep(0.05)

        time.sleep(0.2)

        soft_arm_to_home(bus)

        run_joystick_gait(bus, joy)

    except KeyboardInterrupt:
        print("\nCtrl+C received.")

    except Exception as e:
        print(f"\nERROR: {e}")

    finally:
        print("\nStopping safely...")

        try:
            if bus is not None:
                slow_return_to_home(bus, dict(LAST_CMD))
        except Exception as e:
            print(f"slow return skipped: {e}")

        try:
            if bus is not None:
                for motor_id in ALL_MOTOR_IDS:
                    bus.stop(motor_id, verbose=True)
                    time.sleep(0.03)
        except Exception as e:
            print(f"stop skipped: {e}")

        try:
            if bus is not None:
                bus.close()
        except Exception:
            pass

        try:
            pygame.quit()
        except Exception:
            pass

        print("Done. Motors stopped.")


if __name__ == "__main__":
    main()

