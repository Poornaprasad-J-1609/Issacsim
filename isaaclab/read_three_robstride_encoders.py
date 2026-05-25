#!/usr/bin/env python3
import time
import math
import argparse
import serial

P_MIN, P_MAX = -12.5, 12.5
V_MIN, V_MAX = -30.0, 30.0
T_MIN, T_MAX = -12.0, 12.0

def uint_to_float(x, x_min, x_max, bits):
    return x_min + float(x) * (x_max - x_min) / ((1 << bits) - 1)

def build_can_id(comm_type, motor_id, host_id):
    return ((comm_type & 0x1F) << 24) | ((host_id & 0xFFFF) << 8) | (motor_id & 0xFF)

def encode_at_frame(can_id, data):
    # RobStride official adapter binary frame:
    # b"AT" + uint32_be((can_id << 3) | 0x04) + DLC + DATA + b"\r\n"
    enc_id = ((can_id & 0x1FFFFFFF) << 3) | 0x04
    return b"AT" + enc_id.to_bytes(4, "big") + bytes([len(data)]) + bytes(data) + b"\r\n"

def decode_can_id(can_id):
    comm_type = (can_id >> 24) & 0x1F
    middle = (can_id >> 8) & 0xFFFF
    low = can_id & 0xFF
    return comm_type, middle, low

def decode_feedback(data):
    if len(data) < 8:
        return None

    pos_raw = (data[0] << 8) | data[1]
    vel_raw = (data[2] << 8) | data[3]
    tor_raw = (data[4] << 8) | data[5]
    temp_raw = (data[6] << 8) | data[7]

    pos_rad = uint_to_float(pos_raw, P_MIN, P_MAX, 16)
    vel_rad_s = uint_to_float(vel_raw, V_MIN, V_MAX, 16)
    torque_nm = uint_to_float(tor_raw, T_MIN, T_MAX, 16)
    temp_c = temp_raw / 10.0

    return pos_raw, pos_rad, vel_rad_s, torque_nm, temp_c

def read_at_frames(ser, rx_buffer):
    frames = []

    data = ser.read(512)
    if data:
        rx_buffer.extend(data)

    while True:
        idx = rx_buffer.find(b"AT")
        if idx < 0:
            rx_buffer.clear()
            break

        if idx > 0:
            del rx_buffer[:idx]

        if len(rx_buffer) < 9:
            break

        enc_id = int.from_bytes(rx_buffer[2:6], "big")
        dlc = rx_buffer[6]

        if dlc > 8:
            del rx_buffer[0:2]
            continue

        frame_len = 2 + 4 + 1 + dlc + 2

        if len(rx_buffer) < frame_len:
            break

        frame = bytes(rx_buffer[:frame_len])
        del rx_buffer[:frame_len]

        if not frame.endswith(b"\r\n"):
            continue

        can_id = enc_id >> 3
        payload = frame[7:7 + dlc]
        frames.append((can_id, payload))

    return frames

def update_unwrap(state, pos_rad):
    if state["last_pos"] is None:
        state["unwrapped"] = pos_rad
    else:
        diff = pos_rad - state["last_pos"]

        if diff > math.pi:
            diff -= 2.0 * math.pi
        elif diff < -math.pi:
            diff += 2.0 * math.pi

        state["unwrapped"] += diff

    state["last_pos"] = pos_rad

def print_table(states, motor_ids):
    print("\033[2J\033[H", end="")
    print("=" * 110)
    print(f"ROBSTRIDE {len(motor_ids)}-MOTOR ENCODER / JOINT POSITION READER - OFFICIAL AT ADAPTER")
    print("=" * 110)
    print("Move motors manually. Press Ctrl+C to stop.")
    print("This script does NOT enable motors and does NOT command motion.")
    print("=" * 110)
    print(
        f"{'Joint':>8} | {'Motor ID':>8} | {'Raw':>6} | {'Angle rad':>12} | "
        f"{'Angle deg':>12} | {'Unwrapped rad':>14} | {'Vel rad/s':>10} | {'Temp C':>8} | Status"
    )
    print("-" * 110)

    now = time.time()

    default_names = ["Collar", "Hip", "Knee"]
    names = {
        mid: default_names[index] if index < len(default_names) else f"Joint{index}"
        for index, mid in enumerate(motor_ids)
    }

    for mid in motor_ids:
        s = states[mid]
        joint = names.get(mid, "Joint")

        if s["last_seen"] is None:
            print(
                f"{joint:>8} | 0x{mid:02X}     | {'-':>6} | {'-':>12} | "
                f"{'-':>12} | {'-':>14} | {'-':>10} | {'-':>8} | NO DATA"
            )
            continue

        age = now - s["last_seen"]
        status = "OK" if age < 0.5 else "STALE"

        print(
            f"{joint:>8} | 0x{mid:02X}     | "
            f"{s['raw']:6d} | "
            f"{s['pos_rad']:+12.5f} | "
            f"{math.degrees(s['pos_rad']):+12.2f} | "
            f"{s['unwrapped']:+14.5f} | "
            f"{s['vel']:+10.4f} | "
            f"{s['temp']:8.1f} | "
            f"{status}"
        )

    print("=" * 110)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--host-id", type=lambda x: int(x, 0), default=0xFD)
    parser.add_argument("--motor-ids", type=lambda x: int(x, 0), nargs="+", default=[0x04, 0x05, 0x06])
    parser.add_argument("--rate", type=float, default=50.0)
    args = parser.parse_args()

    if len(args.motor_ids) < 1:
        raise SystemExit("Please give at least one motor ID, example: --motor-ids 0x05 0x06")

    motor_ids = args.motor_ids
    dt = 1.0 / args.rate

    states = {}
    for mid in motor_ids:
        states[mid] = {
            "raw": 0,
            "pos_rad": 0.0,
            "vel": 0.0,
            "torque": 0.0,
            "temp": 0.0,
            "last_pos": None,
            "unwrapped": 0.0,
            "last_seen": None,
        }

    print("Opening RobStride official adapter...")
    print(f"port={args.port}, baud={args.baud}, motors={[hex(m) for m in motor_ids]}")

    ser = serial.Serial(args.port, args.baud, timeout=0.001)
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    rx_buffer = bytearray()

    # Communication type 4 = safe stop/poll frame.
    poll_frames = []
    for mid in motor_ids:
        can_id = build_can_id(4, mid, args.host_id)
        poll_frames.append(encode_at_frame(can_id, [0, 0, 0, 0, 0, 0, 0, 0]))

    try:
        last_print = 0.0

        while True:
            cycle_start = time.time()

            for frame in poll_frames:
                ser.write(frame)
                time.sleep(0.001)

            read_deadline = time.time() + dt * 0.8

            while time.time() < read_deadline:
                frames = read_at_frames(ser, rx_buffer)

                for can_id, payload in frames:
                    comm_type, middle, low = decode_can_id(can_id)

                    if comm_type != 2:
                        continue

                    motor_id = middle & 0xFF

                    if motor_id not in states:
                        continue

                    decoded = decode_feedback(payload)
                    if decoded is None:
                        continue

                    pos_raw, pos_rad, vel_rad_s, torque_nm, temp_c = decoded

                    s = states[motor_id]
                    s["raw"] = pos_raw
                    s["pos_rad"] = pos_rad
                    s["vel"] = vel_rad_s
                    s["torque"] = torque_nm
                    s["temp"] = temp_c
                    s["last_seen"] = time.time()

                    update_unwrap(s, pos_rad)

                time.sleep(0.001)

            if time.time() - last_print > 0.1:
                print_table(states, motor_ids)
                last_print = time.time()

            elapsed = time.time() - cycle_start
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        ser.close()
        print("Serial closed.")

if __name__ == "__main__":
    main()
