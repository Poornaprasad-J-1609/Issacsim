#!/usr/bin/env python3
import time


def serial_encoded_ext_id(can_id: int) -> bytes:
    """
    AT-command USB-CAN extended-frame encoding.
    Intended for CH340 / RobStride-style AT USB-CAN adapter.

    This is only the serial packet layer.
    """
    serial_id = ((can_id & 0x1FFFFFFF) << 3) | 0x04
    return serial_id.to_bytes(4, "big")


def make_at_packet(can_id: int, data: bytes = b"") -> bytes:
    if len(data) > 8:
        raise ValueError("CAN data length must be <= 8 bytes")
    return b"AT" + serial_encoded_ext_id(can_id) + bytes([len(data)]) + data + b"\r\n"


class ATUsbCan:
    def __init__(self, port="/dev/ttyUSB0", baud=921600, timeout=0.02):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None

    def open(self):
        import serial

        self.ser = serial.Serial(
            self.port,
            self.baud,
            timeout=self.timeout,
            write_timeout=self.timeout,
        )
        time.sleep(0.1)
        return self

    def close(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def send_raw(self, can_id: int, data: bytes = b""):
        if self.ser is None:
            raise RuntimeError("Serial port is not open")

        pkt = make_at_packet(can_id, data)
        self.ser.write(pkt)
        self.ser.flush()
        return pkt

    def send_signal_frame(self, motor_id: int):
        """
        Harmless signal test frame.

        This only tests that the USB-CAN serial adapter transmits bytes.
        Motors do not need to be connected.

        This is NOT a RobStride position command.
        """
        can_id = int(motor_id) & 0x1FFFFFFF
        data = b""
        return self.send_raw(can_id, data)

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()


if __name__ == "__main__":
    print("AT packet example:")
    print(make_at_packet(0x01, b"").hex())
