"""
Direct SocketCAN access to one OpenArm v1 arm through Enactic's openarm_can.

This is the only module of the package that imports openarm_can: its Python API is a
1:1 port of the C++ classes that upstream plans to redesign, so everything that depends
on it stays behind OpenArmCANBus. The project pins openarm_can 1.4.1.

One bus carries one arm: joints 1-7 on send IDs 0x01-0x07 (replies on 0x11-0x17) and
the gripper on 0x08 (0x18). FakeCANBus has the same interface without any hardware.
"""

import time
from dataclasses import dataclass, field

import numpy as np

# Motor IDs are listed in ascending order: openarm_can indexes motors by reply ID.
ARM_MOTOR_TYPES = ("DM8009", "DM8009", "DM4340", "DM4340", "DM4310", "DM4310", "DM4310")
ARM_SEND_IDS = (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07)
ARM_RECV_IDS = (0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17)
GRIPPER_MOTOR_TYPE = "DM4310"
GRIPPER_SEND_ID = 0x08
GRIPPER_RECV_ID = 0x18
MOTOR_NAMES = tuple(f"joint{i}" for i in range(1, 8)) + ("gripper",)

# Parallel gripper, as in openarm_hardware: closed at 0 rad, 0.044 m open at -1.0472 rad
GRIPPER_OPEN_M = 0.044
GRIPPER_OPEN_RAD = -1.0472

BUS_STATUS_FLAGS = ("bus_off", "error_passive", "tx_overflow", "rx_overflow", "ack_error",
                    "tx_timeout", "write_net_down", "write_no_buffer", "write_other")


def opening_to_motor(opening):
    """Gripper opening in m -> motor angle in rad."""
    return opening / GRIPPER_OPEN_M * GRIPPER_OPEN_RAD


def motor_to_opening(angle):
    """Gripper motor angle in rad -> opening in m."""
    return angle / GRIPPER_OPEN_RAD * GRIPPER_OPEN_M


@dataclass
class ArmState:
    """Latest state of one arm. Angles are URDF joint angles once the zeros are calibrated."""

    q: np.ndarray  # joints 1-7, rad
    dq: np.ndarray  # rad/s
    tau: np.ndarray  # Nm
    gripper: float | None  # opening in m; None when the gripper is not on the bus
    ages: np.ndarray  # seconds since each motor last replied (joints, then gripper)
    faults: list = field(default_factory=list)  # motor error codes and bus errors, as text

    @property
    def age(self):
        return float(np.max(self.ages))

    def silent_motors(self, timeout):
        return [MOTOR_NAMES[i] for i, age in enumerate(self.ages) if age > timeout]


class OpenArmCANBus:
    """One arm on one SocketCAN interface (classic CAN 2.0 unless fd=True)."""

    def __init__(self, interface, gripper=True, fd=False):
        self.interface = interface
        self.gripper = gripper
        self.fd = fd
        self._oa = None
        self._openarm = None

    def open(self):
        try:
            import openarm_can as oa
        except ImportError as exc:
            raise RuntimeError(
                'Hardware support is not installed. Install eurola-openarm-teleop[hardware].'
            ) from exc

        self._oa = oa
        try:
            # Kept for the lifetime of the bus: the arm and gripper components belong to it
            self._openarm = oa.OpenArm(self.interface, self.fd)
            # The control mode is given explicitly: without it openarm_can assumes MIT but
            # leaves the motors in whatever mode they were last put in, and a motor in
            # another mode neither follows nor answers MIT commands. (The gripper's mode
            # is always written.) This writes the mode register of each motor.
            self._openarm.init_arm_motors(
                [getattr(oa.MotorType, name) for name in ARM_MOTOR_TYPES],
                list(ARM_SEND_IDS), list(ARM_RECV_IDS), [oa.ControlMode.MIT],
            )
            if self.gripper:
                self._openarm.init_gripper_motor(
                    getattr(oa.MotorType, GRIPPER_MOTOR_TYPE), GRIPPER_SEND_ID, GRIPPER_RECV_ID
                )
            # Take in the answers to the mode writes as register replies, not as motor states
            self._openarm.set_callback_mode_all(oa.CallbackMode.PARAM)
            time.sleep(0.05)
            self._openarm.recv_all(2000)
            self._openarm.set_callback_mode_all(oa.CallbackMode.STATE)
        except Exception as e:
            self._openarm = None
            raise RuntimeError(f"cannot open {self.interface}: {e}") from e

    def close(self):
        self._openarm = None

    def _components(self):
        components = [self._openarm.get_arm()]
        if self.gripper:
            components.append(self._openarm.get_gripper())
        return components

    def request_state(self):
        """Ask every motor for its state. A read request only; not needed while commanding."""
        self._openarm.refresh_all()

    def read(self):
        """Take in the replies received so far (without waiting) and return the arm state."""
        self._openarm.recv_all(0)
        arm = self._openarm.get_arm()
        motors = arm.get_motors()  # copies: fetched again on every read
        ages = [arm.get_link_stats(i).seconds_since_response() for i in range(len(motors))]
        opening = None
        if self.gripper:
            gripper = self._openarm.get_gripper()
            motors = motors + gripper.get_motors()
            ages.append(gripper.get_link_stats(0).seconds_since_response())
            opening = motor_to_opening(motors[7].get_position())

        faults = [
            f"{self.interface} {MOTOR_NAMES[i]}: "
            f"{self._oa.motor_error_to_string(motor.get_error_code())}"
            for i, motor in enumerate(motors)
            if motor.has_error()
        ]
        if not self._openarm.is_bus_healthy():
            status = self._openarm.get_bus_status()
            flags = [name for name in BUS_STATUS_FLAGS if getattr(status, name)]
            faults.append(f"{self.interface} bus error: {', '.join(flags)}")

        return ArmState(
            q=np.array([motor.get_position() for motor in motors[:7]]),
            dq=np.array([motor.get_velocity() for motor in motors[:7]]),
            tau=np.array([motor.get_torque() for motor in motors[:7]]),
            gripper=opening,
            ages=np.array(ages),
            faults=faults,
        )

    def enable(self):
        self._openarm.enable_all()
        time.sleep(0.05)
        self._openarm.recv_all(0)

    def disable(self):
        self._openarm.disable_all()
        time.sleep(0.01)
        self._openarm.recv_all(0)

    def command(self, q, kp, kd, gripper=None, gripper_kp=0.0, gripper_kd=0.0):
        """
        MIT position command for joints 1-7 (and the gripper opening in m, if given).
        Every motor answers a command with its state, which the next read() takes in.
        """
        oa = self._oa
        self._openarm.get_arm().mit_control_all([
            oa.MITParam(float(kp[i]), float(kd[i]), float(q[i]), 0.0, 0.0) for i in range(7)
        ])
        if self.gripper and gripper is not None:
            self._openarm.get_gripper().mit_control_all([
                oa.MITParam(float(gripper_kp), float(gripper_kd),
                            float(opening_to_motor(gripper)), 0.0, 0.0)
            ])


class FakeCANBus:
    """
    Stand-in for OpenArmCANBus without hardware (--dry-run and tests): an enabled arm
    reaches each command by the next read. Set `faults` or `silent` to rehearse failures.
    """

    def __init__(self, interface, gripper=True, q=None, opening=0.0):
        self.interface = interface
        self.gripper = gripper
        self.enabled = False
        self.opened = False
        self.commands = 0
        self.faults = []  # reported by every read
        self.silent = False  # True: motors stop replying
        self._q = np.zeros(7) if q is None else np.array(q, dtype=float)
        self._opening = opening
        self._last_reply = time.monotonic()

    def open(self):
        self.opened = True

    def close(self):
        self.opened = False

    def request_state(self):
        pass

    def read(self):
        if not self.silent:
            self._last_reply = time.monotonic()
        age = time.monotonic() - self._last_reply
        return ArmState(
            q=self._q.copy(), dq=np.zeros(7), tau=np.zeros(7),
            gripper=self._opening if self.gripper else None,
            ages=np.full(8 if self.gripper else 7, age),
            faults=list(self.faults),
        )

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def command(self, q, kp, kd, gripper=None, gripper_kp=0.0, gripper_kd=0.0):
        self.commands += 1
        if not self.enabled:
            return
        self._q = np.array(q, dtype=float)
        if self.gripper and gripper is not None:
            self._opening = float(gripper)
