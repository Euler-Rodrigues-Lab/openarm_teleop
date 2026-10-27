#!/usr/bin/env python3
"""
Zero-position calibration of one OpenArm v1 arm on a classic CAN 2.0 bus.

The motion sequence, the stop detection (bump_to_limit) and every constant are imported
unchanged from openarm_can's own script, external/openarm_can/setup/
openarm-can-zero-position-calibration. Only its run_calibration() is replaced here,
because it opens the bus in CAN-FD mode.

    uv run scripts/calibrate_zero.py --arm-side right_arm           # can0
    uv run scripts/calibrate_zero.py --arm-side left_arm            # can1
    uv run scripts/calibrate_zero.py --arm-side right_arm --set-zero-only

The default drives every joint into its mechanical stops, backs off by the nominal stop
angles and stores the pose it ends in as zero. bump_to_limit has no travel or time limit:
keep the e-stop in hand and the workspace clear. With --set-zero-only nothing moves: pose
the arm at zero by hand (hanging straight down, gripper closed) and the current position
of every motor is stored as zero.
"""

import argparse
import importlib.machinery
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

UPSTREAM_SCRIPT = (Path(__file__).resolve().parents[1]
                   / "external/openarm_can/setup/openarm-can-zero-position-calibration")
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)] + ["gripper"]


def load_upstream():
    """Import openarm_can's calibration script as a module (it has no .py suffix)."""
    if not UPSTREAM_SCRIPT.exists():
        raise SystemExit(f"Missing {UPSTREAM_SCRIPT}: run `git submodule update --init`.")
    name = "openarm_can_zero_position_calibration"
    loader = importlib.machinery.SourceFileLoader(name, str(UPSTREAM_SCRIPT))
    spec = importlib.util.spec_from_file_location(name, str(UPSTREAM_SCRIPT), loader=loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Do not leave a __pycache__ in the openarm_can submodule
    dont_write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write_bytecode
    return module


def open_arm(oa, canport):
    """The same motors as upstream's run_calibration, on a classic CAN 2.0 socket."""
    openarm = oa.OpenArm(canport, False)
    openarm.init_arm_motors(
        [oa.MotorType.DM8009, oa.MotorType.DM8009, oa.MotorType.DM4340, oa.MotorType.DM4340,
         oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310],
        [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07],
        [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
    )
    openarm.init_gripper_motor(oa.MotorType.DM4310, 0x08, 0x18)
    openarm.set_callback_mode_all(oa.CallbackMode.STATE)
    return openarm


def read_positions(openarm):
    """Positions of joints 1-7 and the gripper in rad, or None if a motor does not reply."""
    for _ in range(5):
        openarm.refresh_all()
        time.sleep(0.02)
        openarm.recv_all(2000)
    arm, grip = openarm.get_arm(), openarm.get_gripper()
    replied = all(arm.get_link_stats(i).seconds_since_response() < 0.5 for i in range(7))
    if not replied or grip.get_link_stats(0).seconds_since_response() > 0.5:
        return None
    return [m.get_position() for m in arm.get_motors() + grip.get_motors()]


def print_positions(title, positions):
    print(title)
    for name, q in zip(JOINT_NAMES, positions):
        print(f"  {name:8s} {np.rad2deg(q):+8.2f} deg")


def set_zero(openarm):
    """Store the current position of every motor as zero (as upstream: disabled first)."""
    openarm.disable_all()
    openarm.recv_all()
    openarm.set_zero_all()
    openarm.recv_all()
    print("wrote zero position to arm")


def run_calibration(upstream, openarm, arm_side, robot_version):
    """upstream.run_calibration on an already opened arm: hold, run the side's sequence, set zero."""
    oa = upstream.oa
    if robot_version == 'v2':
        mech_lim = upstream.MECH_LIM_V2_RIGHT if arm_side == 'right_arm' else upstream.MECH_LIM_V2_LEFT
    else:
        mech_lim = upstream.MECH_LIM_V1

    print("Enabling...")
    openarm.enable_all()
    time.sleep(0.1)
    print("Enabled...")

    # Hold current pose (light PD)
    openarm.refresh_all()
    time.sleep(0.01)
    openarm.recv_all(2000)
    arm = openarm.get_arm()
    grip = openarm.get_gripper()
    initial_arm_q = [m.get_position() for m in arm.get_motors()]
    initial_grip_q = [m.get_position() for m in grip.get_motors()]

    arm_params = [oa.MITParam(kp, kd, q, 0.0, 0.0)
                  for kp, kd, q in zip([300, 300, 150, 150, 40, 40, 30],
                                       [2.5, 2.5, 2.5, 2.5, 0.8, 0.8, 0.8],
                                       initial_arm_q)]
    grip_params = [oa.MITParam(10.0, 0.9, initial_grip_q[0], 0.0, 0.0)]
    arm.mit_control_all(arm_params)
    grip.mit_control_all(grip_params)
    openarm.recv_all()

    try:
        if arm_side == 'right_arm':
            upstream._run_right_sequence(openarm, arm, grip, mech_lim)
        else:
            upstream._run_left_sequence(openarm, arm, grip, mech_lim, robot_version)
        set_zero(openarm)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C pressed -> stopping safely")
    finally:
        openarm.disable_all()
        print("[INFO] Motors disabled, exiting safely.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--canport', type=str, default=None,
                        help='SocketCAN interface (default: can0 for the right arm, can1 for the left)')
    parser.add_argument('--arm-side', type=str, required=True, choices=['right_arm', 'left_arm'])
    parser.add_argument('--robot-version', type=str, default='v1', choices=['v1', 'v2'])
    parser.add_argument('--set-zero-only', action='store_true',
                        help='Do not move: store the current position of every motor as zero')
    parser.add_argument('--yes', action='store_true', help='Do not ask for confirmation')
    args = parser.parse_args()
    if args.canport is None:
        args.canport = 'can1' if args.arm_side == 'left_arm' else 'can0'

    upstream = load_upstream()
    openarm = open_arm(upstream.oa, args.canport)
    before = read_positions(openarm)
    if before is None:
        raise SystemExit(f"Error: not every motor on {args.canport} replies. Is the bus up in "
                         "classic CAN 2.0 mode at 1 Mbps, and the arm powered?")
    print_positions(f"{args.arm_side} on {args.canport} (classic CAN), stored zeros give:", before)

    if args.set_zero_only:
        question = ("Store the CURRENT position of every motor as zero? The arm must hang "
                    "straight down with the gripper closed.")
    else:
        question = (f"Run the {args.robot_version} {args.arm_side} calibration? The arm drives every "
                    "joint into its mechanical stops. E-stop in hand, workspace clear.")
    if not args.yes and input(f"\n{question} [y/N] ").strip().lower() != 'y':
        print("Nothing changed.")
        return

    if args.set_zero_only:
        set_zero(openarm)
    else:
        run_calibration(upstream, openarm, args.arm_side, args.robot_version)

    time.sleep(0.2)
    after = read_positions(openarm)
    if after is not None:
        print_positions("After calibration:", after)


if __name__ == "__main__":
    main()
