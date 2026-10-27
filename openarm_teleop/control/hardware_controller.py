"""
Hardware controller for the physical OpenArm, without ROS.

Drop-in for OpenArmMuJoCoController in the teleop loop (set_joint_goals, q_current_right/left,
the mocap and SEW-frame helpers). Goals are wrapped and clipped to the joint limits, and the
command sent to the motors moves toward them at no more than max_joint_vel, starting from the
measured pose. A background thread reads the motor states and sends the commands at control_hz;
the MuJoCo model only mirrors the measured robot for the viewer and the solver.

A fault (motor error code, CAN bus error, motors that stop replying, or a tracking error that
stays above its threshold) holds the arm at its measured pose until the session ends.
"""

import json
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from openarm_teleop.control.mujoco_controller import OpenArmMuJoCoController
from openarm_teleop.control.openarm_can_interface import (
    GRIPPER_OPEN_M,
    GRIPPER_OPEN_RAD,
    FakeCANBus,
    OpenArmCANBus,
)

SIDES = ("right", "left")

HOME_RAD = np.zeros(7)  # arm hanging straight down: ~0 gravity torque when the motors switch off

# openarm_hardware's default MIT gains (position mode), scaled by gain_scale
ARM_KP = np.array([70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0])
ARM_KD = np.array([2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5])
GRIPPER_KP = 5.0
GRIPPER_KD = 0.1

LIMIT_TOLERANCE = 0.1  # rad a measured joint may sit outside its limit (sag, calibration)
NEAR_HOME = 0.2  # rad; farther than this from home at start is reported
STATE_TIMEOUT = 0.5  # s without a reply from a motor before faulting
TRACKING_ERROR_DURATION = 0.3  # s the tracking error may stay above the threshold
# openarm_hardware's parallel-gripper scale: m of opening per rad of motor travel
GRIPPER_M_PER_MOTOR_RAD = GRIPPER_OPEN_M / abs(GRIPPER_OPEN_RAD)


class HardwareError(RuntimeError):
    """The arm is not ready to be commanded (pre-flight)."""


class JointCommandLimiter:
    """Clips joint goals to the joint limits and steps the command toward them at a bounded speed."""

    def __init__(self, lower, upper, max_vel):
        self.lower = lower
        self.upper = upper
        self.max_vel = max_vel

    def clip(self, q_goal):
        return np.clip(q_goal, self.lower, self.upper)

    def step(self, q_cmd, q_goal, dt):
        max_step = self.max_vel * dt
        return q_cmd + np.clip(q_goal - q_cmd, -max_step, max_step)


class OpenArmHardwareController:
    """
    buses: {side: OpenArmCANBus or FakeCANBus} for the arms to command. An arm that is not
    listed is never commanded; the mirror model simply shows its goal.
    """

    def __init__(self, buses, mujoco_model, mujoco_data, control_hz=250.0, max_joint_vel=1.5,
                 max_gripper_vel=0.1, joint_limit_margin=np.deg2rad(5.0), gain_scale=1.0,
                 max_tracking_error=0.35, command_grippers=True, gripper_max_opening=GRIPPER_OPEN_M,
                 gripper_close_overtravel=0.0):
        self.buses = dict(buses)
        self.arms = tuple(side for side in SIDES if side in self.buses)
        self.model = mujoco_model
        self.data = mujoco_data
        self.control_dt = 1.0 / control_hz
        self.max_tracking_error = max_tracking_error
        self.command_grippers = command_grippers
        self.gripper_close_overtravel = gripper_close_overtravel  # m past closed, to squeeze

        # Lower gains keep the damping ratio: kd scales with sqrt(kp)
        self.kp = ARM_KP * gain_scale
        self.kd = ARM_KD * np.sqrt(gain_scale)
        self.gripper_kp = GRIPPER_KP * gain_scale
        self.gripper_kd = GRIPPER_KD * np.sqrt(gain_scale)

        # Joint bookkeeping, mocap body and SEW frame of the mirror model
        self._mirror = OpenArmMuJoCoController(mujoco_model, mujoco_data)
        self._qpos_addrs = {
            "right": self._mirror.right_arm_qpos_addrs,
            "left": self._mirror.left_arm_qpos_addrs,
        }
        self._hand_qpos_addrs = {
            "right": self._mirror.right_hand_qpos_addrs,
            "left": self._mirror.left_hand_qpos_addrs,
        }

        # Limits of the bundled model (the mechanical stops), pulled in by the margin
        joint_ids = {
            "right": self._mirror.right_arm_joint_ids,
            "left": self._mirror.left_arm_joint_ids,
        }
        self.joint_limits = {}
        self._limiters = {}
        self._gripper_limiters = {}
        for side in SIDES:
            limits = self.model.jnt_range[joint_ids[side]]
            self.joint_limits[side] = (limits[:, 0].copy(), limits[:, 1].copy())
            self._limiters[side] = JointCommandLimiter(
                limits[:, 0] + joint_limit_margin, limits[:, 1] - joint_limit_margin, max_joint_vel
            )
            self._gripper_limiters[side] = JointCommandLimiter(
                -gripper_close_overtravel, gripper_max_opening, max_gripper_vel
            )

        self._lock = threading.Lock()
        self._q_measured = {side: None for side in SIDES}
        self._q_goal = {side: None for side in SIDES}
        self._q_cmd = {side: None for side in SIDES}
        self._g_measured = {side: None for side in SIDES}
        self._g_goal = {side: None for side in SIDES}
        self._g_cmd = {side: None for side in SIDES}

        self.fault = None  # text of the first fault, plus any later fault that escalated it
        self.fault_recoverable = False  # True: the arm can still be moved (tracking error only)
        self._over_limit_since = None
        self._stats = None  # tracking statistics, collected between start_stats and stop_stats
        self._stop = threading.Event()
        self._thread = None
        self._enabled = False

    def __getattr__(self, name):
        # Everything else (mocap body, SEW frame, joint addresses, q_current_*) is the mirror's
        if name == "_mirror":
            raise AttributeError(name)
        return getattr(self._mirror, name)

    # --- Lifecycle ---
    def connect(self, timeout=2.0):
        """
        Open the buses and read every motor, without enabling anything. Raises HardwareError
        if a motor does not reply, reports an error, or sits outside its joint limits.
        Returns a list of warnings.
        """
        for bus in self.buses.values():
            bus.open()

        states = {}
        deadline = time.monotonic() + timeout
        while True:
            for side, bus in self.buses.items():
                bus.request_state()
            time.sleep(0.02)
            states = {side: bus.read() for side, bus in self.buses.items()}
            if all(state.age < 0.2 for state in states.values()):
                break
            if time.monotonic() > deadline:
                silent = [
                    f"{side} arm ({self.buses[side].interface}): {', '.join(state.silent_motors(0.2))}"
                    for side, state in states.items() if state.age >= 0.2
                ]
                raise HardwareError("no reply from " + "; ".join(silent))

        warnings = []
        for side, state in states.items():
            if state.faults:
                raise HardwareError("; ".join(state.faults))
            lower, upper = self.joint_limits[side]
            if np.any(state.q < lower - LIMIT_TOLERANCE) or np.any(state.q > upper + LIMIT_TOLERANCE):
                raise HardwareError(
                    f"the {side} arm reads {np.round(np.rad2deg(state.q), 1)} deg, outside its "
                    "joint limits. Check the stored zero positions before continuing."
                )
            if np.max(np.abs(state.q - HOME_RAD)) > NEAR_HOME:
                warnings.append(f"the {side} arm is more than {NEAR_HOME} rad from home; "
                                "the ready-pose ramp starts from where it is")
            with self._lock:
                self._take_measurement(side, state)
                # Commands start from the measured pose
                self._q_cmd[side] = state.q.copy()
                self._g_cmd[side] = state.gripper
        self._update_current_positions()
        return warnings

    def start(self):
        """Enable the motors and start commanding (holding the measured pose until goals arrive)."""
        self._enabled = True
        for bus in self.buses.values():
            bus.enable()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="openarm-can", daemon=True)
        self._thread.start()

    def stop(self):
        """
        Stop commanding and switch the motors off where they are. Every bus is disabled and
        then closed even if one of them fails; the first failure is re-raised afterwards
        (with the others attached as notes on Python 3.11+). Safe to call more than once.
        """
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        errors = []
        if self._enabled:
            for side, bus in self.buses.items():
                try:
                    bus.disable()
                except Exception as e:  # noqa: BLE001 - keep shutting down the other buses
                    errors.append((side, "disable", e))
        self._enabled = False
        for side, bus in self.buses.items():
            try:
                bus.close()
            except Exception as e:  # noqa: BLE001
                errors.append((side, "close", e))
        if errors:
            side, step, first = errors[0]
            if hasattr(first, "add_note"):
                first.add_note(f"while shutting down the {side} arm ({step})")
                for side, step, e in errors[1:]:
                    first.add_note(f"also failed: {side} arm {step}: {e!r}")
            raise first

    # --- Goals (same interface as OpenArmMuJoCoController) ---
    def set_joint_goals(self, goals):
        """
        goals: 'q_goal_right', 'q_goal_left', and optionally 'right_gripper_val',
        'left_gripper_val' (hand opening from the input device). Ignored after a fault.
        """
        # Map hand opening to gripper opening with the sim controller (offset and 0-0.044 m range)
        gripper_vals = {
            key: goals[key]
            for key in ("right_gripper_val", "left_gripper_val")
            if goals.get(key) is not None
        }
        if gripper_vals:
            self._mirror.set_joint_goals(gripper_vals)

        with self._lock:
            if self.fault is not None:
                return
            for side in SIDES:
                q_goal = goals.get(f"q_goal_{side}")
                if q_goal is not None:
                    self._q_goal[side] = np.array(q_goal, dtype=float)
                if f"{side}_gripper_val" in gripper_vals:
                    opening = float(getattr(self._mirror, f"q_goal_{side}_hand")[0])
                    self._g_goal[side] = self._with_close_overtravel(side, opening)

    def _with_close_overtravel(self, side, opening):
        """Closed -> overtravel past closed, blending back to unchanged at the largest opening."""
        upper = self._gripper_limiters[side].upper
        if upper <= 0:
            return opening
        fraction_open = min(max(opening / upper, 0.0), 1.0)
        return opening - self.gripper_close_overtravel * (1.0 - fraction_open)

    def _limit_goal(self, side):
        # Wrap toward the current command, then clip to the joint limits
        q_goal = OpenArmMuJoCoController.real_angle(self._q_goal[side], self._q_cmd[side])
        return self._limiters[side].clip(q_goal)

    def goals_reached(self, tol=1e-3):
        """True once the commands to the selected arms (and grippers) have reached their goals."""
        with self._lock:
            for side in self.arms:
                if self._q_goal[side] is None or self._q_cmd[side] is None:
                    return False
                if np.max(np.abs(self._limit_goal(side) - self._q_cmd[side])) > tol:
                    return False
                if self.command_grippers and self._g_goal[side] is not None:
                    g_goal = self._gripper_limiters[side].clip(self._g_goal[side])
                    if self._g_cmd[side] is None or abs(g_goal - self._g_cmd[side]) > 1e-4:
                        return False
        return True

    def seconds_to_goals(self):
        """Time the speed-limited arm commands need to reach their goals."""
        with self._lock:
            seconds = [
                np.max(np.abs(self._limit_goal(side) - self._q_cmd[side])
                       / self._limiters[side].max_vel)
                for side in self.arms
                if self._q_goal[side] is not None and self._q_cmd[side] is not None
            ]
        return max(seconds, default=0.0)

    def tracking_error(self):
        """Largest |measured - commanded| arm joint error over the selected arms, rad."""
        with self._lock:
            return self._tracking_error()

    def _tracking_error(self):
        errors = [
            np.max(np.abs(self._q_measured[side] - self._q_cmd[side]))
            for side in self.arms
            if self._q_measured[side] is not None and self._q_cmd[side] is not None
        ]
        return max(errors, default=0.0)

    def hold_at_measured(self):
        """Stop pushing: make the measured pose both the goal and the command."""
        with self._lock:
            self._hold_at_measured()

    def _hold_at_measured(self):
        for side in self.arms:
            if self._q_measured[side] is not None:
                self._q_goal[side] = self._q_measured[side].copy()
                self._q_cmd[side] = self._q_measured[side].copy()
            if self._g_measured[side] is not None:
                self._g_goal[side] = self._g_measured[side]
                self._g_cmd[side] = self._g_measured[side]

    def clear_fault(self):
        """Accept goals again after a recoverable fault (to return home)."""
        with self._lock:
            self.fault = None
            self._over_limit_since = None

    # --- Mirror model ---
    def _update_current_positions(self):
        """Show the measured robot in the mirror model. Call from the thread that owns the viewer."""
        with self._lock:
            for side in SIDES:
                # An arm that is not commanded is shown at its goal
                q = self._q_measured[side] if side in self.arms else self._q_goal[side]
                if q is not None:
                    self.data.qpos[self._qpos_addrs[side]] = q
                if self._g_measured[side] is not None:
                    # Both fingers follow the gripper opening, as on the robot
                    opening = np.clip(self._g_measured[side], 0.0, GRIPPER_OPEN_M)
                    self.data.qpos[self._hand_qpos_addrs[side]] = opening
        mujoco.mj_forward(self.model, self.data)
        self._mirror._update_current_positions()

    # --- Control thread ---
    def _run(self):
        period = self.control_dt
        next_time = last = time.monotonic()
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                # Step by the time that really passed, so the speed limit holds under jitter
                dt = min(max(now - last, 0.0), 2.0 * period) if now > last else period
                if self._stats is not None and now > last:
                    self._stats["periods"].append(now - last)
                last = now
                self._step(dt)

                next_time += period
                delay = next_time - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_time = time.monotonic()  # overran: do not try to catch up
        except Exception as e:  # e.g. the CAN interface went away
            with self._lock:
                self._record_fault(f"control loop stopped: {e}", recoverable=False)

    def _record_fault(self, text, recoverable):
        """
        Record a fault (under the lock). The first fault is kept; a later non-recoverable
        fault escalates a recoverable one so the exit path does not try to move the arm.
        Anything else is ignored.
        """
        if self.fault is None:
            self.fault = text
        elif self.fault_recoverable and not recoverable:
            self.fault = f"{self.fault}; then {text}"
        else:
            return
        self.fault_recoverable = recoverable
        self._hold_at_measured()

    def _take_measurement(self, side, state):
        self._q_measured[side] = state.q
        if state.gripper is not None:
            self._g_measured[side] = state.gripper

    def _step(self, dt):
        """Read every arm, check for faults, then step and send the commands."""
        states = {side: self.buses[side].read() for side in self.arms}

        commands = {}
        with self._lock:
            problems = []
            for side, state in states.items():
                problems += state.faults
                if state.age > STATE_TIMEOUT:
                    silent = ", ".join(state.silent_motors(STATE_TIMEOUT))
                    problems.append(f"no state from the {side} arm ({silent}) for {STATE_TIMEOUT} s")
                else:
                    self._take_measurement(side, state)

            recoverable = False
            error = self._tracking_error()
            if not problems and self.max_tracking_error > 0 and error > self.max_tracking_error:
                now = time.monotonic()
                if self._over_limit_since is None:
                    self._over_limit_since = now
                if now - self._over_limit_since > TRACKING_ERROR_DURATION:
                    problems.append(f"tracking error {error:.2f} rad above "
                                    f"{self.max_tracking_error:.2f} rad")
                    recoverable = True
            else:
                self._over_limit_since = None

            if problems:
                self._record_fault("; ".join(problems), recoverable)

            for side in self.arms:
                before = self._q_cmd[side]
                if self._q_goal[side] is not None:
                    self._q_cmd[side] = self._limiters[side].step(
                        before, self._limit_goal(side), dt
                    )
                g_cmd = None
                if self.command_grippers and self._g_cmd[side] is not None:
                    if self._g_goal[side] is not None:
                        limiter = self._gripper_limiters[side]
                        self._g_cmd[side] = float(
                            limiter.step(self._g_cmd[side], limiter.clip(self._g_goal[side]), dt)
                        )
                    g_cmd = self._g_cmd[side]
                commands[side] = (self._q_cmd[side].copy(), g_cmd)
                self._collect_stats(side, before, dt)

        for side, (q_cmd, g_cmd) in commands.items():
            self.buses[side].command(q_cmd, self.kp, self.kd, g_cmd, self.gripper_kp, self.gripper_kd)

    # --- Tracking statistics ---
    def _collect_stats(self, side, q_cmd_before, dt):
        if self._stats is None or self._q_measured[side] is None:
            return
        # A joint is speed-limited when its command moved by the full allowed step
        max_step = self._limiters[side].max_vel * dt
        self._stats["limited"][side] += np.abs(self._q_cmd[side] - q_cmd_before) >= 0.999 * max_step
        self._stats["samples"][side] += 1
        self._stats["errors"].append(np.max(np.abs(self._q_measured[side] - self._q_cmd[side])))
        if self._stats["record"] is not None and self._q_goal[side] is not None:
            # The goal as asked (wrapped like the command, but not clipped)
            goal = OpenArmMuJoCoController.real_angle(self._q_goal[side], self._q_cmd[side])
            self._stats["record"][side].append(np.concatenate((
                [time.monotonic() - self._stats["start"]],
                goal, self._q_cmd[side], self._q_measured[side],
            )))

    def start_stats(self, record=False):
        """
        Collect speed-limit, tracking-error and loop-timing statistics until stop_stats(). With
        record=True also keep every step's goal, command and measured positions for a log file.
        """
        with self._lock:
            self._stats = {
                "start": time.monotonic(),
                "samples": {side: 0 for side in self.arms},
                "limited": {side: np.zeros(7) for side in self.arms},
                "errors": [],
                "periods": [],
                "record": {side: [] for side in self.arms} if record else None,
            }

    def stop_stats(self, log_path=None, metadata=None):
        """
        Stop collecting and return a printable summary (None if nothing was collected).
        With log_path, also save the recorded steps there as an .npz.
        """
        with self._lock:
            stats, self._stats = self._stats, None
        if not stats or not stats["errors"]:
            return None

        lines = [f"Tracking summary ({time.monotonic() - stats['start']:.0f} s):"]
        for side in self.arms:
            samples = stats["samples"][side]
            if samples:
                share = "  ".join(
                    f"J{i + 1} {100 * count / samples:3.0f}%"
                    for i, count in enumerate(stats["limited"][side])
                )
                lines.append(f"  {side:5s} arm at its speed limit: {share}")
        errors = np.array(stats["errors"])
        threshold = f" (fault at {self.max_tracking_error:.2f})" if self.max_tracking_error > 0 else ""
        lines.append(f"  tracking error mean {errors.mean():.3f}, 95% {np.percentile(errors, 95):.3f}, "
                     f"max {errors.max():.3f} rad{threshold}")
        if stats["periods"]:
            periods = 1000.0 * np.array(stats["periods"])
            lines.append(f"  command period mean {periods.mean():.2f}, 99% {np.percentile(periods, 99):.2f}, "
                         f"max {periods.max():.2f} ms (nominal {1000.0 * self.control_dt:.2f})")

        if log_path is not None and stats["record"] is not None:
            metadata = dict(metadata or {})
            metadata["max_joint_vel"] = float(np.max(self._limiters[self.arms[0]].max_vel))
            data = {"metadata": np.array(json.dumps(metadata))}
            for side, rows in stats["record"].items():
                if rows:
                    rows = np.array(rows)
                    data[f"{side}_t"] = rows[:, 0]
                    data[f"{side}_goal"] = rows[:, 1:8]
                    data[f"{side}_cmd"] = rows[:, 8:15]
                    data[f"{side}_measured"] = rows[:, 15:22]
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(log_path, **data)
            lines.append(f"  log saved to {log_path}")
        return "\n".join(lines)


def make_hardware_controller(args, mujoco_model, mujoco_data):
    """Controller for the arms selected on the command line (FakeCANBus with --dry-run)."""
    arms = SIDES if args.arms == "both" else (args.arms,)
    interfaces = {"right": args.right_can, "left": args.left_can}
    command_grippers = not args.no_grippers
    bus_type = FakeCANBus if args.dry_run else OpenArmCANBus
    buses = {side: bus_type(interfaces[side], gripper=command_grippers) for side in arms}
    return OpenArmHardwareController(
        buses, mujoco_model, mujoco_data,
        control_hz=args.control_hz,
        max_joint_vel=args.max_joint_vel,
        max_gripper_vel=args.max_gripper_vel,
        joint_limit_margin=np.deg2rad(args.joint_limit_margin_deg),
        gain_scale=args.gain_scale,
        max_tracking_error=args.max_tracking_error,
        command_grippers=command_grippers,
        gripper_close_overtravel=np.deg2rad(args.gripper_close_overtravel_deg) * GRIPPER_M_PER_MOTOR_RAD,
    )
