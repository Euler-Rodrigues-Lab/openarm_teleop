"""Shared simulation and hardware loop for XR, MediaPipe, CSV, and NPZ replay."""
import argparse
from contextlib import ExitStack
import math
import signal
import sys
import time

VIEWER_FPS = 60.0  # viewer refresh on hardware, where the solver loop is not rate limited


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', 
                   choices=['xr', 'mediapipe', 'replay'], 
                   default='replay'
                   )
    source = p.add_mutually_exclusive_group()
    source.add_argument('--frames', help='geo_kin_core frame-stream NPZ')
    source.add_argument('--csv', help='XRT body-pose CSV')
    p.add_argument('--hardware', 
                   default=False,
                   action=argparse.BooleanOptionalAction,
                   help='Use real robot hardware, default to False (simulation only)'
                   )
    hw = p.add_argument_group('hardware', 'Only used with --hardware (openarm_can, classic CAN 2.0)')
    hw.add_argument('--dry-run',
                    action='store_true',
                    help='Rehearse the hardware sequence on a fake CAN bus (no motors)'
                    )
    hw.add_argument('--arms',
                    choices=['right', 'left', 'both'],
                    default='both',
                    help='Arms to command'
                    )
    hw.add_argument('--right-can',
                    default='can0',
                    help='SocketCAN interface of the right arm'
                    )
    hw.add_argument('--left-can',
                    default='can1',
                    help='SocketCAN interface of the left arm'
                    )
    hw.add_argument('--control-hz',
                    type=float,
                    default=250.0,
                    help='Rate of motor commands; classic CAN at 1 Mbps saturates near 500'
                    )
    hw.add_argument('--gain-scale',
                    type=float,
                    default=0.5,
                    help='Scale of the stock MIT position gains (kd scales with its square root)'
                    )
    hw.add_argument('--max-joint-vel',
                    type=float,
                    default=0.5,
                    help='Joint speed limit in rad/s'
                    )
    hw.add_argument('--joint-limit-margin-deg',
                    type=float,
                    default=5.0,
                    help='Keep commands this far inside the joint limits'
                    )
    hw.add_argument('--max-tracking-error',
                    type=float,
                    default=0.35,
                    help='Fault above this tracking error in rad (0 disables)'
                    )
    hw.add_argument('--no-grippers',
                    action='store_true',
                    help='Leave the grippers alone'
                    )
    hw.add_argument('--max-gripper-vel',
                    type=float,
                    default=0.1,
                    help='Gripper speed limit in m/s'
                    )
    hw.add_argument('--gripper-close-overtravel-deg',
                    type=float,
                    default=5.0,
                    help='Command a closed hand this many motor degrees past closed, to squeeze (0: off)'
                    )
    hw.add_argument('--log',
                    default='logs/teleop.npz',
                    help='Save goal/command/measured traces while tracking to this .npz'
                    )
    p.add_argument('--no-human-overlay', 
                   '--no_human_overlay',
                   action='store_true',
                   help='Hide the captured human skeleton in the viewer'
                   )
    p.add_argument('--headless', 
                   action='store_true'
                   )
    p.add_argument('--dynamic', 
                   action='store_true',
                   help='Use actuator dynamics instead of directly posing the robot')
    p.add_argument('--steps', 
                   type=int, 
                   default=0, 
                   help='Control frames; 0 runs until stopped'
                   )
    p.add_argument('--playback-speed', 
                   type=float,
                   default=1.0
                   )
    p.add_argument('--loop', 
                   action='store_true'
                   )
    p.add_argument('--rate', 
                   type=float, 
                   default=None,
                   help='Control frames per second (default: 60 in simulation, unlimited on hardware)'
                   )
    p.add_argument('--retarget-mode',
                   choices=['pose','tcp'], 
                   default='pose'
                   )
    p.add_argument('--no-safety-filter', 
                   action='store_true'
                   )
    p.add_argument('--no-functional-offset', 
                   action='store_true'
                   )
    p.add_argument('--limited',
                   action='store_true'
                   )
    p.add_argument('--host',
                   default='0.0.0.0'
                   )
    p.add_argument('--port',
                   type=int,
                   default=8080
                   )
    p.add_argument('--camera',
                   type=int,
                   default=0
                   )
    p.add_argument('--record',
                   action='store_true'
                   )
    return p


def run(args):
    if args.rate is None and not args.hardware:
        args.rate = 60.0
    if ((args.rate is not None and (not math.isfinite(args.rate) or args.rate <= 0)) or args.steps < 0
            or not math.isfinite(args.playback_speed) or args.playback_speed <= 0):
        raise ValueError('rate and playback speed must be finite and positive; steps must be nonnegative')
    if args.hardware:
        if args.dynamic:
            raise ValueError('--dynamic simulates actuators; it cannot be combined with --hardware')
        if args.no_safety_filter and not args.dry_run:
            raise ValueError('refusing --no-safety-filter on real hardware: nothing else keeps the arms apart')
        if not (args.control_hz > 0 and args.gain_scale > 0 and args.max_joint_vel > 0
                and args.max_gripper_vel > 0 and args.joint_limit_margin_deg >= 0):
            raise ValueError('control rate, gain scale and speed limits must be positive')
    if args.device == 'replay' and not (args.frames or args.csv):
        raise ValueError('replay requires --frames or --csv')
    if args.device != 'replay' and (args.frames or args.csv):
        raise ValueError('--frames and --csv require --device replay')
    import numpy as np
    import mujoco
    from openarm_teleop import scene_path
    from openarm_teleop.session import make_session
    from openarm_teleop.control.mujoco_controller import OpenArmMuJoCoController
    from geo_kin_core.types import RetargetFrame
    from xrt_devices.integrations.geo_kin import XRDeviceAdapter, MediaPipeDeviceAdapter, open_motion_source

    # Resolve the solver before starting any device. A missing/wrong license
    # cannot leave a camera or WebRTC process running.
    session = make_session(collision_avoidance=not args.no_safety_filter,
                           retarget_mode=args.retarget_mode, limited=args.limited,
                           functional_offset=not args.no_functional_offset)
    model = mujoco.MjModel.from_xml_path(str(scene_path()))
    data = mujoco.MjData(model)
    if args.hardware:
        from openarm_teleop.control.hardware_controller import make_hardware_controller
        controller = make_hardware_controller(args, model, data)
    else:
        controller = OpenArmMuJoCoController(model, data)
    controller.setup_mocap_body("base_mocap_mover")
    for side in ('right','left'):
        data.qpos[getattr(controller, f'{side}_arm_qpos_addrs')] = session.default_q[side]
    mujoco.mj_forward(model, data)
    controller._update_current_positions()
    controller.set_joint_goals({'q_goal_right':session.default_q['right'], 'q_goal_left':session.default_q['left']})
    session.reset(session.default_q['right'], session.default_q['left'])
    if args.hardware:
        # Read-only pre-flight, before any device starts: nothing is enabled yet.
        _hardware_preflight(args, controller)

    with ExitStack() as stack:
        if args.device == 'xr':
            source = XRDeviceAdapter(host=args.host, port=args.port, record_data=args.record)
            stack.callback(source.cleanup)
        elif args.device == 'mediapipe':
            source = MediaPipeDeviceAdapter(camera_id=args.camera)
            stack.callback(source.cleanup)
        else:
            source = open_motion_source(frames=args.frames, csv_file=args.csv, loop=args.loop,
                                        playback_speed=args.playback_speed)
            print(f"Motion source: {source.describe()}")
        viewer = None
        overlay = None
        if not args.headless:
            import mujoco.viewer
            from openarm_teleop.visualization import ReplayOverlay, configure_camera
            viewer = stack.enter_context(mujoco.viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False))
            with viewer.lock():
                configure_camera(viewer.cam)
            overlay = ReplayOverlay(viewer, show_human=not args.no_human_overlay,
                                    show_safety=not args.no_safety_filter)
        print('OpenArm backend: licensed Rust; model: bundled v1 MJCF')
        if args.hardware:
            # Registered last, so the motors are switched off before the device is closed.
            stack.callback(controller.stop)
            return _run_hardware(args, session, controller, source, viewer, overlay, data)
        count = 0
        start = time.monotonic()
        try:
            while (args.steps == 0 or count < args.steps) and (viewer is None or viewer.is_running()):
                tick = time.monotonic()
                elapsed = count / args.rate if args.device == 'replay' else tick-start
                if args.device == 'replay' and not args.loop and elapsed * args.playback_speed > source.duration:
                    break
                frame = source.frame_at_time(elapsed) if args.device == 'replay' else source.get_frame()
                controller._update_current_positions()
                out = session.solve(frame if frame is not None else RetargetFrame(), engaged=frame is not None,
                                    q_current_right=controller.q_current_right,
                                    q_current_left=controller.q_current_left)
                controller.set_joint_goals({name:getattr(out,name,None) for name in
                                           ('q_goal_right','q_goal_left','left_gripper_val','right_gripper_val')})
                if out.p_world_base is not None and out.R_world_base is not None:
                    controller.update_mocap_body(out.p_world_base, out.R_world_base)
                target_time = (count+1)/args.rate
                if args.dynamic:
                    # Preserve time across non-integer control/physics ratios.
                    while data.time + 1e-12 < target_time:
                        controller.update_position_control()
                        mujoco.mj_step(model,data)
                else:
                    data.time = target_time
                    controller.update_kinematic()
                if not np.isfinite(data.qpos).all():
                    raise RuntimeError('non-finite simulation state')
                if viewer is not None:
                    if overlay is not None:
                        with viewer.lock():
                            overlay.draw(frame, session, to_world=controller.get_sew_transform())
                    viewer.sync()
                count += 1
                if args.device != 'replay' or viewer is not None:
                    time.sleep(max(0.0,1/args.rate-(time.monotonic()-tick)))
        except KeyboardInterrupt:
            pass
        return count


class _StopSession(Exception):
    """Raised when the viewer window is closed."""


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def _hardware_preflight(args, controller):
    from openarm_teleop.control.hardware_controller import HardwareError
    buses = ', '.join(f'{side} on {bus.interface}' for side, bus in controller.buses.items())
    print(f"Pre-flight checks ({'DRY RUN, fake CAN bus' if args.dry_run else buses})...")
    try:
        warnings = controller.connect()
    except (HardwareError, RuntimeError) as e:
        try:
            controller.stop()
        except Exception as stop_error:  # noqa: BLE001 - the pre-flight error is the one to report
            print(f'Warning: shutdown after the failed pre-flight also failed: {stop_error}')
        raise SystemExit(f'Error: {e}')
    limits = f'{args.joint_limit_margin_deg:.1f} deg inside the limits'
    grippers = (f'on (closing {args.gripper_close_overtravel_deg:g} deg past closed)'
                if controller.command_grippers else 'off')
    error = 'off' if args.max_tracking_error <= 0 else f'> {args.max_tracking_error:.2f} rad'
    print(f"Commanding: {', '.join(controller.arms)} arm(s) at {args.control_hz:.0f} Hz, "
          f"<= {args.max_joint_vel:.2f} rad/s, joint limits {limits}, gain scale {args.gain_scale:g}, "
          f"self-collision filter {'OFF' if args.no_safety_filter else 'on'}, grippers {grippers}, "
          f"tracking-error fault {error}")
    if not args.dry_run:
        print('\nMoving the REAL OpenArm. E-stop in hand, workspace clear, second person watching.')
    for warning in warnings:
        print(f'Warning: {warning}.')


def _run_hardware(args, session, controller, source, viewer, overlay, data):
    """Ready pose, tracking under supervision, then home before the motors are switched off."""
    from geo_kin_core.types import RetargetFrame
    from openarm_teleop.control.hardware_controller import HOME_RAD, SIDES

    last_sync = [0.0]

    def refresh(frame=None):
        """Mirror the measured robot; closing the viewer ends the session."""
        controller._update_current_positions()
        if viewer is None:
            return
        if not viewer.is_running():
            raise _StopSession
        if time.monotonic() - last_sync[0] >= 1 / VIEWER_FPS:
            if overlay is not None:
                with viewer.lock():
                    overlay.draw(frame, session, to_world=controller.get_sew_transform())
            viewer.sync()
            last_sync[0] = time.monotonic()

    def wait_for(condition):
        while not condition():
            refresh()
            time.sleep(0.005)

    def report_stats():
        summary = controller.stop_stats(log_path=args.log, metadata={
            'frames': str(args.frames), 'csv': str(args.csv), 'playback_speed': args.playback_speed})
        if summary:
            print(summary)

    # Ctrl-C and SIGTERM both end tracking and send the arm home. Set SIGINT explicitly:
    # a shell starts background jobs with it ignored.
    handlers = {}
    try:
        handlers[signal.SIGINT] = signal.signal(signal.SIGINT, signal.default_int_handler)
        handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    except ValueError:
        pass  # not the main thread
    # The solver loop is not rate limited; hand the interpreter to the command thread promptly.
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(min(switch_interval, 0.0005))

    count = 0
    try:
        try:
            controller.start()
            controller.set_joint_goals({f'q_goal_{side}': session.default_q[side] for side in SIDES})
            print('\nMoving to the ready pose (Ctrl-C to stop)...')
            wait_for(lambda: controller.goals_reached() or controller.fault is not None)

            if controller.fault is None and not getattr(source, 'is_connected', True):
                print('Ready pose reached. Waiting for the input device to connect...')
                wait_for(lambda: source.is_connected or controller.fault is not None)

            if controller.fault is None:
                controller.start_stats(record=args.log is not None)
                print('Tracking.')
            start = time.monotonic()
            while controller.fault is None and (args.steps == 0 or count < args.steps):
                tick = time.monotonic()
                elapsed = tick - start
                if args.device == 'replay' and not args.loop and elapsed * args.playback_speed > source.duration:
                    print('Replay finished.')
                    break
                frame = source.frame_at_time(elapsed) if args.device == 'replay' else source.get_frame()
                controller._update_current_positions()
                # Without a frame the goals stay as they are: the arm holds, and the speed
                # limit ramps it back in when tracking returns.
                out = session.solve(frame if frame is not None else RetargetFrame(), engaged=frame is not None,
                                    q_current_right=controller.q_current_right,
                                    q_current_left=controller.q_current_left)
                controller.set_joint_goals({name: getattr(out, name, None) for name in
                                            ('q_goal_right', 'q_goal_left', 'left_gripper_val', 'right_gripper_val')})
                if out.p_world_base is not None and out.R_world_base is not None:
                    controller.update_mocap_body(out.p_world_base, out.R_world_base)
                data.time = elapsed
                refresh(frame)
                count += 1
                # Always yield to the command thread, even without a rate limit
                period = 1 / args.rate if args.rate is not None else 0.0
                time.sleep(max(0.001, period - (time.monotonic() - tick)))
            report_stats()

            if controller.fault is not None:
                print(f'FAULT: {controller.fault}. Holding at the measured pose.')
                if controller.fault_recoverable:
                    print('Ctrl-C to return home (twice to switch the motors off where they are).')
                else:
                    print('Support the arm, then Ctrl-C: the motors are switched off where they are.')
                wait_for(lambda: False)
        except (KeyboardInterrupt, _StopSession):
            print('\nStopping.')
        finally:
            report_stats()  # before homing, which is not part of tracking

        if controller.fault is not None and not controller.fault_recoverable:
            print('Not returning home after this fault. Switching the motors off.')
            return count
        # Ramp home so the motors can be switched off safely
        controller.clear_fault()
        goals = {f'q_goal_{side}': HOME_RAD for side in SIDES}
        if controller.command_grippers:
            goals.update({f'{side}_gripper_val': 0.0 for side in SIDES})  # closed
        controller.set_joint_goals(goals)
        deadline = time.monotonic() + controller.seconds_to_goals() + 3.0
        print('Returning home (Ctrl-C again to switch the motors off where they are)...')
        try:
            while (not controller.goals_reached() and controller.fault is None
                   and time.monotonic() < deadline):
                controller._update_current_positions()
                if viewer is not None and viewer.is_running():
                    viewer.sync()
                time.sleep(0.005)
            if controller.goals_reached():
                time.sleep(0.5)  # settle
                print('At home. Switching the motors off.')
            else:
                print('Warning: home was not reached; switching the motors off where they are.')
        except KeyboardInterrupt:
            print('\nHoming skipped; switching the motors off where they are.')
        return count
    finally:
        sys.setswitchinterval(switch_interval)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def main():
    args = parser().parse_args()
    if args.device == 'replay' and not (args.frames or args.csv):
        from openarm_teleop import SAMPLE_MOTIONS
        if args.hardware:
            parser().error('hardware replay requires --frames/--csv; use replay_offline for bundled hardware replay')
        args.frames = str(SAMPLE_MOTIONS['ipman_roll'])
    run(args)

if __name__ == '__main__':
    main()
