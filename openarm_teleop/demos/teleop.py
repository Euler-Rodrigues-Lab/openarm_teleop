"""Shared simulation loop for XR, MediaPipe, CSV, and NPZ replay."""
import argparse
from contextlib import ExitStack
import math
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['xr', 'mediapipe', 'replay'], default='xr')
    source = p.add_mutually_exclusive_group()
    source.add_argument('--frames', help='geo_kin_core frame-stream NPZ')
    source.add_argument('--csv', help='XRT body-pose CSV')
    p.add_argument('--no-human-overlay', '--no_human_overlay', action='store_true',
                   help='Hide the captured human skeleton in the viewer')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--dynamic', action='store_true',
                   help='Use actuator dynamics instead of directly posing the robot')
    p.add_argument('--steps', type=int, default=0, help='Control frames; 0 runs until stopped')
    p.add_argument('--playback-speed', type=float, default=1.0)
    p.add_argument('--loop', action='store_true')
    p.add_argument('--rate', type=float, default=60.0)
    p.add_argument('--retarget-mode', choices=['pose','tcp'], default='pose')
    p.add_argument('--no-safety-filter', action='store_true')
    p.add_argument('--no-functional-offset', action='store_true')
    p.add_argument('--limited', action='store_true')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--record', action='store_true')
    return p


def run(args):
    if (not math.isfinite(args.rate) or args.rate <= 0 or args.steps < 0
            or not math.isfinite(args.playback_speed) or args.playback_speed <= 0):
        raise ValueError('rate and playback speed must be finite and positive; steps must be nonnegative')
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
    controller = OpenArmMuJoCoController(model, data)
    controller.setup_mocap_body("base_mocap_mover")
    for side in ('right','left'):
        data.qpos[getattr(controller, f'{side}_arm_qpos_addrs')] = session.default_q[side]
    mujoco.mj_forward(model, data)
    controller._update_current_positions()
    controller.set_joint_goals({'q_goal_right':session.default_q['right'], 'q_goal_left':session.default_q['left']})
    session.reset(session.default_q['right'], session.default_q['left'])

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


def main():
    args = parser().parse_args()
    run(args)

if __name__ == '__main__':
    main()
