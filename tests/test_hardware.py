import time
import numpy as np
import mujoco
import pytest
from openarm_teleop import scene_path
from openarm_teleop.control.hardware_controller import HardwareError, OpenArmHardwareController
from openarm_teleop.control.openarm_can_interface import FakeCANBus, motor_to_opening, opening_to_motor
from openarm_teleop.demos.replay_offline import parse_args

READY = np.array([0., 0., 0., np.pi / 2, 0., 0., 0.])


def make(arms=('right', 'left'), buses=None, **kwargs):
    model = mujoco.MjModel.from_xml_path(str(scene_path()))
    data = mujoco.MjData(model)
    if buses is None:
        buses = {side: FakeCANBus(f'fake_{side}') for side in arms}
    kwargs.setdefault('control_hz', 500.0)
    return OpenArmHardwareController(buses, model, data, **kwargs), buses


def wait(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.005)
    return condition()


def test_gripper_mapping():
    assert opening_to_motor(0.044) == pytest.approx(-1.0472)
    assert opening_to_motor(0.0) == 0.0
    assert motor_to_opening(opening_to_motor(0.02)) == pytest.approx(0.02)


def test_ramp_respects_speed_and_joint_limits():
    controller, buses = make(max_joint_vel=4.0, gripper_close_overtravel=0.002)
    assert controller.connect() == []
    controller.start()
    try:
        assert all(bus.enabled for bus in buses.values())
        start = time.monotonic()
        # J4 beyond its upper limit, and a hand closed past the gripper range
        goal = READY.copy(); goal[3] = 3.0
        controller.set_joint_goals(dict(q_goal_right=goal, q_goal_left=READY,
                                        right_gripper_val=0.0, left_gripper_val=0.1))
        assert wait(controller.goals_reached)
        elapsed = time.monotonic() - start
        upper = controller.joint_limits['right'][1][3] - np.deg2rad(5.0)
        assert elapsed >= 0.9 * upper / 4.0  # no faster than the speed limit
        np.testing.assert_allclose(buses['right'].read().q[3], upper, atol=1e-3)
        np.testing.assert_allclose(buses['left'].read().q, READY, atol=1e-3)
        assert buses['right'].read().gripper == pytest.approx(-0.002, abs=1e-4)  # squeeze
        assert buses['left'].read().gripper == pytest.approx(0.044, abs=1e-4)
        controller._update_current_positions()
        np.testing.assert_allclose(controller.q_current_left, READY, atol=1e-3)
        assert controller.fault is None and controller.tracking_error() < 0.05
    finally:
        controller.stop()
    assert not any(bus.enabled or bus.opened for bus in buses.values())


class FailingBus(FakeCANBus):
    """FakeCANBus whose named steps raise, to rehearse a bus dying during shutdown."""

    def __init__(self, interface, fail_on=(), **kwargs):
        super().__init__(interface, **kwargs)
        self.fail_on = set(fail_on)

    def disable(self):
        if 'disable' in self.fail_on:
            raise RuntimeError(f'{self.interface}: disable failed')
        super().disable()

    def close(self):
        if 'close' in self.fail_on:
            raise RuntimeError(f'{self.interface}: close failed')
        super().close()


@pytest.mark.parametrize('step', ['disable', 'close'])
def test_stop_shuts_down_every_bus_when_the_first_one_fails(step):
    # The right bus comes first in iteration order, so its failure used to abort the loop
    # and leave the left arm enabled and open.
    buses = {'right': FailingBus('fake_right', fail_on=[step]), 'left': FakeCANBus('fake_left')}
    controller, _ = make(buses=buses)
    controller.connect()
    controller.start()
    assert buses['left'].enabled and buses['left'].opened

    with pytest.raises(RuntimeError, match=f'fake_right: {step} failed'):
        controller.stop()

    assert not buses['left'].enabled and not buses['left'].opened
    if step == 'disable':
        assert not buses['right'].opened  # closing still ran after disable raised
    assert not controller._enabled
    assert controller._thread is None or not controller._thread.is_alive()

    # Shutdown is idempotent: a second stop() does not retry disable or raise.
    buses['right'].fail_on = {'disable'}
    controller.stop()


def test_stop_closes_every_bus_after_a_failed_connect():
    # stop() also runs from the pre-flight error path, before anything is enabled.
    buses = {'right': FailingBus('fake_right', fail_on=['close']), 'left': FakeCANBus('fake_left')}
    controller, _ = make(buses=buses)
    controller.connect()
    assert buses['left'].opened

    with pytest.raises(RuntimeError, match='fake_right: close failed'):
        controller.stop()
    assert not buses['left'].opened and not buses['left'].enabled


def test_only_selected_arm_is_commanded():
    controller, buses = make(arms=('left',))
    controller.connect()
    controller.start()
    try:
        controller.set_joint_goals(dict(q_goal_right=READY, q_goal_left=READY))
        assert wait(controller.goals_reached)
        assert set(buses) == {'left'}
        controller._update_current_positions()
        # The arm without a bus is shown at its goal
        np.testing.assert_allclose(controller.q_current_right, READY)
    finally:
        controller.stop()


def test_preflight_rejects_silent_motor_error_and_bad_zero():
    controller, buses = make()
    buses['left'].silent = True
    buses['left']._last_reply -= 10
    with pytest.raises(HardwareError, match='no reply from left arm'):
        controller.connect(timeout=0.1)

    controller, buses = make()
    buses['right'].faults = ['fake_right joint2: OVERCURRENT']
    with pytest.raises(HardwareError, match='OVERCURRENT'):
        controller.connect()

    controller, buses = make()
    buses['right']._q[3] = -1.0  # J4 cannot be negative
    with pytest.raises(HardwareError, match='outside its joint limits'):
        controller.connect()
    assert not any(bus.enabled for bus in buses.values())


@pytest.mark.parametrize('failure,text,recoverable', [
    ('silent', 'no state from the right arm', False),
    ('error', 'OVERLOAD', False),
    ('stuck', 'tracking error', True)])
def test_fault_holds_at_measured_pose(monkeypatch, failure, text, recoverable):
    import openarm_teleop.control.hardware_controller as module
    monkeypatch.setattr(module, 'STATE_TIMEOUT', 0.05)
    monkeypatch.setattr(module, 'TRACKING_ERROR_DURATION', 0.05)
    controller, buses = make(max_joint_vel=1.0)
    controller.connect()
    controller.start()
    try:
        controller.set_joint_goals(dict(q_goal_right=READY, q_goal_left=READY))
        assert wait(lambda: buses['right'].read().q[3] > 0.2)
        if failure == 'silent':
            buses['right'].silent = True
        elif failure == 'error':
            buses['right'].faults = ['fake_right joint1: OVERLOAD']
        else:
            buses['right'].enabled = False  # the arm stops following its commands
        assert wait(lambda: controller.fault is not None)
        assert text in controller.fault and controller.fault_recoverable is recoverable
        held = buses['left'].read().q.copy()
        assert held[3] < READY[3] - 0.1
        controller.set_joint_goals(dict(q_goal_left=READY))  # ignored after a fault
        time.sleep(0.1)
        np.testing.assert_allclose(buses['left'].read().q, held, atol=1e-6)
        assert controller.goals_reached()
    finally:
        controller.stop()


@pytest.mark.parametrize('failure,text', [
    ('error', 'OVERCURRENT'),
    ('silent', 'no state from the right arm'),
    ('can', 'control loop stopped')])
def test_later_fault_escalates_recoverable_tracking_fault(monkeypatch, failure, text):
    # A recoverable tracking fault followed by a motor error, stale feedback or a CAN error
    # must leave the session non-recoverable, so the exit path does not attempt homing.
    import openarm_teleop.control.hardware_controller as module
    monkeypatch.setattr(module, 'STATE_TIMEOUT', 0.05)
    monkeypatch.setattr(module, 'TRACKING_ERROR_DURATION', 0.05)
    controller, buses = make(max_joint_vel=1.0)
    controller.connect()
    controller.start()
    try:
        controller.set_joint_goals(dict(q_goal_right=READY, q_goal_left=READY))
        assert wait(lambda: buses['right'].read().q[3] > 0.2)
        buses['right'].enabled = False  # the arm stops following its commands
        assert wait(lambda: controller.fault is not None)
        assert 'tracking error' in controller.fault and controller.fault_recoverable

        if failure == 'error':
            buses['right'].faults = ['fake_right joint2: OVERCURRENT']
        elif failure == 'silent':
            buses['right'].silent = True
        else:
            def read():
                raise RuntimeError('CAN bus error')
            monkeypatch.setattr(buses['right'], 'read', read)
        assert wait(lambda: not controller.fault_recoverable)
        assert 'tracking error' in controller.fault and text in controller.fault
        if failure == 'can':
            assert 'CAN bus error' in controller.fault
            assert wait(lambda: not controller._thread.is_alive())

        escalated = controller.fault
        time.sleep(0.1)
        assert controller.fault == escalated  # recorded once, not every step
        controller.set_joint_goals(dict(q_goal_left=READY))  # still ignored
        assert controller.goals_reached()
    finally:
        controller.stop()


def test_dry_run_replay_reaches_ready_tracks_and_homes(monkeypatch, tmp_path, capsys):
    from geo_kin_core.types import RetargetOutput
    import openarm_teleop.control.hardware_controller as hardware
    import openarm_teleop.session as module
    from openarm_teleop.demos.teleop import run
    calls = []
    buses = []

    class Session:
        default_q = {side: READY for side in ('right', 'left')}
        def reset(self, *args): pass
        def solve(self, frame, **kwargs):
            calls.append(kwargs)
            return RetargetOutput(q_goal_right=READY, q_goal_left=READY, right_gripper_val=0.1)

    class Bus(FakeCANBus):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            buses.append(self)

    monkeypatch.setattr(module, 'make_session', lambda **kw: Session())
    monkeypatch.setattr(hardware, 'FakeCANBus', Bus)
    log = tmp_path / 'log.npz'
    args = parse_args(['--hardware', '--dry-run', '--headless', '--arms', 'both', '--steps', '20', '--max-joint-vel', '6',
                       '--log', str(log)])
    assert args.rate is None and args.loop is False
    assert run(args) == 20
    out = capsys.readouterr().out
    assert 'Tracking.' in out and 'At home.' in out and 'Tracking summary' in out
    # Tracking started from the ready pose, with the measured pose fed back to the solver
    np.testing.assert_allclose(calls[0]['q_current_right'], READY, atol=0.05)  # one command step behind
    assert len(buses) == 2 and not any(bus.enabled or bus.opened for bus in buses)
    for bus in buses:
        np.testing.assert_allclose(bus.read().q[3], np.deg2rad(5.0), atol=1e-3)  # home, inside J4's stop
    with np.load(log) as saved:
        assert saved['right_cmd'].shape[1] == 7 and len(saved['left_t']) > 0


def test_hardware_refuses_unsafe_flags():
    from openarm_teleop.demos.teleop import run
    with pytest.raises(ValueError, match='--dynamic'):
        run(parse_args(['--hardware', '--dry-run', '--dynamic', '--headless']))
    with pytest.raises(ValueError, match='--no-safety-filter'):
        run(parse_args(['--hardware', '--no-safety-filter', '--headless']))
