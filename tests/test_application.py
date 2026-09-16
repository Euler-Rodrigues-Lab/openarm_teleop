import hashlib
import subprocess
import sys
import numpy as np
import mujoco
import pytest
from geo_kin_core.frames import load_frames
from openarm_teleop import model_path, spec_path, SAMPLE_MOTIONS
from openarm_teleop.control.mujoco_controller import OpenArmMuJoCoController
from openarm_teleop.demos.replay_offline import parse_args


def test_exact_model_and_controller():
    assert hashlib.sha256(model_path().read_bytes()).hexdigest() == 'e59c0daa8fd07714c2a2a5ec4ccd0c68f3e1c233da93851f617cbcc044a04f8f'
    with np.load(spec_path(), allow_pickle=False) as spec:
        assert str(spec['meta/source_sha256']) == hashlib.sha256(model_path().read_bytes()).hexdigest()
    model=mujoco.MjModel.from_xml_path(str(model_path()))
    data=mujoco.MjData(model)
    controller=OpenArmMuJoCoController(model,data)
    assert len(controller.right_arm_actuator_ids) == len(controller.left_arm_actuator_ids) == 7
    goal=np.array([0.,0.,0.,np.pi/2,0.,0.,0.])
    for side in ('right','left'):
        data.qpos[getattr(controller,f'{side}_arm_qpos_addrs')] = goal
    mujoco.mj_forward(model,data)
    controller.set_joint_goals(dict(q_goal_right=goal,q_goal_left=goal,left_gripper_val=.035,right_gripper_val=.1))
    controller.update_position_control()
    np.testing.assert_allclose(data.ctrl[controller.right_arm_actuator_ids],goal)
    np.testing.assert_allclose(data.ctrl[controller.left_hand_actuator_ids],[.025])
    np.testing.assert_allclose(data.ctrl[controller.right_hand_actuator_ids],[.044])
    for _ in range(10):
        mujoco.mj_step(model,data)
    assert np.isfinite(data.qpos).all()


@pytest.mark.parametrize('sample,n,sha',[
 ('ipman_roll',600,'b18ad2ed75428304249626f0451eaba9865bab964289845d837969c05c994698'),
 ('picking_up_mustard',843,'38c32eb839d89f4b203cf6ee8923028b595c2ff76a625321f934f2033f03a04f')])
def test_sample_and_default_selection(sample,n,sha):
    path=SAMPLE_MOTIONS[sample]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    stream=load_frames(path)
    assert len(stream)==n
    assert stream.fps==60
    assert stream.frame_at_time(0).right_sew is not None
    args=parse_args(['--sample',sample,'--headless','--max_frames','3'])
    assert args.frames==str(path) and args.device=='replay' and args.steps==3
    assert parse_args(['--frames','custom.npz']).frames=='custom.npz'


def test_base_import_is_lightweight():
    subprocess.run([sys.executable,'-c',
        "import sys,openarm_teleop; assert not {'mujoco','torch','geo_kin','cv2'} & sys.modules.keys()"],check=True)


def test_replay_loop_uses_shared_source(monkeypatch):
    from geo_kin_core.types import RetargetOutput
    import openarm_teleop.session as module
    from openarm_teleop.demos.teleop import run
    calls=[]
    class Session:
        default_q={side:np.array([0.,0.,0.,np.pi/2,0.,0.,0.]) for side in ('right','left')}
        def reset(self,*args): pass
        def solve(self,frame,**kwargs):
            calls.append((frame,kwargs))
            return RetargetOutput(q_goal_right=self.default_q['right'],q_goal_left=self.default_q['left'])
    monkeypatch.setattr(module,'make_session',lambda **kw:Session())
    assert run(parse_args(['--headless','--steps','3']))==3
    assert len(calls)==3 and all(kw['engaged'] for _,kw in calls)
    assert all(kw['q_current_right'].shape==(7,) for _,kw in calls)


def test_device_cleanup_on_solve_failure(monkeypatch):
    import openarm_teleop.session as module
    import xrt_devices.integrations.geo_kin as inputs
    from openarm_teleop.demos.teleop import run,parser
    closed=[]
    class Source:
        def __init__(self,**kw): pass
        def get_frame(self): return load_frames(SAMPLE_MOTIONS['ipman_roll']).frame_at_time(0)
        def cleanup(self): closed.append(True)
    class Session:
        default_q={side:np.array([0.,0.,0.,np.pi/2,0.,0.,0.]) for side in ('right','left')}
        def reset(self,*args): pass
        def solve(self,*args,**kwargs): raise RuntimeError('test failure')
    monkeypatch.setattr(module,'make_session',lambda **kw:Session())
    monkeypatch.setattr(inputs,'XRDeviceAdapter',Source)
    with pytest.raises(RuntimeError,match='test failure'):
        run(parser().parse_args(['--headless','--steps','1']))
    assert closed==[True]
