"""Smoke-check an installed release, outside the source checkout."""
from importlib.metadata import version
assert version('eurola-openarm-teleop') == "0.1.2"
import mujoco
from openarm_teleop import scene_path, SAMPLE_MOTIONS
from geo_kin_core.frames import load_frames
model = mujoco.MjModel.from_xml_path(str(scene_path()))
assert model.nq > 0
for path in SAMPLE_MOTIONS.values():
    assert path.is_file()
    load_frames(path)
from openarm_teleop.demos.replay_offline import parse_args
assert parse_args(['--headless']).device == 'replay'
from importlib.util import find_spec
from importlib.metadata import requires
from openarm_teleop.demos.teleop import parser
args = parser().parse_args([])
assert args.device == 'replay' and args.hardware is False
assert find_spec('openarm_can') is None, 'Base wheel must not install the CAN driver'
can_requirements = [r for r in requires('eurola-openarm-teleop') if r.startswith('openarm-can')]
assert len(can_requirements) == 1 and 'hardware' in can_requirements[0] and 'extra' in can_requirements[0]
print("Release smoke check passed")
