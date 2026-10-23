"""Smoke-check an installed release, outside the source checkout."""
from importlib.metadata import version
assert version('openarm-teleop') == "0.1.1"
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
print("Release smoke check passed")
