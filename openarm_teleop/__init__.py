"""OpenArm teleoperation assets and application interfaces (lazy imports)."""
from pathlib import Path

def model_path() -> Path:
    return Path(__file__).parent / "assets/v1/openarm_bimanual.xml"

def spec_path() -> Path:
    return Path(__file__).parent / "assets/specs/openarm_v1.npz"

SAMPLE_MOTIONS = {name: Path(__file__).parent / "assets/sample_motion" / f"{name}.npz"
                  for name in ("ipman_roll", "picking_up_mustard")}
SAMPLE_MOTION = SAMPLE_MOTIONS["ipman_roll"]


def scene_path() -> Path:
    return Path(__file__).parent / "assets/v1/scene.xml"
