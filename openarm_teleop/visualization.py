"""Shared human overlay in the solver-aligned MuJoCo world frame."""
import numpy as np
from geo_kin_core.viz import HumanCapsuleViz, capsules, draw_filtered_sew


def configure_camera(camera):
    camera.lookat[:] = [0.0, 0.0, 1.0]
    camera.distance = 2.6
    camera.azimuth = 135
    camera.elevation = -18


class ReplayOverlay:
    def __init__(self, viewer, *, show_human=True, show_safety=False):
        self.viewer = viewer
        self.show_human = show_human
        self.show_safety = show_safety
        self.human = HumanCapsuleViz(viewer, color=(0.15, 0.55, 1.0, 0.55), size=0.035)

    def draw(self, frame, session, to_world=None):
        capsules.clear(self.viewer)
        rotation = getattr(session, 'R_mocap_world', None)
        position = getattr(session, 'p_mocap_world', None)
        if rotation is not None and position is not None:
            self.human.set_base_offset(position, np.asarray(rotation).T)
        count = 0
        if self.show_safety and frame is not None:
            # Read the Rust session's actual XPBD capsules, including torso
            # and shoulder. No application-side proxy geometry or radii.
            count += draw_filtered_sew(self.viewer, session, to_world=to_world)
        if self.show_human:
            count += self.human.draw(frame)
        return count
