from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_elf3_arm_motion import motion_metrics
from check_elf3_visual_intersections import segment_surface_crossings


class ArmMetricTests(unittest.TestCase):
    def test_constant_angular_motion_not_euler_wrap_jump(self):
        angles = np.deg2rad([178., 179., 180., 181., 182.])
        r = Rotation.from_euler("z", angles).as_matrix()
        result = motion_metrics(np.zeros((5, 3)), r)
        self.assertAlmostEqual(result["angular_speed_rad_s"]["max"], np.deg2rad(50.))
        self.assertLess(result["angular_acceleration_rad_s2"]["max"], 1e-9)

    def test_loop_jump_separate_from_inside_clip(self):
        result = motion_metrics(np.arange(5)[:, None] * np.array([[.01, 0., 0.]]), np.tile(np.eye(3), (5, 1, 1)))
        self.assertAlmostEqual(result["speed_m_s"]["max"], .5)
        self.assertAlmostEqual(result["loop_position_jump_m"], .04)
        self.assertLess(result["acceleration_m_s2"]["max"], 1e-9)

    def test_actual_triangle_crossing_and_clear_segment(self):
        import trimesh
        mesh = trimesh.creation.box()
        crossed = segment_surface_crossings(mesh, np.array([[-1., 0., .1]]), np.array([[1., 0., .1]]))
        clear = segment_surface_crossings(mesh, np.array([[-1., 2., .1]]), np.array([[1., 2., .1]]))
        self.assertEqual(crossed["crossing_edges"], 1)
        self.assertEqual(clear["crossing_hits"], 0)


if __name__ == "__main__":
    unittest.main()
