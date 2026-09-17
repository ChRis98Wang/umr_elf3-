from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from elf3_morphology_targets import MorphologyTargets


class MorphologyTargetTests(unittest.TestCase):
    def setUp(self):
        self.human = np.array([[1., 2., 3.], [4., 5., 6.]])
        self.robot = self.human + [[.1, 0., 0.], [0., .2, 0.]]
        self.rotation = np.tile(np.eye(3), (2, 1, 1))

    def test_canonical_points_match_robot(self):
        model = MorphologyTargets(self.human, self.robot, self.rotation)
        np.testing.assert_allclose(model.positions(self.human, self.rotation), self.robot)

    def test_zero_offset_keeps_deforming_surface_exactly(self):
        model = MorphologyTargets(self.human, self.human, self.rotation)
        moving = self.human + [[.7, .4, -.2], [-.5, .2, .1]]
        np.testing.assert_array_equal(model.positions(moving, self.rotation), moving)

    def test_offset_rotates_with_bone(self):
        model = MorphologyTargets(self.human, self.robot, self.rotation)
        rot = np.tile([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], (2, 1, 1))
        np.testing.assert_allclose(model.positions(self.human, rot), self.human + [[0., .1, 0.], [-.2, 0., 0.]])

    def test_nonidentity_canonical_rotation_and_world_equivariance(self):
        rot = np.tile([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], (2, 1, 1))
        hp, rp = self.human @ rot[0].T + 4., self.robot @ rot[0].T + 4.
        model = MorphologyTargets(hp, rp, rot)
        np.testing.assert_allclose(model.positions(hp, rot), rp)

    def test_skin_deformation_is_not_discarded(self):
        model = MorphologyTargets(self.human, self.robot, self.rotation)
        deformation = np.array([[.013, .02, -.004], [-.02, -.01, .01]])
        np.testing.assert_allclose(model.positions(self.human + deformation, self.rotation), self.robot + deformation)

    def test_inputs_not_mutated(self):
        original = self.human.copy()
        model = MorphologyTargets(self.human, self.robot, self.rotation)
        model.positions(self.human, self.rotation)
        np.testing.assert_array_equal(self.human, original)

    def test_invalid_geometry_rejected(self):
        for rotation in (self.rotation * 2., np.zeros((3, 3)), self.rotation * np.nan):
            with self.subTest(rotation=rotation.shape), self.assertRaises(ValueError):
                MorphologyTargets(self.human, self.robot, rotation)

    def test_moving_shape_mismatch_rejected(self):
        model = MorphologyTargets(self.human, self.robot, self.rotation)
        with self.assertRaises(ValueError):
            model.positions(self.human[:1], self.rotation)


if __name__ == "__main__":
    unittest.main()
