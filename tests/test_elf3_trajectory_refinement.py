"""Analytic derivatives checked against the original ELF3 mesh model."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external/umr_trial_20260908"))
from refine_elf3_umr_trajectory import BoundSurface, TrajectoryObjective, correction_prior


class CorrectionPriorTests(unittest.TestCase):
    def test_zero_correction_not_original_motion_smoothing(self):
        value, grad = correction_prior(np.zeros((9, 31)))
        self.assertEqual(value, 0.)
        np.testing.assert_array_equal(grad, 0.)

    def test_temporal_gradient(self):
        rng = np.random.default_rng(13)
        delta, direction = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
        value, grad = correction_prior(delta)
        self.assertGreater(value, 0.)
        eps = 1e-6
        numerical = (correction_prior(delta + eps * direction)[0] - correction_prior(delta - eps * direction)[0]) / (2. * eps)
        self.assertAlmostEqual(numerical, float(np.sum(grad * direction)), places=5)


MODEL = ROOT / "local/elf3_model_20260915a/elf3.xml"
MOTION = ROOT / "local/elf3_contact_recover_20260915b/motion.npz"


@unittest.skipUnless(MODEL.exists() and MOTION.exists() and importlib.util.find_spec("mujoco"), "Optional ELF3 pilot assets")
class GeometryGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        cls.model = mujoco.MjModel.from_xml_path(str(MODEL))
        with np.load(MOTION, allow_pickle=False) as z:
            cls.qpos = z["qpos"].copy()

    def test_bound_surface_jacobian(self):
        import mujoco
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.qpos[318]
        mujoco.mj_forward(self.model, data)
        body = self.model.body("l_wrist_z_link").id
        surface = BoundSurface(self.model, [body], [[.013, -.02, .017]])
        _, jac = surface.evaluate(data)
        direction = np.random.default_rng(3).normal(size=self.model.nv)
        positions = []
        eps = 1e-6
        for sign in (1., -1.):
            data.qpos[:] = self.qpos[318]
            mujoco.mj_integratePos(self.model, data.qpos, direction, sign * eps)
            mujoco.mj_forward(self.model, data)
            positions.append(surface.evaluate(data, False)[0])
        np.testing.assert_allclose((positions[0] - positions[1]) / (2 * eps), jac @ direction, atol=1e-8)

    def test_mesh_collision_objective_directional_derivative(self):
        import mujoco
        import mink
        body = self.model.body("l_wrist_z_link").id
        surface = BoundSurface(self.model, [body], [[.013, -.02, .017]])
        geoms = mink.get_subtree_geom_ids(self.model, 1)
        pairs = mink.CollisionAvoidanceLimit(self.model, [(geoms, geoms)]).geom_id_pairs
        # Locate actual penetration from the archived trajectory, not an assumed frame.
        data = mujoco.MjData(self.model)
        best, index = 0., 1
        for i in range(1, len(self.qpos) - 1):
            data.qpos[:] = self.qpos[i]
            mujoco.mj_forward(self.model, data)
            depth = max((-c.dist for c in data.contact if self.model.geom_bodyid[c.geom1]
                         and self.model.geom_bodyid[c.geom2]), default=0.)
            if depth > best:
                best, index = depth, i
        self.assertGreater(best, .005)
        objective = TrajectoryObjective(self.model, self.qpos[index-1:index+2], surface, pairs)
        x = objective.base[:, objective.address].ravel()
        _, grad = objective(x)
        direction = np.random.default_rng(8).normal(size=x.shape)
        direction /= np.linalg.norm(direction)
        eps = 1e-6
        numerical = (objective(x + eps * direction)[0] - objective(x - eps * direction)[0]) / (2 * eps)
        np.testing.assert_allclose(numerical, grad @ direction, rtol=3e-3, atol=5e-4)


if __name__ == "__main__":
    unittest.main()
