from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from refine_elf3_arm_spline import spline_basis, arm_joint_prior, ArmObjective
from refine_elf3_umr_trajectory import BoundSurface


class ArmSplineTests(unittest.TestCase):
    def test_basis_preserves_clock_and_partition_of_unity(self):
        times = np.arange(330)/50.
        basis, knots = spline_basis(times, .12)
        self.assertEqual(basis.shape[0], len(times))
        np.testing.assert_allclose(basis.sum(axis=1), 1.)
        self.assertGreaterEqual(basis.min(), 0.)
        self.assertEqual(knots[-1], times[-1])

    def test_original_angle_bounds_follow_from_convex_basis(self):
        basis, _ = spline_basis(np.arange(101)/50., .12)
        coefficients = np.random.default_rng(4).uniform(-.3, 2.1, size=(basis.shape[1], 14))
        angles = basis @ coefficients
        self.assertTrue(np.all(angles >= -.3))
        self.assertTrue(np.all(angles <= 2.1))

    def test_full_trajectory_prior_penalizes_original_jitter(self):
        theta = np.zeros((8, 31))
        theta[::2, 20] = .2
        value, _ = arm_joint_prior(theta, theta, np.array([20]))
        self.assertGreater(value, 0.)

    def test_prior_and_basis_chain_gradient(self):
        rng = np.random.default_rng(3)
        basis, _ = spline_basis(np.arange(51)/50., .12)
        coefficients = rng.normal(size=(basis.shape[1], 2))
        direction = rng.normal(size=coefficients.shape)
        base = np.zeros((51, 31))
        arms = np.array([17, 24])

        def cost(c):
            theta = base.copy()
            theta[:, arms] = basis @ c
            value, grad = arm_joint_prior(theta, base, arms)
            return value, basis.T @ grad[:, arms]

        _, grad = cost(coefficients)
        eps = 1e-6
        numerical = (cost(coefficients + eps*direction)[0] - cost(coefficients-eps*direction)[0])/(2*eps)
        self.assertAlmostEqual(numerical, float(np.sum(grad*direction)), places=6)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "local/elf3_model_20260915a/elf3.xml"
MOTION = ROOT / "local/elf3_trajectory_reuse_20260915a/motion.npz"


@unittest.skipUnless(MODEL.exists() and MOTION.exists(), "Optional real ELF3 model")
class WristPoseGradientTests(unittest.TestCase):
    def test_full_mesh_pose_objective_and_spline_chain(self):
        import mujoco
        from run_elf3_umr_trial import UMR
        sys.path.insert(0, str(UMR))
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        with np.load(MOTION, allow_pickle=False) as z:
            qpos = z["qpos"][25:32].copy()
        surface = BoundSurface(model, [model.body("l_wrist_z_link").id], [[.02, .03, .01]])
        basis, _ = spline_basis(np.arange(len(qpos)) / 50., .12)
        objective = ArmObjective(model, qpos, surface, [], basis=basis, wrist_pose_loss=True)
        seed = np.linalg.lstsq(basis, qpos[:, objective.address[objective.arms]], rcond=None)[0]
        rng = np.random.default_rng(52)
        seed += rng.normal(size=seed.shape) * .07
        direction = rng.normal(size=seed.shape)
        direction /= np.linalg.norm(direction)
        x, d = seed.ravel(), direction.ravel()
        _, gradient = objective.coefficients_objective(x)
        eps = 1e-6
        numerical = (objective.coefficients_objective(x+eps*d)[0] - objective.coefficients_objective(x-eps*d)[0])/(2*eps)
        np.testing.assert_allclose(numerical, gradient @ d, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
