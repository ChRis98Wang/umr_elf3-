"""Signed-distance recovery acts on displacement; no dependence on IK dt units."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external/umr_trial_20260908"))
import numpy as np
import mujoco
import mink
from replay_elf3_umr_stage2 import recovering_contact_class


class ContactRecoveryTests(unittest.TestCase):
    def setup_pair(self, x):
        model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
        <body name="a"><freejoint/><geom name="a_geom" type="sphere" size="0.1"/></body>
        <body name="b" pos="{x} 0 0"><freejoint/><geom name="b_geom" type="sphere" size="0.1"/></body>
        </worldbody></mujoco>''')
        config = mink.Configuration(model)
        limit = recovering_contact_class(mink.CollisionAvoidanceLimit)(
            model, [(["a_geom"], ["b_geom"])], minimum_distance_from_collisions=.02,
            collision_detection_distance=.1)
        return model, config, limit

    def test_positive_clearance_bound(self):
        _, config, limit = self.setup_pair(.23)
        constraint = limit.compute_qp_inequalities(config, .02)
        np.testing.assert_allclose(constraint.h, [.0085], atol=1e-12)

    def test_inside_margin_requires_escape(self):
        _, config, limit = self.setup_pair(.21)
        constraint = limit.compute_qp_inequalities(config, .02)
        self.assertLess(constraint.h[0], 0.)

    def test_dt_independent_delta_q_constraint(self):
        _, config, limit = self.setup_pair(.23)
        a = limit.compute_qp_inequalities(config, .02)
        b = limit.compute_qp_inequalities(config, .005)
        np.testing.assert_array_equal(a.G, b.G)
        np.testing.assert_array_equal(a.h, b.h)

    def test_penetration_recovery_direction(self):
        model, config, limit = self.setup_pair(.19)
        before = mujoco.mj_geomDistance(model, config.data, 0, 1, 1., None)
        velocity = mink.solve_ik(config, [], .02, "clarabel", damping=1., limits=[limit])
        config.integrate_inplace(velocity, .02)
        after = mujoco.mj_geomDistance(model, config.data, 0, 1, 1., None)
        self.assertGreater(after, before + .02)

    def test_far_pair_inactive(self):
        _, config, limit = self.setup_pair(.5)
        self.assertTrue(limit.compute_qp_inequalities(config, .02).inactive)


if __name__ == "__main__":
    unittest.main()
