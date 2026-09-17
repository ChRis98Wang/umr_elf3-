from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'external/umr_trial_20260908')]
from replay_elf3_umr_stage2 import foot_feasible_initial_pose


@unittest.skipUnless((ROOT/'local/elf3_model_20260915a/elf3.xml').exists(),'Local ELF3 assets')
class InitialPoseTests(unittest.TestCase):
    def test_actual_feet_determine_initial_height_without_changing_joints(self):
        import mujoco
        from refine_elf3_umr_trajectory import BoundSurface
        from run_elf3_umr_trial import elf3_mesh_sole_points
        m=mujoco.MjModel.from_xml_path(str(ROOT/'local/elf3_model_20260915a/elf3.xml'))
        q=m.key_qpos[0].copy();q[2]=.98
        original=q.copy();fixed,report=foot_feasible_initial_pose(m,q,0.,.002)
        np.testing.assert_array_equal(q,original)
        np.testing.assert_array_equal(fixed[np.arange(38)!=2],q[np.arange(38)!=2])
        self.assertGreater(report['initial_root_z_correction_m'],0.)
        d=mujoco.MjData(m);d.qpos[:]=fixed;mujoco.mj_forward(m,d)
        bodies,points,_=elf3_mesh_sole_points(m)
        self.assertAlmostEqual(BoundSurface(m,bodies,points).evaluate(d,False)[0][:,2].min(),.002,places=10)
        fixed[2]+=.1
        unchanged,report=foot_feasible_initial_pose(m,fixed,0.,.002)
        np.testing.assert_array_equal(unchanged,fixed)
        self.assertEqual(report['initial_root_z_correction_m'],0.)


if __name__=='__main__':unittest.main()
