from itertools import product
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from elf3_mesh_distance import certified_separated_distance, CheckedMeshDistance


class CertifiedDistanceTests(unittest.TestCase):
    def test_known_box_distances_and_rigid_invariance(self):
        box=np.array(list(product((-1.,1.),repeat=3)))
        rotation=Rotation.from_rotvec([.2,-.5,1.]).as_matrix()
        for shift in ([3.,0.,0.],[3.,4.,0.],[3.,4.,5.]):
            expected=np.linalg.norm(np.maximum(np.abs(shift)-2.,0.))
            for r,t in ((np.eye(3),np.zeros(3)),(rotation,np.array([4.,7.,-2.]))):
                a,b=box@r.T+t,(box+shift)@r.T+t
                distance,witness=certified_separated_distance(a,b)
                self.assertAlmostEqual(distance,expected,places=9)
                self.assertAlmostEqual(np.linalg.norm(witness[3:]-witness[:3]),expected,places=9)

    def test_overlap_and_touch_do_not_become_clearance(self):
        box=np.array(list(product((-1.,1.),repeat=3)))
        for shift in ([1.,0.,0.],[2.,0.,0.]):
            with self.assertRaises(ValueError):certified_separated_distance(box,box+shift)

    @unittest.skipUnless((ROOT/'local/elf3_trajectory_reuse_20260915a/motion.npz').exists(),'Local assets')
    def test_real_inconsistent_zero_is_recovered_by_separation_certificate(self):
        import mujoco
        from elf3_whole_body_model import FullBodyChart
        from refine_elf3_arm_spline import spline_basis
        from diagnose_elf3_arm_motion import visual_arm_core_pairs
        m=mujoco.MjModel.from_xml_path(str(ROOT/'local/elf3_model_20260915a/elf3.xml'))
        data=mujoco.MjData(m)
        with np.load(ROOT/'local/elf3_trajectory_reuse_20260915a/motion.npz') as z:q=z['qpos'][25:32].copy()
        chart=FullBodyChart(q);basis,_=spline_basis(np.arange(7)/50.,.12)
        seed=np.linalg.lstsq(basis,chart.original,rcond=None)[0]
        direction=np.random.default_rng(18).normal(size=seed.shape);direction/=np.linalg.norm(direction)
        data.qpos[:]=chart.qpos(basis@(seed+1e-6*direction))[6];mujoco.mj_forward(m,data)
        a,b=visual_arm_core_pairs(m)[57];witness=np.zeros(6)
        raw=mujoco.mj_geomDistance(m,data,a,b,.01,witness)
        checked=CheckedMeshDistance(m);distance=checked(data,a,b,.01,witness)
        self.assertGreater(distance,.0014);self.assertLess(distance,.0016)
        if raw==0.:self.assertEqual(checked.recoveries,1)
        normal=(witness[3:]-witness[:3])/distance
        gap=np.min(checked.world_vertices(data,b)@normal)-np.max(checked.world_vertices(data,a)@normal)
        self.assertLess(abs(gap-distance),1e-8)


if __name__=='__main__':unittest.main()
