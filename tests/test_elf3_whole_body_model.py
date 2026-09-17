from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'external/umr_trial_20260908')]
from elf3_whole_body_model import FullBodyChart, right_jacobian, temporal_prior, WholeBodyObjective
from refine_elf3_arm_spline import spline_basis
from refine_elf3_umr_trajectory import BoundSurface
from diagnose_elf3_arm_motion import visual_arm_core_pairs


class RotationChartTests(unittest.TestCase):
    def test_pi_crossing_and_quaternion_sign_do_not_jump(self):
        q = np.zeros((101,38))
        q[:,3:7] = Rotation.from_euler('z', np.linspace(0,3.4,101)).as_quat(scalar_first=True)
        q[::2,3:7] *= -1
        chart=FullBodyChart(q)
        self.assertLess(np.abs(np.diff(chart.original[:,3:6],axis=0)).max(),.04)
        np.testing.assert_allclose(Rotation.from_quat(chart.qpos(chart.original)[:,3:7],scalar_first=True).as_matrix(),
                                   Rotation.from_quat(q[:,3:7],scalar_first=True).as_matrix(),atol=1e-12)

    def test_multiple_turns_rejected_not_cropped(self):
        q=np.zeros((101,38));q[:,3:7]=Rotation.from_euler('z',np.linspace(0,6.1,101)).as_quat(scalar_first=True)
        with self.assertRaises(ValueError):FullBodyChart(q)

    def test_right_jacobian_finite_difference(self):
        for v in (np.zeros(3),np.array([.2,-.3,3.2])):
            direction=np.array([.1,-.2,.3]);eps=1e-7
            numerical=(Rotation.from_rotvec(v).inv()*Rotation.from_rotvec(v+eps*direction)).as_rotvec()/eps
            np.testing.assert_allclose(numerical,right_jacobian(v)@direction,atol=2e-8)

    def test_temporal_gradient(self):
        rng=np.random.default_rng(6);x=rng.normal(size=(8,37));v=rng.normal(size=x.shape);base=x*.9;eps=1e-6
        _,g=temporal_prior(x,base)
        numerical=(temporal_prior(x+eps*v,base)[0]-temporal_prior(x-eps*v,base)[0])/(2*eps)
        np.testing.assert_allclose(numerical,np.sum(g*v),rtol=1e-7)


MODEL=ROOT/'local/elf3_model_20260915a/elf3.xml'
MOTION=ROOT/'local/elf3_trajectory_reuse_20260915a/motion.npz'


@unittest.skipUnless(MODEL.exists() and MOTION.exists(),'Optional real ELF3 assets')
class FullFKGradientTests(unittest.TestCase):
    def test_root_chart_plus_all_joints_full_objective_gradient(self):
        import mujoco
        m=mujoco.MjModel.from_xml_path(str(MODEL))
        with np.load(MOTION) as z:q=z['qpos'][25:32].copy()
        s=BoundSurface(m,[m.body('l_wrist_z_link').id,m.body('r_ankle_x_link').id],[[.02,.03,.01],[.01,.01,0.]])
        b,_=spline_basis(np.arange(len(q))/50.,.12)
        objective=WholeBodyObjective(m,q,s,[],[],b)
        seed=np.linalg.lstsq(b,objective.chart.original,rcond=None)[0]
        rng=np.random.default_rng(81);seed+=rng.normal(size=seed.shape)*.005
        direction=rng.normal(size=seed.shape);direction/=np.linalg.norm(direction)
        x,d=seed.ravel(),direction.ravel();eps=1e-6
        _,gradient=objective(x)
        numerical=(objective(x+eps*d)[0]-objective(x-eps*d)[0])/(2*eps)
        np.testing.assert_allclose(numerical,gradient@d,rtol=2e-5,atol=2e-6)

    def test_signed_mesh_clearance_gradient(self):
        import mujoco
        m=mujoco.MjModel.from_xml_path(str(MODEL))
        with np.load(MOTION) as z:q=z['qpos'][25:32].copy()
        s=BoundSurface(m,[m.body('r_wrist_z_link').id],[[.02,.03,.01]])
        b,_=spline_basis(np.arange(len(q))/50.,.12)
        objective=WholeBodyObjective(m,q,s,[],visual_arm_core_pairs(m),b,clearance=.001)
        seed=np.linalg.lstsq(b,objective.chart.original,rcond=None)[0]
        d=np.random.default_rng(18).normal(size=seed.shape);d/=np.linalg.norm(d)
        x,d=seed.ravel(),d.ravel()
        _,gradient=objective(x)
        self.assertGreater(objective.latest['collision'],0.)
        # Contact distances have numerical noise at ~1e-6 coefficient steps
        # (unlike the smooth FK test above). Check convergence at three resolved
        # scales, with a STRICTER relative tolerance, not a widened tolerance at
        # a sub-resolution step. This does not claim an exact contact derivative.
        for eps in (.0005,.001,.002):
            numerical=(objective(x+eps*d)[0]-objective(x-eps*d)[0])/(2*eps)
            np.testing.assert_allclose(numerical,gradient@d,rtol=2e-5,atol=2e-5)


if __name__=='__main__':unittest.main()
