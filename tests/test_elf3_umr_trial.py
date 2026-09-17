"""CPU-only unit tests; optional real-asset checks do not start a simulator GUI."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external/umr_trial_20260908"))

import numpy as np
from elf3_umr_asset import TPOSE, pose_attributes, safe_mesh_path, transform
from run_elf3_umr_trial import elf3_config, anatomical_metrics, motion_audit


class PathAndFrameTests(unittest.TestCase):
    def test_mesh_reference(self):
        self.assertEqual(safe_mesh_path("./meshes/a.STL"), "meshes/a.STL")

    def test_unsafe_mesh_reference(self):
        for value in ("/tmp/a.STL", "../meshes/a.STL", "meshes/../a.STL", "http://a.STL", "meshes/a.obj"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_mesh_path(value)

    def test_identity_origin(self):
        np.testing.assert_array_equal(transform(None), np.eye(4))
        self.assertEqual(pose_attributes(None)["quat"], "1 0 0 0")

    def test_rpy_rotation(self):
        origin = ET.fromstring('<origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>')
        t = transform(origin)
        np.testing.assert_allclose(t @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-12)

    def test_distinct_elf3_tpose(self):
        self.assertEqual(set(TPOSE), {"l_shoulder_x_joint", "r_shoulder_x_joint", "l_elbow_y_joint", "r_elbow_y_joint"})
        self.assertGreater(TPOSE["l_elbow_y_joint"], 1.5)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "UMR YAML runtime not installed")
    def test_complete_robot_config_replacement(self):
        cfg = elf3_config(Path("/tmp/elf3.xml"), 1024, 1500)
        self.assertNotIn("g1", json.dumps(cfg.robot).lower())
        self.assertEqual(cfg.robot["foot_bodies"], ["l_ankle_x_link", "r_ankle_x_link"])
        self.assertNotIn("max_velocity", cfg.retarget)
        self.assertEqual(cfg.retarget["tpose_offset"], 0.)

    def test_elf3_anatomy_metric_names(self):
        from umr_smplx_source import SEGMENTS
        seg = np.array([SEGMENTS.index("l_hand"), SEGMENTS.index("r_foot"), SEGMENTS.index("head")])
        result = anatomical_metrics(seg, np.array([0, 1, 2]), ["l_wrist_z_link", "r_ankle_x_link", "head_y_link"])
        self.assertEqual(result["overall"], 1.)

    def test_wrong_side_anatomy_fails(self):
        from umr_smplx_source import SEGMENTS
        result = anatomical_metrics(np.array([SEGMENTS.index("l_hand")]), np.array([0]), ["r_wrist_z_link"])
        self.assertEqual(result["overall"], 0.)


@unittest.skipUnless((ROOT / "local/elf3_model_20260915a/elf3.xml").exists()
                     and importlib.util.find_spec("mujoco"), "Optional local ELF3 asset not built")
class RealAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        cls.model = mujoco.MjModel.from_xml_path(str(ROOT / "local/elf3_model_20260915a/elf3.xml"))
        cls.urdf = ROOT / "local/elf3_assets_20260915a/elf3.urdf"

    def test_independent_urdf_fk(self):
        from elf3_umr_asset import validate_urdf_fk
        result = validate_urdf_fk(ET.parse(self.urdf).getroot(), self.model)
        self.assertLess(result["max_position_component_error_m"], 1e-10)

    def test_bad_trajectory_shape_rejected(self):
        with self.assertRaises(ValueError):
            motion_audit(self.model, np.zeros((2, 36)), 50., self.urdf)

    def test_quaternion_rejected(self):
        with self.assertRaises(ValueError):
            motion_audit(self.model, np.zeros((2, 38)), 50., self.urdf)

    def test_audit_detects_velocity_violation(self):
        q = np.tile(self.model.key_qpos[0], (2, 1))
        q[1, self.model.joint("head_z_joint").qposadr[0]] = 1.
        result = motion_audit(self.model, q, 50., self.urdf)
        self.assertEqual(result["joint_speed_over_urdf_limit_intervals"], 1)
        self.assertAlmostEqual(result["joint_speed_max_rad_s"], 50.)
        self.assertFalse(result["training_approved"])

    def test_audit_detects_angle_violation(self):
        q = np.tile(self.model.key_qpos[0], (2, 1))
        joint = self.model.joint("head_z_joint")
        q[:, joint.qposadr[0]] = joint.range[1] + .1
        result = motion_audit(self.model, q, 50., self.urdf)
        self.assertEqual(result["joint_limit_violation_frames"], 2)
        self.assertAlmostEqual(result["joint_limit_max_excess_rad"], .1)


if __name__ == "__main__":
    unittest.main()
