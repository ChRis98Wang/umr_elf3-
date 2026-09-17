"""Synthetic tests for the official adapter; no upstream checkout/GPU/data needed."""
import json
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_official_elf3 as official
from make_official_demo_sources import author_poses, DEMO_NAMES
from check_release import DEMO_MEDIA, media_registry, verify_media


class OfficialSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="official-elf3-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.npz"
        self.fields = dict(poses=np.zeros((459, 165)), trans=np.zeros((459, 3)),
                           betas=np.arange(16, dtype=float), gender="neutral",
                           surface_model_type="smplx", mocap_frame_rate=120.)

    def save(self, **changes):
        np.savez(self.source, **{**self.fields, **changes})

    def test_complete_120hz_to_50hz_clock(self):
        self.fields["trans"][:, 0] = np.arange(459) / 120
        self.save()
        values, report = official.resample_source(self.source)
        self.assertEqual(values["poses"].shape, (191, 165))
        self.assertEqual(report["source_frames"], 459)
        self.assertTrue(report["full_duration"])
        self.assertAlmostEqual(report["duration_s"], 3.8)
        self.assertLess(report["tail_residual_s"], .02)
        np.testing.assert_allclose(values["trans"][:, 0], np.arange(191) / 50)
        np.testing.assert_array_equal(values["betas"], self.fields["betas"])

    def test_ignores_unrelated_pickled_marker_arrays(self):
        self.save(latent_labels=np.array([{"unused": 1}], dtype=object))
        values, _ = official.resample_source(self.source)
        self.assertNotIn("latent_labels", values)
        self.assertFalse(any(v.dtype.hasobject for v in values.values()))

    def test_same_clock_preserves_rotations(self):
        rng = np.random.default_rng(81)
        poses = rng.normal(size=(5, 165)) * .1
        self.save(poses=poses, trans=np.zeros((5, 3)), mocap_frame_rate=50.)
        result, _ = official.resample_source(self.source)
        np.testing.assert_allclose(result["poses"], poses, atol=1e-12)

    def test_shortest_arc_across_pi_not_rotvec_linear_interpolation(self):
        poses = np.zeros((3, 165))
        poses[:, 2] = np.deg2rad([179, -179, -177])
        self.save(poses=poses, trans=np.zeros((3, 3)), mocap_frame_rate=25.)
        result, _ = official.resample_source(self.source)
        midpoint = Rotation.from_rotvec(result["poses"][1, :3]).magnitude()
        self.assertAlmostEqual(midpoint, np.pi, places=10)

    def test_invalid_metadata_and_shapes_rejected(self):
        for changes in ({"mocap_frame_rate": 0.}, {"mocap_frame_rate": np.nan},
                        {"surface_model_type": "smplh"}, {"gender": "unknown"},
                        {"trans": np.zeros((1, 3))}, {"betas": np.zeros(2)},
                        {"poses": np.full((459, 165), np.nan)}):
            with self.subTest(changes=list(changes)):
                self.save(**changes)
                with self.assertRaises(ValueError):
                    official.resample_source(self.source)

    def test_no_assumed_source_rate(self):
        self.fields.pop("mocap_frame_rate")
        self.save()
        with self.assertRaisesRegex(ValueError, "frame rate"):
            official.resample_source(self.source)

    def test_duplicate_npz_members_rejected(self):
        self.save()
        with zipfile.ZipFile(self.source, "a") as archive:
            with self.assertWarns(UserWarning):
                archive.writestr("poses.npy", b"invalid duplicate")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            official.resample_source(self.source)

    def test_forged_or_changed_protected_inputs_rejected(self):
        protected = self.root / "protected.json"
        protected.write_text("original")
        inputs = {"schema": "umr_elf3.official_input/1", "upstream_commit": official.OFFICIAL_COMMIT,
                  "upstream": "fixture", "assets": "fixture",
                  "protected_sha256": {str(protected): official.digest(protected)}}
        (self.root / "inputs.json").write_text(json.dumps(inputs))
        protected.write_text("changed")
        with patch.object(official, "verify_official"), patch.object(official, "verify_assets"):
            with self.assertRaisesRegex(ValueError, "Protected input"):
                official.verify_inputs(self.root)
        inputs["upstream_commit"] = "wrong"
        (self.root / "inputs.json").write_text(json.dumps(inputs))
        with self.assertRaisesRegex(ValueError, "Invalid official"):
            official.verify_inputs(self.root)

    def test_prepare_never_overwrites_existing_directory(self):
        args = SimpleNamespace(upstream=self.root, assets=self.root, source=self.source,
                               body_model=self.root / "body.npz", robot_xml=self.root / "robot.xml",
                               output=self.root)
        with patch.object(official, "verify_official"), patch.object(official, "verify_assets"):
            with self.assertRaises(FileExistsError):
                official.prepare(args)

    def test_nonunit_and_nonfinite_output_rejected_before_geometry(self):
        model = SimpleNamespace()
        for qpos in (np.zeros((3, 38)), np.full((3, 38), np.nan), np.zeros((3, 36))):
            with self.assertRaisesRegex(ValueError, "Invalid finite"):
                official.basic_audit(qpos, model, self.root / "does_not_exist.urdf")

    def test_authored_examples_are_finite_closed_human_pose_sequences(self):
        for name in DEMO_NAMES:
            poses = author_poses(name)
            self.assertEqual(poses.shape, (151, 165))
            self.assertTrue(np.isfinite(poses).all())
            np.testing.assert_allclose(poses[0], poses[-1], atol=1e-12)
            self.assertGreater(np.max(np.abs(poses[75] - poses[0])), .4)

    def test_authored_source_kind_is_not_relabelled_as_amass(self):
        self.save(source_kind="authored_parametric_smplx")
        values, report = official.resample_source(self.source)
        self.assertEqual(report["source_kind"], "authored_parametric_smplx")
        self.assertEqual(values["source_kind"].item(), "authored_parametric_smplx")


class OfficialMediaReleaseTests(unittest.TestCase):
    def manifest(self):
        rows = []
        for name in DEMO_NAMES:
            rows.append({"source_kind": "authored_parametric_smplx", "amass_used": False,
                         "policy_inference": False, "training_approved": False,
                         "files": {f"official_{name}.{ext}": {"bytes": 12, "sha256": "0" * 64}
                                   for ext in ("mp4", "gif")}})
        return {"schema": "umr_elf3.authored_demo_media/1",
                "upstream_commit": official.OFFICIAL_COMMIT, "rows": rows}

    def test_only_six_declared_self_authored_media_paths_allowed(self):
        self.assertEqual(set(media_registry(json.dumps(self.manifest()))), DEMO_MEDIA)

    def test_amass_or_policy_claims_rejected(self):
        for field in ("amass_used", "policy_inference", "training_approved"):
            value = self.manifest()
            value["rows"][0][field] = True
            with self.assertRaises(ValueError):
                media_registry(json.dumps(value))

    def test_extra_or_missing_media_rejected(self):
        value = self.manifest()
        value["rows"][0]["files"]["../other.gif"] = {"bytes": 12, "sha256": "0" * 64}
        with self.assertRaises(ValueError):
            media_registry(json.dumps(value))
        value = self.manifest()
        value["rows"].pop()
        with self.assertRaises(ValueError):
            media_registry(json.dumps(value))

    def test_media_hash_and_header_checked(self):
        name = "docs/media/official_arm_raise.gif"
        raw = b"GIF89a123456"
        registry = {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
        verify_media(name, raw, registry)
        with self.assertRaises(ValueError):
            verify_media(name, raw[:-1], registry)
        bad = b"NOTGIF123456"
        registry[name]["sha256"] = hashlib.sha256(bad).hexdigest()
        with self.assertRaisesRegex(ValueError, "header"):
            verify_media(name, bad, registry)


if __name__ == "__main__":
    unittest.main()
