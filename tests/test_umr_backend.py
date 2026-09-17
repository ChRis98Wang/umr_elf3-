from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import umr_backend as backend


def payload(frames=3):
    qpos = np.zeros((frames, 36), dtype=np.float64)
    qpos[:, 3] = 1
    qpos[:, 7:] = np.arange(29)
    return {"qpos": qpos, "fps": 50.0, "dof_names": list(backend.G1_JOINT_NAMES)}


class UmrContractTests(unittest.TestCase):
    def test_reorders_by_names_without_mutating_input(self):
        source = payload()
        source["dof_names"] = source["dof_names"][::-1]
        before = source["qpos"].copy()
        out = backend.convert_umr_payload(source)
        np.testing.assert_array_equal(out["dof_pos"][0], np.arange(29)[::-1])
        np.testing.assert_array_equal(out["root_rot"], np.tile([0, 0, 0, 1], (3, 1)))
        np.testing.assert_array_equal(source["qpos"], before)
        out["root_pos"][0] = 50
        np.testing.assert_array_equal(source["qpos"], before)
        self.assertEqual(set(out), {"fps", "root_pos", "root_rot", "dof_pos"})

    def test_converts_nontrivial_wxyz_and_preserves_world_height(self):
        source = payload()
        source["qpos"][:, 3:7] = [0.5, 0.5, -0.5, -0.5]
        source["qpos"][:, :3] = [3, 5, -0.025]
        out = backend.convert_umr_payload(source)
        np.testing.assert_array_equal(out["root_rot"][0], [0.5, -0.5, -0.5, 0.5])
        np.testing.assert_array_equal(out["root_pos"], source["qpos"][:, :3])

    def test_native_umr_fields_without_qpos(self):
        source = payload()
        qpos = source.pop("qpos")
        source.update(root_trans=qpos[:, :3], root_rot=qpos[:, [4, 5, 6, 3]],
                      dof=qpos[:, 7:], dof_full=qpos[:, 7:])
        out = backend.convert_umr_payload(source)
        np.testing.assert_array_equal(out["dof_pos"], qpos[:, 7:])

    def test_tiny_norm_drift_is_normalized(self):
        source = payload()
        source["qpos"][:, 3] += 1e-6
        out = backend.convert_umr_payload(source)
        np.testing.assert_array_equal(out["root_rot"][:, 3], np.ones(3))

    def test_rejects_invalid_quaternions(self):
        for quaternion in ([0, 0, 0, 0], [2, 0, 0, 0], [0.99, 0, 0, 0]):
            with self.subTest(quaternion=quaternion):
                source = payload()
                source["qpos"][:, 3:7] = quaternion
                with self.assertRaisesRegex(ValueError, "normalized"):
                    backend.convert_umr_payload(source)

    def test_rejects_invalid_fps(self):
        for fps in (0, -1, np.nan, np.inf, 1001, [50], "50", True):
            with self.subTest(fps=fps):
                source = payload()
                source["fps"] = fps
                with self.assertRaises(ValueError):
                    backend.convert_umr_payload(source)

    def test_rejects_missing_duplicate_wrong_or_mismatched_names(self):
        source = payload()
        for names in (None, list(backend.G1_JOINT_NAMES[:-1]), ["x"] * 29,
                      list(backend.G1_JOINT_NAMES[:-1]) + ["unknown"]):
            with self.subTest(names=names):
                source = payload()
                if names is None:
                    source.pop("dof_names")
                else:
                    source["dof_names"] = names
                with self.assertRaises(ValueError):
                    backend.convert_umr_payload(source)
        with self.assertRaisesRegex(ValueError, "disagree"):
            backend.convert_umr_payload(payload(), source_joint_names=backend.G1_JOINT_NAMES[::-1])

    def test_explicit_names_for_npz_qpos(self):
        source = payload()
        source.pop("dof_names")
        out = backend.convert_umr_payload(source, source_joint_names=backend.G1_JOINT_NAMES)
        self.assertEqual(out["dof_pos"].shape, (3, 29))

    def test_rejects_shapes_nonfinite_and_non_numeric(self):
        for qpos in (np.zeros((0, 36)), np.zeros((1, 36)), np.zeros((3, 35)),
                     np.zeros(36), np.full((3, 36), np.nan), np.full((3, 36), "0"),
                     np.zeros((3, 36), dtype=complex)):
            with self.subTest(shape=qpos.shape):
                source = payload()
                source["qpos"] = qpos
                with self.assertRaises(ValueError):
                    backend.convert_umr_payload(source)

    def test_redundant_fields_cannot_disagree(self):
        for key, value in (("root_trans", np.ones((3, 3))),
                           ("root_rot", np.ones((3, 4))),
                           ("dof", np.ones((3, 29))),
                           ("dof_full", np.ones((3, 29)))):
            with self.subTest(key=key):
                source = payload()
                source[key] = value
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    backend.convert_umr_payload(source)

    def test_xml_names_equal_actual_native_robot_order(self):
        robot = Path(__file__).resolve().parents[1] / "external/umr_trial_20260908" / backend.UMR_ROBOT_XML
        self.assertEqual(backend.xml_joint_names(robot), list(backend.G1_JOINT_NAMES))


class UmrArtifactTests(unittest.TestCase):
    def test_npz_ignores_unneeded_pickled_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion.npz"
            source = payload()
            source["timings"] = np.array([{"ignored": 1}], dtype=object)
            np.savez(path, **source)
            out = backend.load_artifact(path, Path(temporary))
            self.assertNotIn("timings", out)
            backend.convert_umr_payload(out)

    def test_npz_rejects_unnamed_wrong_robot(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion.npz"
            source = payload()
            source.pop("dof_names")
            source["robot_xml"] = "/not/our/robot.xml"
            expected = Path(temporary) / backend.UMR_ROBOT_XML
            expected.parent.mkdir(parents=True)
            expected.touch()
            np.savez(path, **source)
            with self.assertRaisesRegex(ValueError, "exact pinned robot"):
                backend.load_artifact(path, Path(temporary))

    def test_pickle_requires_explicit_trust(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion.pkl"
            with path.open("wb") as stream:
                pickle.dump(payload(), stream)
            with mock.patch.object(backend.pickle, "load") as load:
                with self.assertRaisesRegex(ValueError, "execute code"):
                    backend.load_artifact(path, Path(temporary))
                load.assert_not_called()
            self.assertEqual(backend.load_artifact(path, Path(temporary), trusted_pickle=True)["fps"], 50)

    def test_pin_and_clean_checkout_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(backend.subprocess, "check_output", side_effect=[str(root), "wrong"]):
                with self.assertRaisesRegex(ValueError, "pin mismatch"):
                    backend.verify_repository(root)
            with mock.patch.object(backend.subprocess, "check_output", side_effect=[str(root), backend.UMR_COMMIT, " M file"]):
                with self.assertRaisesRegex(ValueError, "dirty"):
                    backend.verify_repository(root)

    def _args(self, root):
        repo = root / "repo"
        for relative in (backend.UMR_ROBOT_XML, backend.UMR_CONFIG):
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("input")
        source = root / "walk.bvh"
        source.write_text("trusted local BVH")
        artifact = root / "motion.npz"
        np.savez(artifact, **payload(), bvh_path=str(source))
        return argparse.Namespace(command="convert", umr_repo=repo, source=source,
                                  output_dir=root / "new", origin_id="xsens/actor/walk",
                                  split="validation", input=artifact, trusted_pickle=False)

    def test_writes_native_contract_and_traceable_nonaccepted_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(Path(temporary))
            with mock.patch.object(backend, "verify_repository", return_value={"commit": backend.UMR_COMMIT}):
                destination = backend.execute(args)
            with destination.open("rb") as stream:
                result = pickle.load(stream)
            self.assertEqual(result["dof_pos"].shape, (3, 29))
            receipt = json.loads((args.output_dir / "provenance.json").read_text())
            self.assertEqual(receipt["status"], "CONVERTED_NOT_QUALITY_ACCEPTED")
            self.assertEqual(receipt["origin_id"], args.origin_id)
            self.assertEqual(receipt["split"], "validation")
            self.assertEqual(receipt["output_sha256"], backend.sha256(destination))
            self.assertFalse(receipt["quality_accepted"])
            self.assertFalse(receipt["training_started"])
            self.assertTrue(receipt["protected_inputs_unchanged"])
            self.assertIn("does not attest historical", receipt["artifact_origin_verification"])

    def test_refuses_even_empty_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(Path(temporary))
            args.output_dir.mkdir()
            with mock.patch.object(backend, "verify_repository", return_value={}):
                with self.assertRaises(FileExistsError):
                    backend.execute(args)
            self.assertEqual(list(args.output_dir.iterdir()), [])

    def test_failure_writes_receipt_without_publishing_motion(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(Path(temporary))
            with mock.patch.object(backend, "verify_repository", return_value={}), \
                    mock.patch.object(backend, "convert_umr_payload", side_effect=ValueError("bad motion")):
                with self.assertRaisesRegex(ValueError, "bad motion"):
                    backend.execute(args)
            receipt = json.loads((args.output_dir / "provenance.json").read_text())
            self.assertEqual(receipt["status"], "FAILED")
            self.assertFalse((args.output_dir / "motion.pkl").exists())

    def test_atomic_dump_failure_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion.pkl"
            def partial_failure(motion, stream, **kwargs):
                stream.write(b"partial")
                raise RuntimeError("disk full")
            with mock.patch.object(backend.pickle, "dump", side_effect=partial_failure):
                with self.assertRaisesRegex(RuntimeError, "disk full"):
                    backend._save_motion(path, {})
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.iterdir()), [])

    def test_atomic_publication_never_overwrites_existing_motion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion.pkl"
            path.write_bytes(b"previous")
            with self.assertRaises(FileExistsError):
                backend._save_motion(path, {})
            self.assertEqual(path.read_bytes(), b"previous")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_atomic_dump_preserves_preexisting_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "motion.pkl"
            previous = path.with_name(".motion.pkl.tmp")
            previous.write_bytes(b"owned by previous run")
            with self.assertRaises(FileExistsError):
                backend._save_motion(path, {})
            self.assertEqual(previous.read_bytes(), b"owned by previous run")

    def test_source_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(Path(temporary))
            np.savez(args.input, **payload(), bvh_path=str(args.source.parent / "other.bvh"))
            with mock.patch.object(backend, "verify_repository", return_value={}):
                with self.assertRaisesRegex(ValueError, "source path"):
                    backend.execute(args)

    def test_amass_not_silently_treated_as_bvh(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(Path(temporary))
            args.source = args.input
            with self.assertRaisesRegex(ValueError, "AMASS/SMPL-X"):
                backend.execute(args)
            self.assertFalse(args.output_dir.exists())

    def test_subprocess_is_bounded_and_reaped_on_timeout(self):
        process = mock.Mock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired(["python"], 1), 0, 0]
        with mock.patch.object(backend.subprocess, "Popen", return_value=process) as popen, \
                mock.patch.object(backend.os, "killpg") as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                backend.run_bounded(["python"], cwd=Path("."), timeout=1)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(popen.call_args.kwargs["env"]["PYTHONNOUSERSITE"], "1")
            self.assertNotIn("PYTHONPATH", popen.call_args.kwargs["env"])
            self.assertNotIn("PYTHONHOME", popen.call_args.kwargs["env"])
            self.assertEqual(process.wait.call_count, 3)
            self.assertEqual(kill.call_count, 2)


if __name__ == "__main__":
    unittest.main()
