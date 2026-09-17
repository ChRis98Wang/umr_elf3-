"""Dependency-light source geometry/identity tests; no IsaacLab or GPU required."""
from __future__ import annotations

import importlib.util
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np

SPEC = importlib.util.spec_from_file_location("umr_smplx_source", Path(__file__).parents[1] / "scripts/umr_smplx_source.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class SmplxSourceTests(unittest.TestCase):
    def prepared(self, **overrides):
        count, points = 2, 512
        metadata = {"schema": module.SCHEMA, "frames": count, "points": points}
        out = {"metadata_json": json.dumps(metadata), "canonical_points": np.zeros((points, 3)),
               "canonical_normals": np.tile([0., 0., 1.], (points, 1)),
               "sequence_points": np.zeros((count, points, 3)),
               "sequence_normals": np.tile([0., 0., 1.], (count, points, 1)),
               "canonical_joint_positions": np.zeros((55, 3)),
               "canonical_joint_rotations": np.tile(np.eye(3), (55, 1, 1)),
               "joint_positions": np.zeros((count, 55, 3)),
               "joint_rotations": np.tile(np.eye(3), (count, 55, 1, 1)),
               "face_indices": np.zeros(points, dtype=np.int64),
               "barycentric": np.tile([.2, .3, .5], (points, 1)),
               "segment": np.zeros(points, dtype=np.int64),
               "binding_joint_ids": np.zeros(points, dtype=np.int64),
               "times": np.array([0., .02]), "fps": 50., "actor_height": 1.7,
               "segment_names": np.array(module.SEGMENTS), "foot_low": np.zeros(count),
               "sample_lower": np.array([0, 2]), "sample_upper": np.array([1, 3]),
               "sample_alpha": np.zeros(count), "parents": np.array([-1] + [0] * 54)}
        out.update(overrides)
        return out

    def read_prepared(self, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prepared.npz"
            module.save_npz_exclusive(path, **self.prepared(**overrides))
            return module.load_prepared_source(path)

    def raw(self, **overrides):
        out = dict(root_orient=np.zeros((4, 3)), pose_body=np.zeros((4, 63)),
                   trans=np.zeros((4, 3)), betas=np.zeros(16), gender="neutral",
                   surface_model_type="smplx", mocap_frame_rate=100.)
        out.update(overrides)
        return out

    def load_raw(self, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.npz"
            np.savez(path, **self.raw(**overrides))
            return module.load_amass(path)

    def test_source_numeric_fields_do_not_unpickle_unrelated_markers(self):
        raw = self.load_raw(markers_latent=np.array([object()], dtype=object))
        self.assertEqual(raw["gender"], "neutral")
        self.assertEqual(len(raw["betas"]), 16)

    def test_unknown_model_and_gender_are_not_silently_replaced(self):
        for field in ({"surface_model_type": "smplh"}, {"gender": "unknown"}):
            with self.assertRaises(ValueError):
                self.load_raw(**field)

    def test_invalid_motion_fields_rejected(self):
        for field in ({"pose_body": np.zeros((4, 60))}, {"trans": np.full((4, 3), np.nan)},
                      {"mocap_frame_rate": np.inf}, {"mocap_frame_rate": np.array([50., 100.])}):
            with self.assertRaises(ValueError):
                self.load_raw(**field)

    def test_static_framewise_shape_accepted_but_changing_shape_rejected(self):
        self.assertEqual(self.load_raw(betas=np.ones((4, 16)))["betas"].shape, (16,))
        betas = np.zeros((4, 16))
        betas[1, 0] = .1
        with self.assertRaisesRegex(ValueError, "Time-varying"):
            self.load_raw(betas=betas)

    def test_gender_model_file_is_validated_even_when_file_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            neutral = Path(directory) / "SMPLX_NEUTRAL_2020.npz"
            neutral.touch()
            self.assertEqual(module.model_file_for_gender(neutral, "neutral"), neutral)
            with self.assertRaisesRegex(ValueError, "gender female"):
                module.model_file_for_gender(neutral, "female")

    def test_sampling_grid_does_not_drop_first_amass_frame(self):
        times, lower, upper, alpha = module.sampling_grid(301, 100., 50., 0., 3.)
        self.assertEqual(len(times), 151)
        np.testing.assert_allclose(np.diff(times), .02)
        self.assertEqual(lower[0], 0)
        self.assertEqual(lower[-1], 300)
        self.assertEqual(upper[-1], 300)
        np.testing.assert_allclose(alpha, 0., atol=1e-12)

    def test_sampling_grid_never_stretches_last_partial_frame(self):
        times, _, _, _ = module.sampling_grid(298, 100., 50., .1, 3.)
        self.assertLessEqual(times[-1], 2.97)
        np.testing.assert_allclose(np.diff(times), .02)
        self.assertAlmostEqual(times[-1], 2.96)

    def test_grid_rejects_unbounded_or_empty_intervals(self):
        for values in ((3, 100, 50, 1, 3), (100, 100, 50, 0, 31),
                       (100, 100, 50, -1, 2), (100, 100, 50, 0, np.inf)):
            with self.assertRaises(ValueError):
                module.sampling_grid(*values)

    def test_rotation_interpolation_takes_short_arc(self):
        value = np.array([[0., 0., np.deg2rad(179.)], [0., 0., np.deg2rad(-179.)]])
        interpolated = module.interpolate_rotvec(value, [0], [1], np.array([.5]))
        self.assertAlmostEqual(abs(interpolated[0, 2]), np.pi)

    def test_barycentric_transport_follows_deforming_surface_not_joint(self):
        vertices = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                             [[0., 0., 0.], [2., 0., 0.], [0., 1., 1.]]])
        points, normals = module.barycentric_transport(vertices, [[0, 1, 2]], [0], [[.2, .3, .5]])
        np.testing.assert_allclose(points[:, 0], [[.3, .5, 0.], [.6, .5, .5]])
        np.testing.assert_allclose(normals[1, 0], [0., -1 / np.sqrt(2), 1 / np.sqrt(2)])

    def test_canonical_to_posed_basis_applies_exactly_once(self):
        from scipy.spatial.transform import Rotation
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        canonical = vertices @ module.CANONICAL_ROTATION.T
        points, normals = module.barycentric_transport(canonical, [[0, 1, 2]], [0], [[.2, .3, .5]])
        local_pose = Rotation.from_matrix(module.CANONICAL_ROTATION).as_rotvec().reshape(1, 3)
        posed_root_rotation = module.global_joint_rotations(local_pose, np.array([-1]))[0, 0]
        posed, posed_normals = module.barycentric_transport(vertices @ posed_root_rotation.T,
                                                           [[0, 1, 2]], [0], [[.2, .3, .5]])
        np.testing.assert_allclose(points, posed, atol=1e-12)
        np.testing.assert_allclose(normals, posed_normals, atol=1e-12)
        np.testing.assert_allclose(module.CANONICAL_ROTATION @ [0, 1, 0], [0, 0, 1])
        np.testing.assert_allclose(module.CANONICAL_ROTATION @ [0, 0, 1], [1, 0, 0])
        self.assertAlmostEqual(np.linalg.det(module.CANONICAL_ROTATION), 1.)

    def test_global_rotation_chain_not_independent_local_rotations(self):
        from scipy.spatial.transform import Rotation
        angles = np.array([[[0., 0., np.pi / 2], [np.pi / 2, 0., 0.]]])
        world = module.global_joint_rotations(angles, np.array([-1, 0]))
        expected = Rotation.from_euler("z", np.pi / 2).as_matrix() @ Rotation.from_euler("x", np.pi / 2).as_matrix()
        np.testing.assert_allclose(world[0, 1], expected, atol=1e-12)

    def test_degenerate_triangle_and_invalid_barycentric_rejected(self):
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            module.barycentric_transport(np.zeros((3, 3)), [[0, 1, 2]], [0], [[.2, .3, .5]])
        with self.assertRaisesRegex(ValueError, "sum to one"):
            module.barycentric_transport(np.eye(3), [[0, 1, 2]], [0], [[.2, .3, .6]])

    def test_area_samples_preserve_triangle_identity_and_seed(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        weights = np.zeros((3, 55))
        weights[:, 0] = 1.
        a = module.sample_canonical_surface(vertices, np.array([[0, 1, 2]]), weights, 512, seed=4)
        b = module.sample_canonical_surface(vertices, np.array([[0, 1, 2]]), weights, 512, seed=4)
        np.testing.assert_array_equal(a["barycentric"], b["barycentric"])
        np.testing.assert_array_equal(a["face_indices"], np.zeros(512))
        reproduced, _ = module.barycentric_transport(vertices, [[0, 1, 2]], a["face_indices"], a["barycentric"])
        np.testing.assert_allclose(reproduced, a["canonical_points"])

    def test_exclusive_npz_never_overwrites_or_uses_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.npz"
            module.save_npz_exclusive(path, plain=np.arange(3))
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                module.save_npz_exclusive(path, plain=np.arange(9))
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "pickle"):
                module.save_npz_exclusive(path.parent / "bad.npz", bad=np.array([{}], dtype=object))

    def test_human_protocol_uses_regressed_pelvis_and_constant_floor_offset(self):
        source = {"actor_height": 2., "foot_low": np.array([.1, .1]),
                  "canonical_joint_positions": np.zeros((55, 3)),
                  "canonical_joint_rotations": np.tile(np.eye(3), (55, 1, 1)),
                  "joint_positions": np.ones((2, 55, 3)),
                  "joint_rotations": np.tile(np.eye(3), (2, 55, 1, 1)),
                  "sequence_points": np.ones((2, 2, 3)),
                  "sequence_normals": np.tile([0., 0., 1.], (2, 2, 1))}
        original = source["sequence_points"].copy()
        human = module.SmplxSurfaceHuman(source, 1.)
        points, normals = human.targets(1)
        np.testing.assert_allclose(human.data.xpos[0], [.5, .5, .45])
        np.testing.assert_allclose(points, [[.5, .5, .45], [.5, .5, .45]])
        np.testing.assert_array_equal(source["sequence_points"], original)

    def test_mesh_retargeter_rejects_rigid_tpose_compensation(self):
        class Base:
            def __init__(self, human, **kwargs):
                self.human = human
                self.kwargs = kwargs
        target = object()
        human = type("Human", (), {"targets": lambda self, frame: target})()
        cls = module.retargeter_class(Base)
        self.assertIs(cls(human).human_targets(0), target)
        self.assertEqual(cls(human).kwargs["tpose_offset"], 0.)
        with self.assertRaisesRegex(ValueError, "Rigid T-pose"):
            cls(human, tpose_offset=1.)

    def test_root_initialization_uses_smplx_forward_and_preserves_initial_height(self):
        from scipy.spatial.transform import Rotation
        from types import SimpleNamespace

        class Base:
            def __init__(self, robot, human, **kwargs):
                self.robot, self.human = robot, human

        class Robot:
            def __init__(self, use_key):
                self.q = np.zeros(36)
                self.q[2], self.q[3] = .8, 1.
                key = self.q.copy()
                key[2] = .9
                self.model = SimpleNamespace(nkey=int(use_key), key_qpos=np.array([key]))

            def set_qpos(self, q):
                self.q = q

        class Human:
            def __init__(self, rotation):
                self.body_ids = np.array([0])
                self.data = SimpleNamespace(xpos=np.array([[1., 2., 1.1]]), xmat=rotation.reshape(1, 9))

            def set_frame(self, frame):
                pass

        for angle in (0., np.pi / 4, -np.pi / 2, np.pi):
            for use_key in (False, True):
                with self.subTest(angle=angle, use_key=use_key):
                    expected = Rotation.from_euler("z", angle).as_matrix()
                    human = Human(expected @ module.CANONICAL_ROTATION)
                    robot = Robot(use_key)
                    retargeter = module.retargeter_class(Base)(robot, human)
                    retargeter.initialize_root(0)
                    np.testing.assert_allclose(Rotation.from_quat(robot.q[3:7], scalar_first=True).as_matrix(),
                                               expected, atol=1e-12)
                    np.testing.assert_allclose(robot.q[:3], [1., 2., .9 if use_key else .8])

        # A forward axis that points straight up cannot define robot yaw.
        with self.assertRaisesRegex(ValueError, "horizontal initial heading"):
            module.retargeter_class(Base)(Robot(False), Human(np.eye(3))).initialize_root(0)

    def test_prepared_source_validates_all_rotation_and_index_contracts(self):
        self.assertEqual(self.read_prepared()["metadata"]["frames"], 2)
        for changes in (
            {"barycentric": np.tile([.2, .3, .6], (512, 1))},
            {"barycentric": np.tile([-.1, .6, .5], (512, 1))},
            {"segment": np.full(512, 21)},
            {"segment": np.full(512, .5)},
            {"binding_joint_ids": np.full(512, 55)},
            {"face_indices": np.full(512, -1)},
            {"sample_upper": np.array([2, 5])},
            {"sample_alpha": np.array([0., 1.1])},
            {"canonical_joint_rotations": np.tile(np.diag([1., 1., -1.]), (55, 1, 1))},
            {"joint_rotations": np.tile(np.diag([1., 1., 2.]), (2, 55, 1, 1))},
            {"parents": np.array([-1] + [54] * 54)},
            {"actor_height": 0.},
        ):
            with self.subTest(changes=list(changes)):
                with self.assertRaises(ValueError):
                    self.read_prepared(**changes)

    def test_loaded_geometry_fingerprint_detects_changed_mesh_without_xml(self):
        model = type("Model", (), {"mesh_vert": np.zeros((3, 3)),
                                   "mesh_face": np.array([[0, 1, 2]]),
                                   "geom_pos": np.zeros((1, 3)),
                                   "jnt_range": np.zeros((29, 2))})()
        a = module.model_geometry_fingerprint(model)
        model.mesh_vert[0, 0] = .01
        b = module.model_geometry_fingerprint(model)
        self.assertNotEqual(a["sha256"], b["sha256"])
        self.assertEqual(a["arrays"]["mesh_face"], b["arrays"]["mesh_face"])

    def test_pin_check_includes_untracked_files_before_and_after(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(module.subprocess, "check_output",
                                   side_effect=[directory, module.UMR_COMMIT, "?? unexpected.py"]) as call:
                with self.assertRaisesRegex(ValueError, "untracked"):
                    module.verify_umr_checkout(Path(directory))
                self.assertIn("--untracked-files=normal", call.call_args.args[0])

    def test_native_export_uses_common_name_quaternion_contract_and_keeps_height(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
        from umr_backend import G1_JOINT_NAMES
        qpos = np.zeros((3, 36))
        qpos[:, 2] = -.015
        qpos[:, 3:7] = [.5, -.5, .5, -.5]
        qpos[:, 7:] = np.arange(29)
        original = qpos.copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scalebfm_motion.pkl"
            report = module.export_scalebfm_motion(path, qpos, 50., G1_JOINT_NAMES[::-1])
            with path.open("rb") as stream:
                native = pickle.load(stream)
            np.testing.assert_array_equal(native["root_pos"][:, 2], np.full(3, -.015))
            np.testing.assert_array_equal(native["root_rot"][0], [-.5, .5, -.5, .5])
            np.testing.assert_array_equal(native["dof_pos"][0], np.arange(29)[::-1])
            np.testing.assert_array_equal(qpos, original)
            self.assertEqual(report["sha256"], module.sha256(path))
            self.assertFalse(report["quality_accepted"])
            with self.assertRaises(FileExistsError):
                module.export_scalebfm_motion(path, qpos, 50., G1_JOINT_NAMES)


class CanonicalSetupCacheTests(unittest.TestCase):
    def source(self):
        source = SmplxSourceTests().prepared()
        metadata = json.loads(source.pop("metadata_json"))
        metadata.update(source_gender="neutral", body_model_sha256="a" * 64,
                        shape_policy="source_betas_all_static", num_betas=16,
                        flat_hand_mean=True, expression="zero",
                        canonical_rotation=module.CANONICAL_ROTATION.tolist(), sampling_seed=0,
                        source_sha256="b" * 64, source_file="/source/actor/walk.npz",
                        start=0., requested_duration=3., posed_heading_rotation=np.eye(3).tolist())
        source["metadata"] = metadata
        source["betas"] = np.zeros(16, dtype=np.float32)
        return source

    def options(self):
        return dict(scale=.75, robot_geometry={"sha256": "c" * 64},
                    robot_xml_sha256="d" * 64, robot_config={"name": "g1", "xml": "/robot/g1.xml"},
                    sampling_config={"seed": 0, "oversample": 4.},
                    correspondence_config={"epochs": 2500, "seed": 0, "lr": .001},
                    robot_cloud={"points": np.zeros((512, 3)),
                                 "segment_names": np.array(module.SEGMENTS, dtype=object)},
                    epochs=2500, device="cuda", runtime={"numpy": "2.2.6"})

    def identity(self):
        return module.canonical_setup_identity(self.source(), **self.options())

    def build(self, identity, *, empty=False, wrong_stamp=False):
        def builder(directory, key):
            if empty:
                return
            count = identity["source_shape"]["points"]
            points = np.zeros((count, 3))
            normals = np.tile([0., 0., 1.], (count, 1))
            np.savez(directory / "bodies.npz", stamp="incorrect" if wrong_stamp else key,
                     human_points=points, human_normals=normals,
                     robot_points=points, robot_normals=normals)
            parts = ("correspondence/2", key, identity["correspondence_config"], identity["epochs"])
            stamp = hashlib.sha1(json.dumps(parts, sort_keys=True, default=str,
                                            ensure_ascii=False).encode()).hexdigest()[:16]
            np.savez(directory / "correspondence.npz", stamp=stamp,
                     bind_body_ids=np.zeros(count, dtype=np.int64), bind_local_pos=points,
                     bind_local_normal=normals, bind_snap_distance=np.zeros(count),
                     bind_world_pos=points, inherited_segment=np.zeros(count, dtype=np.int64),
                     unused_debug_object=np.array([object()], dtype=object))
        return builder

    def test_same_canonical_actor_reuses_across_different_motion_windows(self):
        source = self.source()
        first = module.canonical_setup_identity(source, **self.options())
        source["metadata"].update(source_sha256="e" * 64, source_file="/different/motion.npz",
                                  start=20., requested_duration=5., frames=51,
                                  posed_heading_rotation=np.diag([-1., -1., 1.]).tolist())
        source["sequence_points"] += 10.
        source["joint_positions"] += 5.
        source["times"] += 20.
        source["foot_low"] += .3
        second = module.canonical_setup_identity(source, **self.options())
        self.assertEqual(module.canonical_setup_key(first), module.canonical_setup_key(second))

    def test_every_canonical_geometry_and_shape_change_invalidates(self):
        original = self.source()
        base_key = module.canonical_setup_key(module.canonical_setup_identity(original, **self.options()))
        for name in ("canonical_points", "canonical_normals", "canonical_joint_positions",
                     "canonical_joint_rotations", "face_indices", "barycentric", "segment",
                     "binding_joint_ids", "parents", "betas", "actor_height"):
            with self.subTest(array=name):
                changed = copy.deepcopy(original)
                value = np.array(changed[name], copy=True)
                value.flat[0] += 1
                changed[name] = value
                key = module.canonical_setup_key(module.canonical_setup_identity(changed, **self.options()))
                self.assertNotEqual(base_key, key)
        for field, value in (("body_model_sha256", "0" * 64), ("source_gender", "female"),
                             ("sampling_seed", 1), ("flat_hand_mean", False)):
            with self.subTest(metadata=field):
                changed = copy.deepcopy(original)
                changed["metadata"][field] = value
                key = module.canonical_setup_key(module.canonical_setup_identity(changed, **self.options()))
                self.assertNotEqual(base_key, key)

    def test_config_mesh_runtime_and_algorithm_changes_invalidate(self):
        options = self.options()
        baseline = module.canonical_setup_key(module.canonical_setup_identity(self.source(), **options))
        changes = {"scale": .8, "robot_geometry": {"sha256": "f" * 64},
                   "robot_xml_sha256": "e" * 64, "robot_config": {"name": "g1", "floor": .2},
                   "sampling_config": {"seed": 1}, "correspondence_config": {"lr": .002},
                   "robot_cloud": {"points": np.ones((512, 3))}, "epochs": 5000,
                   "device": "cpu", "runtime": {"numpy": "1.26.4"},
                   "umr_commit": "9" * 40, "algorithm": "test/new_algorithm"}
        for name, value in changes.items():
            with self.subTest(option=name):
                changed = {**options, name: value}
                key = module.canonical_setup_key(module.canonical_setup_identity(self.source(), **changed))
                self.assertNotEqual(baseline, key)
        relocated = copy.deepcopy(options)
        relocated["robot_config"]["xml"] = "/another/identical_robot.xml"
        self.assertEqual(baseline, module.canonical_setup_key(
            module.canonical_setup_identity(self.source(), **relocated)))

    def test_key_rejects_missing_actual_betas_or_metadata(self):
        for key in ("betas", "metadata"):
            source = self.source()
            if key == "betas":
                source.pop("betas")
            else:
                source["metadata"].pop("body_model_sha256")
            with self.assertRaisesRegex(ValueError, "model/shape"):
                module.canonical_setup_identity(source, **self.options())
        source = self.source()
        source["betas"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "static source beta"):
            module.canonical_setup_identity(source, **self.options())
        with self.assertRaisesRegex(ValueError, "Python objects"):
            module.canonical_array_fingerprint(np.array([object()], dtype=object))

    def test_build_then_hit_preserves_bytes_and_never_calls_builder_again(self):
        identity = self.identity()
        verify = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder = mock.Mock(side_effect=self.build(identity))
            entry, first = module.obtain_cached_setup(root, identity, builder, verify,
                                                      provenance={"source": "first"})
            originals = {path.name: path.read_bytes() for path in entry.iterdir()}
            second_entry, second = module.obtain_cached_setup(root, identity, builder, verify,
                                                               provenance={"source": "second"})
            self.assertFalse(first["hit"])
            self.assertTrue(second["hit"])
            self.assertEqual(entry, second_entry)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertEqual(first["artifacts"], second["artifacts"])
            self.assertEqual(builder.call_count, 1)
            self.assertEqual(verify.call_count, 3)
            self.assertEqual(originals, {path.name: path.read_bytes() for path in entry.iterdir()})
            self.assertEqual(entry.stat().st_mode & 0o222, 0)
            self.assertTrue(all(path.stat().st_mode & 0o222 == 0 for path in entry.iterdir()))

    def test_corrupted_artifact_is_rejected_not_overwritten(self):
        identity = self.identity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry, _ = module.obtain_cached_setup(root, identity, self.build(identity), lambda: None)
            artifact = entry / "correspondence.npz"
            artifact.chmod(0o644)
            artifact.write_bytes(artifact.read_bytes() + b"changed")
            builder = mock.Mock()
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                module.obtain_cached_setup(root, identity, builder, lambda: None)
            builder.assert_not_called()
            self.assertTrue(artifact.read_bytes().endswith(b"changed"))

    def test_incomplete_empty_or_symlink_entries_are_not_hits(self):
        identity = self.identity()
        for kind in ("empty_entry", "empty_artifact", "symlink", "manifest"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                entry = root / module.canonical_setup_key(identity)
                if kind == "symlink":
                    target = root / "elsewhere"
                    target.mkdir()
                    entry.symlink_to(target, target_is_directory=True)
                else:
                    entry.mkdir()
                    if kind == "empty_artifact":
                        for name in (*module.SETUP_ARTIFACTS, "manifest.json"):
                            (entry / name).touch()
                    elif kind == "manifest":
                        (entry / "manifest.json").write_text("{}")
                builder = mock.Mock()
                with self.assertRaises(ValueError):
                    module.obtain_cached_setup(root, identity, builder, lambda: None)
                builder.assert_not_called()
                self.assertTrue(entry.exists())

    def test_failed_empty_or_wrong_stamp_build_is_never_published(self):
        identity = self.identity()
        for option in ("empty", "wrong_stamp"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(ValueError):
                    module.obtain_cached_setup(root, identity, self.build(identity, **{option: True}), lambda: None)
                self.assertFalse((root / module.canonical_setup_key(identity)).exists())
                self.assertFalse(any(".building-" in path.name for path in root.iterdir()))

    def test_changed_inputs_after_build_prevent_publication(self):
        identity = self.identity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verify = mock.Mock(side_effect=[None, ValueError("robot asset changed")])
            with self.assertRaisesRegex(ValueError, "robot asset changed"):
                module.obtain_cached_setup(root, identity, self.build(identity), verify)
            self.assertFalse((root / module.canonical_setup_key(identity)).exists())
            self.assertFalse(any(".building-" in path.name for path in root.iterdir()))

    def test_lock_contention_fails_fast_and_releases_without_residual_process(self):
        identity = self.identity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = module.canonical_setup_key(identity)
            builder = mock.Mock(side_effect=self.build(identity))
            with module.exclusive_setup_lock(root, key):
                with self.assertRaisesRegex(RuntimeError, "being built"):
                    module.obtain_cached_setup(root, identity, builder, lambda: None)
            builder.assert_not_called()
            _, report = module.obtain_cached_setup(root, identity, builder, lambda: None)
            self.assertFalse(report["hit"])

    def test_cli_forwards_optional_cache_path(self):
        with mock.patch.object(module, "retarget_source", return_value={}) as retarget:
            module.main(["retarget", "--source", "a.npz", "--umr-root", "umr", "--output", "out",
                         "--robot-xml", "g1.xml", "--setup-cache", "local/pilot/canonical"])
        self.assertEqual(retarget.call_args.kwargs["setup_cache"], Path("local/pilot/canonical"))


if __name__ == "__main__":
    unittest.main()
