import copy
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from elf3_source_reuse import STATIC_ARRAYS, SHAPE_FIELDS, human_stage1_identity, require_same_canonical, load_replay_source


class CanonicalIdentityTests(unittest.TestCase):
    def setUp(self):
        self.source = {k: np.arange(6, dtype=float).reshape(2, 3) for k in STATIC_ARRAYS}
        self.source["metadata"] = {k: "test" for k in SHAPE_FIELDS}

    def test_motion_paths_and_frames_do_not_retrain_static_network(self):
        candidate = copy.deepcopy(self.source)
        candidate["sequence_points"] = np.zeros((100, 2, 3))
        candidate["metadata"].update(source_file="different_take.npz", frames=100)
        self.assertEqual(require_same_canonical(self.source, candidate), human_stage1_identity(self.source))

    def test_every_canonical_array_participates(self):
        for field in STATIC_ARRAYS:
            candidate = copy.deepcopy(self.source)
            candidate[field][0, 0] += 1e-10
            with self.subTest(field=field), self.assertRaises(ValueError):
                require_same_canonical(self.source, candidate)

    def test_every_shape_metadata_field_participates(self):
        for field in SHAPE_FIELDS:
            candidate = copy.deepcopy(self.source)
            candidate["metadata"][field] = "changed"
            with self.subTest(field=field), self.assertRaises(ValueError):
                require_same_canonical(self.source, candidate)

    def test_actor_folder_does_not_authorize_different_shape(self):
        candidate = copy.deepcopy(self.source)
        self.source["metadata"]["source_file"] = candidate["metadata"]["source_file"] = "same_actor/take.npz"
        candidate["betas"][0, 0] = 7.
        with self.assertRaises(ValueError):
            require_same_canonical(self.source, candidate)

    def test_missing_field_rejected(self):
        del self.source["betas"]
        with self.assertRaises(ValueError):
            human_stage1_identity(self.source)


PREPARED = ROOT / "local/umr_practical_source_v7_practical-v7-prepare-20260913a"


@unittest.skipUnless(PREPARED.exists(), "Optional licensed local motion surfaces")
class FullSourceReuseTests(unittest.TestCase):
    def test_same_actual_shape_different_complete_motion(self):
        first = load_replay_source(PREPARED / "origin_002/prepared.npz", require_full=True)
        second = load_replay_source(PREPARED / "origin_000/prepared.npz", require_full=True)
        self.assertEqual(len(second["times"]), 504)
        require_same_canonical(first, second)

    def test_another_actual_actor_rejected(self):
        first = load_replay_source(PREPARED / "origin_002/prepared.npz", require_full=True)
        second = load_replay_source(PREPARED / "origin_006/prepared.npz", require_full=True)
        with self.assertRaises(ValueError):
            require_same_canonical(first, second)


if __name__ == "__main__":
    unittest.main()
