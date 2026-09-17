import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_elf3_pilot_suite import validate_motion_layout


class MotionLayoutTests(unittest.TestCase):
    def setUp(self):
        self.order = [f"joint_{i}" for i in range(31)]
        self.source = {"times": np.arange(4) / 50.}
        self.arrays = {"times": self.source["times"].copy(), "qpos": np.zeros((4, 38)),
                       "dof_names": np.array(self.order), "fps": np.array(50.),
                       "root_body": np.array("torso_link"), "quaternion_order": np.array("wxyz")}

    def test_full_layout_passes(self):
        validate_motion_layout(self.arrays, self.source, self.order)

    def test_semantic_and_clock_mismatches_rejected(self):
        bad = {"root_body": np.array("pelvis"), "quaternion_order": np.array("xyzw"),
               "times": self.arrays["times"] + .02, "fps": np.array(60.),
               "dof_names": self.arrays["dof_names"][::-1], "qpos": np.zeros((4, 36))}
        for key, value in bad.items():
            candidate = copy.deepcopy(self.arrays)
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_motion_layout(candidate, self.source, self.order)

    def test_nonfinite_trajectory_rejected(self):
        self.arrays["qpos"][1, 10] = np.nan
        with self.assertRaises(ValueError):
            validate_motion_layout(self.arrays, self.source, self.order)


if __name__ == "__main__":
    unittest.main()
