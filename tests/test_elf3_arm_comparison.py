from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_elf3_arm_candidate import fidelity_stats, load_pair


class FidelityTests(unittest.TestCase):
    def test_endpoint_change_not_diluted_by_body_points(self):
        before = {"tool_positions": np.zeros((6, 2, 3)), "wrist_positions": np.zeros((6, 2, 3)),
                  "wrist_rotations": np.tile(np.eye(3), (6, 2, 1, 1))}
        after = {k:v.copy() for k,v in before.items()}
        after["tool_positions"][:, 1, 0] = .12
        after["wrist_rotations"][:, 1] = Rotation.from_euler("z", .2).as_matrix()
        report = fidelity_stats(before, after)
        self.assertEqual(report["left"]["tip_displacement_m"]["max"], 0.)
        self.assertAlmostEqual(report["right"]["tip_displacement_m"]["p95"], .12)
        self.assertAlmostEqual(report["right"]["rotation_error_rad"]["max"], .2)


CANDIDATE = Path(__file__).resolve().parents[1] / "local/elf3_arm_spline_20260915c"
WHOLE_BODY = Path(__file__).resolve().parents[1] / 'local/elf3_whole_body_smoke_20260916a'


@unittest.skipUnless((CANDIDATE / "receipt.json").exists(), "Optional local candidate")
class PairTests(unittest.TestCase):
    def test_real_source_clock_and_nonarm_identity(self):
        model, clips, _, _ = load_pair(CANDIDATE)
        self.assertEqual(model.nq, 38)
        self.assertEqual(clips[0].shape, (328, 38))
        np.testing.assert_array_equal(clips[0][:, :7], clips[1][:, :7])

    @unittest.skipUnless((WHOLE_BODY/'receipt.json').exists(),'Optional whole-body smoke output')
    def test_explicit_whole_body_schema_allows_root_change(self):
        _,clips,_,receipt=load_pair(WHOLE_BODY)
        self.assertTrue(receipt['root_and_all_31_joints_optimized'])
        self.assertEqual(clips[0].shape,(362,38))
        self.assertFalse(np.array_equal(clips[0][:,:7],clips[1][:,:7]))

    @unittest.skipUnless((WHOLE_BODY/'receipt.json').exists(),'Optional whole-body smoke output')
    def test_whole_body_candidate_is_paired_with_correct_gui_action(self):
        from view_elf3_umr_suite import load_suite, display_label
        suite=Path(__file__).resolve().parents[1]/'local/elf3_pilot_suite_20260915a/suite.json'
        _,clips=load_suite(suite,[WHOLE_BODY])
        self.assertEqual(clips[5]['paired_index'],6)
        self.assertEqual(clips[6]['paired_index'],5)
        self.assertIn('全身优化候选',display_label(clips[6]))


if __name__ == "__main__":
    unittest.main()
