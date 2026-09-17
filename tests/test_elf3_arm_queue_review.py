from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_elf3_prepared_batch import reviewed_status


class ArmQueueReviewTests(unittest.TestCase):
    def test_old_gates_do_not_grant_full_acceptance(self):
        receipt = {"kinematic_checks": {"old_gate": True}, "kinematic_candidate_pass": True}
        review = {"mesh_screen": {"visual_hull_over_5mm_frames": 0}}
        self.assertEqual(reviewed_status(receipt, review)[0], "arm_review_required")

    def test_hull_alarm_is_quarantined_not_claimed_exact_depth(self):
        receipt = {"kinematic_checks": {"old_gate": True}, "kinematic_candidate_pass": True}
        review = {"mesh_screen": {"visual_hull_over_5mm_frames": 1}}
        status, reasons = reviewed_status(receipt, review)
        self.assertEqual(status, "quarantined")
        self.assertIn("mesh_review", reasons[0])

    def test_failed_optimizer_still_quarantined(self):
        receipt = {"kinematic_checks": {"solver_converged": False}, "kinematic_candidate_pass": False}
        review = {"mesh_screen": {"visual_hull_over_5mm_frames": 0}}
        self.assertEqual(reviewed_status(receipt, review), ("quarantined", ["solver_converged"]))


if __name__ == "__main__":
    unittest.main()
