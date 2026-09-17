from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_elf3_arm_batch import stage_command, candidate_status


class ArmBatchTests(unittest.TestCase):
    def test_both_stages_keep_same_source_and_clock(self):
        warm = stage_command("original path", "stage1", "urdf", "warm")
        final = stage_command("original path", "stage1", "urdf", "final", "warm")
        for command in (warm, final):
            self.assertEqual(command[command.index("--source-run") + 1], "original path")
            self.assertEqual(command[command.index("--knot-seconds") + 1], "0.12")
        self.assertNotIn("--visual-screen-loss", warm)
        self.assertIn("--visual-screen-loss", final)
        self.assertIn("--wrist-pose-loss", final)
        self.assertEqual(final[final.index("--initial-spline") + 1], "warm")

    def receipt(self):
        return {"optimizer_success": True, "root_and_non_arm_unchanged": True,
                "audit": dict.fromkeys(("joint_limit_violation_frames", "joint_speed_over_urdf_limit_intervals",
                       "foot_below_minus_5mm_frames", "original_collision_self_penetration_over_5mm_frames"), 0),
                "after_visual_screen": {"visual_hull_over_5mm_frames": 0},
                "arm_metrics": {side: {stage: {key: {"max": value} for key in (
                    "angular_speed_rad_s", "angular_acceleration_rad_s2", "acceleration_m_s2")}
                    for stage, value in (("before", 10.), ("after", 5.))} for side in ("left", "right")}}

    def test_no_automatic_training_pass_even_if_numerically_better(self):
        status, blockers = candidate_status(self.receipt())
        self.assertEqual(status, "fidelity_and_physics_review_required")
        self.assertEqual(blockers, [])

    def test_peak_regression_and_nonconvergence_isolated(self):
        receipt = self.receipt()
        receipt["optimizer_success"] = False
        receipt["arm_metrics"]["right"]["after"]["angular_speed_rad_s"]["max"] = 11.
        status, blockers = candidate_status(receipt)
        self.assertEqual(status, "quarantined")
        self.assertIn("optimizer_not_converged", blockers)
        self.assertIn("right_angular_speed_rad_s_peak_regression", blockers)


if __name__ == "__main__":
    unittest.main()
