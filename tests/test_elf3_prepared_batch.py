import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from elf3_umr_asset import digest
from run_elf3_prepared_batch import checked_completed, checked_motion, child, retarget_command


class QueueIntegrityTests(unittest.TestCase):
    def test_original_umr_then_model_optimization_command(self):
        command = retarget_command("trial", "source with spaces.npz", "output", "elf3.urdf", "upstream")
        self.assertEqual(command[command.index("--contact") + 1], "upstream")
        self.assertIn("source with spaces.npz", command)
        self.assertNotIn("--nonlinear-check", command)
        self.assertEqual(command[command.index("--targets") + 1], "raw")

    def test_unknown_solver_variant_rejected(self):
        with self.assertRaises(ValueError):
            retarget_command("trial", "source", "output", "elf3.urdf", "ignore_safety")

    def test_initialization_choice_is_explicit_and_checked(self):
        command=retarget_command('trial','source','output','elf3.urdf','upstream','foot-feasible')
        self.assertEqual(command[command.index('--initialization')+1],'foot-feasible')
        with self.assertRaises(ValueError):
            retarget_command('trial','source','output','elf3.urdf','upstream','shift_saved_frames')

    def test_completed_artifacts_verified_on_resume(self):
        with tempfile.TemporaryDirectory(prefix="elf3-queue-test-") as d:
            directory = Path(d)
            artifact = directory / "motion.npz"
            artifact.write_bytes(b"test motion bytes")
            job = {"source_sha256": "a" * 64}
            receipt = directory / "job_receipt.json"
            receipt.write_text(json.dumps({"job": job, "artifacts": {str(artifact): digest(artifact)}}))
            self.assertEqual(checked_completed(receipt, job)["job"], job)
            artifact.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                checked_completed(receipt, job)

    def test_wrong_job_not_silently_skipped(self):
        with tempfile.TemporaryDirectory(prefix="elf3-queue-test-") as d:
            receipt = Path(d) / "job_receipt.json"
            receipt.write_text(json.dumps({"job": {"source": "A"}, "artifacts": {}}))
            with self.assertRaises(ValueError):
                checked_completed(receipt, {"source": "B"})

    def test_motion_hash_required(self):
        with tempfile.TemporaryDirectory(prefix="elf3-queue-test-") as d:
            directory = Path(d)
            (directory / "motion.npz").write_bytes(b"a")
            (directory / "receipt.json").write_text(json.dumps({"motion_sha256": "0" * 64}))
            with self.assertRaises(ValueError):
                checked_motion(directory)

    def test_child_exit_and_exclusive_log(self):
        with tempfile.TemporaryDirectory(prefix="elf3-queue-test-") as d:
            log = Path(d) / "run.log"
            self.assertEqual(child([sys.executable, "-c", "print('complete')"], log, 5), 0)
            self.assertIn("complete", log.read_text())
            with self.assertRaises(FileExistsError):
                child([sys.executable, "-c", "pass"], log, 5)

    def test_timed_out_owned_child_terminated(self):
        with tempfile.TemporaryDirectory(prefix="elf3-queue-test-") as d:
            log = Path(d) / "run.log"
            self.assertEqual(child([sys.executable, "-c", "import time; time.sleep(20)"], log, .05), 124)


if __name__ == "__main__":
    unittest.main()
