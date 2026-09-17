import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plan_elf3_batch import plan
from prepare_elf3_migration import parse_policy_parameters

import elf3_joint_contract as elf3


class BatchPlanTests(unittest.TestCase):
    def plan(self, rows):
        return plan({"schema": "bfm.elf3_source_inventory/1", "rows": rows},
                    {"schema": "bfm.elf3_joint_contract/1", "urdf_joint_order": []})

    def row(self, name, split, digest="a" * 64):
        return {"origin_id": name, "source_sha256": digest, "split": split, "target_50hz_frames": 100}

    def test_keeps_validation_and_deduplicates(self):
        p = self.plan([self.row("A", "validation"), self.row("B", "unassigned_new_source")])
        self.assertEqual(p["unique_jobs"], 1)
        self.assertEqual(p["jobs"][0]["split"], "validation")
        self.assertEqual(p["target_frames"], 100)

    def test_rejects_existing_split_leak(self):
        with self.assertRaises(ValueError):
            self.plan([self.row("A", "validation"), self.row("B", "train")])

    def test_new_split_is_deterministic(self):
        rows = [self.row("A", "unassigned_new_source"), self.row("B", "unassigned_new_source", "b" * 64)]
        self.assertEqual(self.plan(rows), self.plan(rows[::-1]))
        self.assertFalse(self.plan(rows)["training_auto_start"])


CONTRACT = ROOT / "local/elf3_contract_20260915b/joint_contract.json"


@unittest.skipUnless(CONTRACT.exists(), "Optional pinned local contract not prepared")
class ContractTests(unittest.TestCase):
    def test_pinned_table_matches_urdf(self):
        c = elf3.load_elf3_contract(CONTRACT)
        self.assertEqual(len(c["joints"]), 31)
        self.assertEqual(c["body_policy_dof"], 29)
        self.assertEqual(c["motion_anchor_body"], "torso_link")
        self.assertEqual(c["sparse_pelvis_body"], "waist_z_link")

    def test_pelvis_as_reset_root_rejected(self):
        self.tampered(lambda c: c.update(motion_anchor_body="waist_z_link"))

    def tampered(self, change):
        with tempfile.TemporaryDirectory(prefix="elf3-contract-test-") as d:
            directory = Path(d)
            for name in ("upstream_joints.py", "upstream_demo.py"):
                shutil.copyfile(CONTRACT.parent / name, directory / name)
            c = json.loads(CONTRACT.read_text())
            change(c)
            p = directory / "joint_contract.json"
            p.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                elf3.load_elf3_contract(p)

    def test_g1_limit_substitution_rejected(self):
        self.tampered(lambda c: c["joints"][0]["limit"].update(velocity=37.))

    def test_joint_axis_substitution_rejected(self):
        self.tampered(lambda c: c["joints"][0].update(axis=[1, 0, 0]))

    def test_pd_substitution_rejected(self):
        self.tampered(lambda c: c["joints"][0].update(kp=300.))

    def test_head_dropped_rejected(self):
        self.tampered(lambda c: c["joints"].pop())

    def test_ast_parser_does_not_execute_upstream(self):
        source = (CONTRACT.parent / "upstream_joints.py").read_text()
        demo = (CONTRACT.parent / "upstream_demo.py").read_text()
        policy, isaac, parameters = parse_policy_parameters("raise RuntimeError('must not execute')\n" + source, demo)
        self.assertEqual(len(policy), 29)
        self.assertEqual(len(parameters), 31)
        self.assertEqual(parameters["head_z_joint"]["kp"], 16.747)


if __name__ == "__main__":
    unittest.main()
