#!/usr/bin/env python3
"""Freeze an all-source ELF3 batch manifest without promoting data or training.

Keeps existing train/validation membership, deduplicates source bytes, and assigns
new source bytes deterministically (10% validation). Metadata-only files stay in
the inventory rejection report. Exact canonical shape cache identity is checked
at execution, never inferred solely from a folder name or beta fingerprint.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from elf3_umr_asset import digest, write_json


def plan(inventory, contract):
    if inventory["schema"] != "bfm.elf3_source_inventory/1" or contract["schema"] != "bfm.elf3_joint_contract/1":
        raise ValueError("Unexpected migration source schema")
    grouped = defaultdict(list)
    for row in inventory["rows"]:
        grouped[row["source_sha256"]].append(row)
    jobs = []
    for source_hash, group in sorted(grouped.items()):
        existing_splits = {r["split"] for r in group} - {"unassigned_new_source"}
        if len(existing_splits) > 1:
            raise ValueError("Duplicate source crosses old train/validation boundary")
        split = next(iter(existing_splits)) if existing_splits else (
            "validation" if int(source_hash[:8], 16) % 10 == 0 else "train")
        origin = min(group, key=lambda r: (r["split"] == "unassigned_new_source", r["origin_id"]))
        jobs.append({**origin, "split": split, "aliases": [r["origin_id"] for r in group],
                     "state": "pending_model_quality_validation", "output_robot": "elf3_dof31"})
    return {"schema": "bfm.elf3_batch_plan/1", "jobs": jobs, "unique_jobs": len(jobs),
            "by_split": dict(Counter(r["split"] for r in jobs)),
            "target_frames": sum(r["target_50hz_frames"] for r in jobs),
            "joint_order": contract["urdf_joint_order"], "root_body": "torso_link",
            "source_split_policy": "preserve_old; deduplicate_sha256; new_hash_prefix_mod10_validation",
            "full_duration_required": True, "stage1_points": 4096, "stage1_epochs": 2500,
            "bounded_materialization": "one_human_surface_source_at_a_time; retain_ELF3_qpos_and_receipts",
            "training_auto_start": False,
            "required_before_training": ["ELF3_correspondence_and_self_contact_quality_pass",
                                         "batch_motion_audits_and_quarantine",
                                         "IsaacLab_URDF_joint_and_FK_parity",
                                         "named_joint_motion_packaging",
                                         "small_physical_tracking_test"],
            "execution_status": "PLANNED_NOT_RETARGETED"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--contract", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    result = plan(json.loads(args.inventory.read_text()), json.loads(args.contract.read_text()))
    result["inventory_sha256"] = digest(args.inventory)
    result["contract_sha256"] = digest(args.contract)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "batch_plan.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "jobs"}, indent=2))


if __name__ == "__main__":
    main()
