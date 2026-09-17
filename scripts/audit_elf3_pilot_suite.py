#!/usr/bin/env python3
"""Re-audit saved ELF3 pilot motions and identify coverage without promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from elf3_umr_asset import digest, write_json
from elf3_source_reuse import load_replay_source
from run_elf3_umr_trial import UMR, motion_audit
from umr_smplx_source import model_geometry_fingerprint, verify_umr_checkout


def validate_motion_layout(arrays, prepared, order):
    if (arrays["qpos"].shape != (len(prepared["times"]), 38)
            or not np.isfinite(arrays["qpos"]).all()
            or len(order) != 31 or len(set(order)) != 31
            or arrays["dof_names"].tolist() != order
            or str(arrays["root_body"].item()) != "torso_link"
            or str(arrays["quaternion_order"].item()) != "wxyz"
            or float(arrays["fps"]) != 50.
            or not np.array_equal(arrays["times"], prepared["times"])):
        raise ValueError("ELF3 complete clock, all 31 named joints or root convention disagrees")


def run(args):
    import mujoco
    verify_umr_checkout(UMR)
    plan = json.loads(args.batch_plan.read_text())
    if plan["schema"] != "bfm.elf3_batch_plan/1":
        raise ValueError("Expected all-source ELF3 split plan")
    planned = {j["source_sha256"]: j for j in plan["jobs"]}
    rows, seen = [], set()
    for folder in args.runs:
        folder = folder.resolve()
        inputs = json.loads((folder / "inputs.json").read_text())
        receipt = json.loads((folder / "receipt.json").read_text())
        if (receipt["schema"] != "bfm.elf3_trajectory_refinement/1"
                or digest(folder / "motion.npz") != receipt["motion_sha256"]
                or digest(inputs["source"]) != inputs["source_sha256"]
                or digest(inputs["robot_xml"]) != inputs["robot_xml_sha256"]):
            raise ValueError("Refined motion or its source changed")
        source = load_replay_source(inputs["source"], require_full=True)
        source_hash = source["metadata"]["source_sha256"]
        if source_hash in seen or source_hash not in planned:
            raise ValueError("Suite repeats a source or is not represented in the full-library plan")
        seen.add(source_hash)
        with np.load(folder / "motion.npz", allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        validate_motion_layout(arrays, source, plan["joint_order"])
        model = mujoco.MjModel.from_xml_path(inputs["robot_xml"])
        if model_geometry_fingerprint(model) != inputs["robot_geometry"]:
            raise ValueError("Robot mesh changed")
        audit = motion_audit(model, arrays["qpos"], 50., args.urdf)
        # Recompute model measurements instead of trusting a cached pass flag.
        for key in ("joint_limit_violation_frames", "joint_speed_over_urdf_limit_intervals",
                    "foot_below_minus_5mm_frames", "original_collision_self_penetration_over_5mm_frames"):
            if audit[key] != receipt["audit"][key]:
                raise ValueError(f"Fresh geometry audit disagrees: {key}")
        if abs(audit["original_collision_self_penetration_max_m"] - receipt["audit"]["original_collision_self_penetration_max_m"]) > 1e-7:
            raise ValueError("Fresh collision depth disagrees with saved audit")
        parent_path = Path(receipt["source_run"]) / "receipt.json"
        if digest(parent_path) != receipt["protected_inputs"][str(parent_path)]:
            raise ValueError("Original Stage II receipt changed")
        parent = json.loads(parent_path.read_text())
        row = {"origin_id": planned[source_hash]["origin_id"], "split": planned[source_hash]["split"],
               "source_sha256": source_hash, "run": str(folder), "frames": len(arrays["qpos"]),
               "duration_s": audit["duration_s"], "motion_sha256": receipt["motion_sha256"],
               "receipt_sha256": digest(folder / "receipt.json"),
               "kinematic_candidate_pass": receipt["kinematic_candidate_pass"],
               "kinematic_checks": receipt["kinematic_checks"], "fresh_audit": audit,
               "raw_human_surface_error_mean_m": receipt["raw_human_surface_error_mean_m"],
               "surface_change_p95_m": receipt["surface_change_p95_m"],
               "stage2_contact_mode": parent.get("contact_mode", "upstream"),
               "source_solver_run": receipt["source_run"]}
        rows.append(row)
    result = {"schema": "bfm.elf3_pilot_suite/1", "rows": rows, "unique_motions": len(rows),
              "frames": sum(r["frames"] for r in rows), "duration_s": sum(r["duration_s"] for r in rows),
              "kinematic_candidates": sum(r["kinematic_candidate_pass"] for r in rows),
              "whole_library_unique_motions": plan["unique_jobs"],
              "whole_library_processed": False, "physical_tracking_validated": False,
              "training_started": False, "promoted_to_training": False,
              "batch_plan_sha256": digest(args.batch_plan), "audit_script_sha256": digest(__file__),
              "collision_scope": "only enabled original URDF colliders; 15 visual links lack original colliders"}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "suite.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, type=Path)
    p.add_argument("--batch-plan", required=True, type=Path)
    p.add_argument("--urdf", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    sys.path.insert(0, str(UMR))
    run(p.parse_args())
