#!/usr/bin/env python3
"""Finite, resumable two-stage arm refinement; no automatic training approval.

Recipe is fixed across all inputs: 500-step original-collider spline seed,
then up to 1000 steps with visual-hull and wrist SE(3) fidelity terms. Both
stages fit the same original complete trajectory, never the preceding output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from elf3_source_reuse import load_replay_source, require_same_canonical
from elf3_umr_asset import digest, write_json
from run_elf3_prepared_batch import ROOT, checked_completed, checked_motion, child, unit_guard


def stage_command(source_run, stage1, urdf, output, initial=None):
    cmd = [sys.executable, str(ROOT / "scripts/refine_elf3_arm_spline.py"),
           "--source-run", str(source_run), "--stage1", str(stage1), "--urdf", str(urdf),
           "--output", str(output), "--knot-seconds", "0.12",
           "--max-iterations", "500" if initial is None else "1000"]
    if initial is not None:
        cmd += ["--initial-spline", str(initial), "--visual-screen-loss", "--wrist-pose-loss"]
    return cmd


def candidate_status(receipt):
    """Report numerical blockers without declaring model/policy acceptance."""
    audit = receipt["audit"]
    blockers = []
    if not receipt["optimizer_success"]:
        blockers.append("optimizer_not_converged")
    if not receipt["root_and_non_arm_unchanged"]:
        blockers.append("non_arm_trajectory_changed")
    for key in ("joint_limit_violation_frames", "joint_speed_over_urdf_limit_intervals",
                "foot_below_minus_5mm_frames", "original_collision_self_penetration_over_5mm_frames"):
        if audit[key]:
            blockers.append(key)
    if receipt["after_visual_screen"]["visual_hull_over_5mm_frames"]:
        blockers.append("visual_hull_alarm_requires_exact_mesh_review")
    for side, metrics in receipt["arm_metrics"].items():
        for key in ("angular_speed_rad_s", "angular_acceleration_rad_s2", "acceleration_m_s2"):
            if metrics["after"][key]["max"] > metrics["before"][key]["max"] + 1e-6:
                blockers.append(side + "_" + key + "_peak_regression")
    return ("quarantined" if blockers else "fidelity_and_physics_review_required"), blockers


def run(args):
    unit = unit_guard()
    started = time.monotonic()
    if not 1 <= len(args.source_runs) <= 8 or not 600 <= args.max_seconds <= 14400:
        raise ValueError("Bounded 1..8-motion queue with 600..14400-second budget required")
    stage1, urdf, output = (p.resolve() for p in (args.stage1, args.urdf, args.output))
    checked_motion(stage1)
    seed_inputs = json.loads((stage1 / "inputs.json").read_text())
    reference = load_replay_source(seed_inputs["source"], require_full=True)
    jobs = []
    for source_run in args.source_runs:
        source_run = source_run.resolve()
        receipt = checked_motion(source_run)
        inputs = json.loads((source_run / "inputs.json").read_text())
        if (receipt.get("schema") != "bfm.elf3_trajectory_refinement/1"
                or not receipt["kinematic_checks"]["input_umr_no_solver_failures"]
                or inputs["robot_geometry"] != seed_inputs["robot_geometry"]
                or digest(inputs["source"]) != inputs["source_sha256"]):
            raise ValueError("Expected full model-refined UMR source without input solver failures")
        source = load_replay_source(inputs["source"], require_full=True)
        identity = require_same_canonical(reference, source)
        if not 4 <= len(source["times"]) <= 1501:
            raise ValueError("Only complete 4..1501-frame sources, no cropping")
        jobs.append({"source_run": str(source_run), "source": inputs["source"],
                     "source_sha256": inputs["source_sha256"], "frames": len(source["times"]),
                     "raw_source_sha256": source["metadata"]["source_sha256"],
                     "canonical_sha256": identity["sha256"]})
    if len({j["raw_source_sha256"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate raw motion in arm queue")
    paths = [Path(__file__), urdf, Path(seed_inputs["source"]), Path(seed_inputs["robot_xml"])]
    paths += [ROOT / "scripts" / name for name in (
        "refine_elf3_arm_spline.py", "refine_elf3_umr_trajectory.py", "diagnose_elf3_arm_motion.py",
        "elf3_source_reuse.py", "elf3_umr_asset.py", "run_elf3_prepared_batch.py", "run_elf3_umr_trial.py", "umr_smplx_source.py")]
    paths += [stage1 / name for name in ("inputs.json", "receipt.json", "motion.npz", "setup/bodies.npz", "setup/correspondence.npz")]
    for job in jobs:
        paths += [Path(job["source_run"]) / name for name in ("inputs.json", "receipt.json", "motion.npz")]
        paths.append(Path(job["source"]))
    protected = {str(p): digest(p) for p in paths}
    plan = {"schema": "bfm.elf3_arm_batch_plan/1", "jobs": jobs, "protected": protected,
            "stage1": str(stage1), "knot_seconds": .12, "seed_iterations": 500, "final_iterations": 1000,
            "max_seconds": args.max_seconds, "training_approved": False}
    if output.exists():
        if not args.resume or output.is_symlink() or json.loads((output / "plan.json").read_text()) != plan:
            raise ValueError("Resume needs the exact same recipe, inputs and code")
    else:
        output.mkdir(parents=True)
        write_json(output / "plan.json", plan)
    results = []
    for index, job in enumerate(jobs):
        folder = output / f"job_{index:03d}"
        folder.mkdir(exist_ok=True)
        completed = folder / "job_receipt.json"
        if completed.exists():
            results.append(checked_completed(completed, job))
            continue
        attempt = next(folder / f"attempt_{i:03d}" for i in range(100) if not (folder / f"attempt_{i:03d}").exists())
        attempt.mkdir()
        warm, final = attempt / "warm", attempt / "final"
        print(f"[start] {index}: {job['source_run']} ({job['frames']} frames)", flush=True)
        reason, status = [], "quarantined"
        for name, target, initial, timeout in (("warm", warm, None, 900), ("final", final, warm, 1500)):
            remaining = args.max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Arm batch global time budget exhausted; partial output retained")
            code = child(stage_command(job["source_run"], stage1, urdf, target, initial),
                         attempt / f"{name}.log", min(timeout, remaining))
            if code != 0:
                reason = [f"{name}_exit_{code}"]
                break
            receipt = checked_motion(target)
            if name == "final":
                status, reason = candidate_status(receipt)
        for path, checksum in protected.items():
            if digest(path) != checksum:
                raise ValueError("Protected input/code changed during arm batch")
        result = {"schema": "bfm.elf3_arm_batch_job/1", "job": job, "status": status, "reason": reason,
                  "final_run": str(final) if (final / "receipt.json").exists() else None,
                  "artifacts": {str(p): digest(p) for p in attempt.rglob("*") if p.is_file()},
                  "training_approved": False, "physical_tracking_validated": False}
        write_json(completed, result)
        results.append(result)
        print(f"[done] {index}: {status} {reason}", flush=True)
    summary = {"schema": "bfm.elf3_arm_batch_result/1", "jobs": len(results),
               "frames": sum(j["frames"] for j in jobs), "results": results, "unit": unit,
               "training_approved": False, "physical_tracking_validated": False}
    if (output / "summary.json").exists():
        previous = json.loads((output / "summary.json").read_text())
        if {k:v for k,v in previous.items() if k != "unit"} != {k:v for k,v in summary.items() if k != "unit"}:
            raise ValueError("Saved arm summary changed")
    else:
        write_json(output / "summary.json", summary)
    print(json.dumps({"jobs": len(results), "statuses": [r["status"] for r in results]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-runs", type=Path, nargs="+", required=True)
    p.add_argument("--stage1", type=Path, required=True)
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=3600)
    p.add_argument("--resume", action="store_true")
    run(p.parse_args())
