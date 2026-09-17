#!/usr/bin/env python3
"""Resumable bounded ELF3 pilot queue: prepared full sources -> UMR -> refinement.

Only exact-shape Stage-I reuse is supported here. This is not the all-AMASS
raw-source worker and never starts training. Completed outputs are hash-checked
on resume; incomplete attempts are retained and retried into a new directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from elf3_umr_asset import digest, write_json
from elf3_source_reuse import load_replay_source, require_same_canonical
from diagnose_elf3_arm_motion import audit_candidate_arms
from run_elf3_umr_trial import UMR

ROOT = Path(__file__).resolve().parents[1]


def reviewed_status(receipt, arm_review):
    """Old collider gates alone can never produce an accepted training item."""
    failed = [k for k, passed in receipt["kinematic_checks"].items() if not passed]
    if not receipt["kinematic_candidate_pass"] or failed:
        return "quarantined", failed or ["legacy_candidate_gate_failed"]
    if arm_review["mesh_screen"]["visual_hull_over_5mm_frames"]:
        return "quarantined", ["visual_arm_core_hull_alarm_requires_mesh_review"]
    return "arm_review_required", ["source_fidelity_and_full_arm_review_pending"]


def unit_guard():
    matches = re.findall(r"(?:^|/)(bfm-elf3-prepared-batch-[A-Za-z0-9_-]+\.service)(?:/|$)",
                         Path("/proc/self/cgroup").read_text(), re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError("Use a finite owned bfm-elf3-prepared-batch-*.service")
    result = subprocess.check_output(["systemctl", "--user", "show", matches[0], "-p", "KillMode",
        "-p", "Restart", "-p", "RuntimeMaxUSec", "-p", "MemoryMax", "-p", "TasksMax"], text=True, timeout=10)
    props = dict(line.split("=", 1) for line in result.splitlines())
    if (props.get("KillMode") != "control-group" or props.get("Restart") != "no"
            or any(props.get(k) in (None, "", "0", "infinity") for k in ("RuntimeMaxUSec", "MemoryMax", "TasksMax"))):
        raise RuntimeError("Finite runtime/memory/tasks with entire-control-group cleanup required")
    return matches[0]


def checked_motion(directory):
    directory = Path(directory)
    receipt = json.loads((directory / "receipt.json").read_text())
    if digest(directory / "motion.npz") != receipt["motion_sha256"]:
        raise ValueError("Saved motion changed after audit")
    return receipt


def checked_completed(path, expected_job):
    result = json.loads(path.read_text())
    if result["job"] != expected_job:
        raise ValueError("Completed job identity changed")
    for filename, checksum in result["artifacts"].items():
        if digest(filename) != checksum:
            raise ValueError("Completed job artifact changed")
    return result


def retarget_command(trial, source, output, urdf, contact, initialization='human-root'):
    if contact not in ("upstream", "recover"):
        raise ValueError("Unknown Stage II contact variant")
    if initialization not in ('human-root','foot-feasible'):
        raise ValueError('Unknown explicit initial-pose recipe')
    return [sys.executable, str(ROOT / "scripts/replay_elf3_umr_stage2.py"),
            "--trial", str(trial), "--source", str(source), "--output", str(output),
            "--urdf", str(urdf), "--contact", contact, "--margin-m", "0", "--targets", "raw",
            '--initialization', initialization]


def child(command, logfile, timeout):
    """Own the subprocess session; cleanup also covers any spawned descendants."""
    with logfile.open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            # A reaped direct child does not prove its session has no descendants.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run(args):
    unit = unit_guard()
    if not 1 <= len(args.sources) <= 8 or not 10 <= args.max_iterations <= 1000:
        raise ValueError("Prepared pilot queue supports 1..8 complete motions and <=1000 refinement iterations")
    trial, output, urdf = (p.resolve() for p in (args.trial, args.output, args.urdf))
    inputs = json.loads((trial / "inputs.json").read_text())
    reference = load_replay_source(inputs["source"], require_full=True)
    jobs, seen = [], set()
    for source in args.sources:
        source = source.resolve()
        source_hash = digest(source)
        if source_hash in seen:
            raise ValueError("Duplicate prepared source in pilot queue")
        seen.add(source_hash)
        candidate = load_replay_source(source, require_full=True)
        identity = require_same_canonical(reference, candidate)
        frames = len(candidate["times"])
        if not 3 <= frames <= 1501:
            raise ValueError("Trajectory-refinement pilot supports <=1501 full frames; never crop to fit")
        jobs.append({"source": str(source), "source_sha256": source_hash, "frames": frames,
                     "raw_source_sha256": candidate["metadata"]["source_sha256"],
                     "raw_source": candidate["metadata"]["source_file"],
                     "canonical_identity_sha256": identity["sha256"]})
        del candidate
    del reference
    protected = {str(p): digest(p) for p in (Path(__file__),
        ROOT / "scripts/replay_elf3_umr_stage2.py", ROOT / "scripts/refine_elf3_umr_trajectory.py",
        ROOT / "scripts/elf3_source_reuse.py", ROOT / "scripts/elf3_morphology_targets.py",
        ROOT / "scripts/diagnose_elf3_arm_motion.py",
        ROOT / "scripts/run_elf3_umr_trial.py", ROOT / "scripts/umr_smplx_source.py",
        trial / "inputs.json", trial / "receipt.json", trial / "motion.npz",
        trial / "setup/bodies.npz", trial / "setup/correspondence.npz", urdf)}
    plan = {"schema": "bfm.elf3_prepared_pilot_queue/1", "trial": str(trial), "jobs": jobs,
            "protected_inputs": protected, "urdf": str(urdf), "max_iterations": args.max_iterations,
            "retarget_settings": {"contact": args.contact, "recover_margin_m": 0., "targets": "raw",
                                  'initialization':args.initialization},
            "refinement_collision_weight": 10000., "training_auto_start": False,
            "full_arm_review_required": True,
            "scope": "prepared_complete_motions_with_exact_shared_ELF3_canonical_correspondence"}
    if output.exists():
        if not args.resume or output.is_symlink() or json.loads((output / "queue.json").read_text()) != plan:
            raise ValueError("Existing queue requires --resume and identical sources, code and settings")
    else:
        output.mkdir(parents=True)
        write_json(output / "queue.json", plan)
    results = []
    for index, job in enumerate(jobs):
        folder = output / f"job_{index:03d}"
        folder.mkdir(exist_ok=True)
        completed = folder / "job_receipt.json"
        if completed.exists():
            results.append(checked_completed(completed, job))
            print(f"[resume] {index}: verified {results[-1]['status']}", flush=True)
            continue
        if digest(job["source"]) != job["source_sha256"]:
            raise ValueError("Source changed since queue preflight")
        attempt = next((folder / f"attempt_{k:03d}" for k in range(100)
                        if not (folder / f"attempt_{k:03d}").exists()), None)
        if attempt is None:
            raise RuntimeError("Too many incomplete attempts; inspect queue manually")
        attempt.mkdir()
        raw, refined = attempt / "raw", attempt / "refined"
        print(f"[start] {index}: {job['raw_source']} ({job['frames']} frames)", flush=True)
        command = retarget_command(trial, job["source"], raw, urdf, args.contact,args.initialization)
        code = child(command, attempt / "retarget.log", 240)
        status, reason, motion = "quarantined", f"retarget_exit_{code}", None
        if code == 0:
            raw_receipt = checked_motion(raw)
            if raw_receipt["solve_failures"] or raw_receipt["warmup_failures"]:
                reason = "input_umr_solver_failure"
            else:
                code = child([sys.executable, str(ROOT / "scripts/refine_elf3_umr_trajectory.py"),
                    "--source-run", str(raw), "--stage1", str(trial), "--urdf", str(urdf),
                    "--output", str(refined), "--max-iterations", str(args.max_iterations)],
                    attempt / "refine.log", 600)
                reason = f"refinement_exit_{code}"
                if code == 0:
                    receipt = checked_motion(refined)
                    motion = str(refined / "motion.npz")
                    if str(UMR) not in sys.path:
                        sys.path.insert(0, str(UMR))
                    arm_review = audit_candidate_arms(refined)
                    write_json(attempt / "arm_review.json", arm_review)
                    status, reason = reviewed_status(receipt, arm_review)
        for filename, checksum in protected.items():
            if digest(filename) != checksum:
                raise ValueError("Pinned queue code/input changed during execution")
        artifacts = {str(p): digest(p) for p in attempt.rglob("*") if p.is_file()}
        result = {"schema": "bfm.elf3_prepared_pilot_job/1", "job": job, "status": status,
                  "reason": reason, "motion": motion, "artifacts": artifacts,
                  "arm_review_complete": False,
                  "policy_inference": False, "promoted_to_training": False}
        write_json(completed, result)
        results.append(result)
        print(f"[done] {index}: {status} {reason}", flush=True)
    summary = {"schema": "bfm.elf3_prepared_pilot_result/1", "jobs": len(results),
               "frames": sum(j["frames"] for j in jobs), "by_status": dict(Counter(r["status"] for r in results)),
               "queue_sha256": digest(output / "queue.json"), "training_started": False,
               "all_library_processed": False, "unit": unit}
    final = output / "summary.json"
    if not final.exists():
        write_json(final, summary)
    else:
        saved = json.loads(final.read_text())
        if {k: v for k, v in saved.items() if k != "unit"} != {k: v for k, v in summary.items() if k != "unit"}:
            raise ValueError("Saved queue summary disagrees with verified job receipts")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial", required=True, type=Path)
    p.add_argument("--sources", required=True, type=Path, nargs="+")
    p.add_argument("--urdf", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-iterations", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--contact", choices=("upstream", "recover"), default="upstream",
                   help="Original UMR plus soft trajectory optimization; recover retained only for ablation")
    p.add_argument('--initialization',choices=('human-root','foot-feasible'),default='human-root')
    run(p.parse_args())
