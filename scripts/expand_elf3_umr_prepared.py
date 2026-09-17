#!/usr/bin/env python3
"""Finite multi-shape ELF3 preparation queue, with explicit Stage-I adoption.

Uses already prepared complete SMPL-X sources, learns each exact canonical shape
once, then runs the same upstream-UMR + trajectory-optimization batch recipe.
No training, source overwrites or implicit G1 correspondence reuse. A running
owned Stage-I job can be adopted by explicit path+unit, without starting it twice.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from elf3_umr_asset import digest, verify_assets, write_json
from elf3_source_reuse import human_stage1_identity, load_replay_source
from run_elf3_prepared_batch import ROOT, checked_motion, child, unit_guard


def group_sources(sources, batch_plan):
    planned = {r["source_sha256"]: r for r in batch_plan["jobs"]}
    groups, seen = defaultdict(list), set()
    for source in sources:
        source = Path(source).resolve()
        prepared = load_replay_source(source, require_full=True)
        raw_hash = prepared["metadata"]["source_sha256"]
        if raw_hash in seen or raw_hash not in planned:
            raise ValueError("Duplicate source or motion absent from full-library split plan")
        seen.add(raw_hash)
        if not 3 <= len(prepared["times"]) <= 1501:
            raise ValueError("Prepared expansion does not truncate >1501-frame sources")
        identity = human_stage1_identity(prepared)["sha256"]
        groups[identity].append({"source": str(source), "source_sha256": digest(source),
                                 "raw_source_sha256": raw_hash,
                                 "origin_id": planned[raw_hash]["origin_id"],
                                 "split": planned[raw_hash]["split"], "frames": len(prepared["times"])})
    rows = []
    for identity, members in groups.items():
        if len(members) > 8:
            raise ValueError("This prepared expansion is limited to <=8 motions per exact shape")
        representative = min(members, key=lambda r: (r["frames"], r["source"]))
        if representative["frames"] > 501:
            raise ValueError("Current Stage-I pilot needs a complete representative <=501 frames")
        rows.append({"canonical_sha256": identity, "sources": members,
                     "representative": representative["source"]})
    if not 1 <= len(rows) <= 4:
        raise ValueError("Explicit expansion supports 1..4 canonical shapes")
    return rows


def validate_trial(trial, group, xml):
    receipt = checked_motion(trial)
    inputs = json.loads((trial / "inputs.json").read_text())
    source = Path(inputs["source"])
    cfg = inputs["config"]
    if (digest(source) != inputs["source_sha256"] or digest(xml) != inputs["robot_xml_sha256"]
            or Path(inputs["robot_xml"]).resolve() != xml
            or cfg["sampling"]["n_points"] != 4096 or cfg["correspondence"]["epochs"] != 2500
            or cfg["robot"]["name"] != "elf3_dof31"
            or human_stage1_identity(load_replay_source(source, require_full=True))["sha256"] != group["canonical_sha256"]):
        raise ValueError("Stage-I trial does not match the exact shape/model/learning recipe")
    # Presence of motion/receipt alone is not proof the learned artifacts exist.
    for name in ("bodies.npz", "correspondence.npz"):
        digest(trial / "setup" / name)
    return receipt


def run(args):
    unit = unit_guard()
    started = time.monotonic()
    if not 600 <= args.max_seconds <= 10800:
        raise ValueError("Explicit 10-minute..3-hour overall budget required")
    assets, xml, output = (p.resolve() for p in (args.assets, args.robot_xml, args.output))
    verify_assets(assets)
    plan_data = json.loads(args.batch_plan.read_text())
    if plan_data["schema"] != "bfm.elf3_batch_plan/1":
        raise ValueError("Wrong robot full-library plan")
    groups = group_sources(args.sources, plan_data)
    adopted = {}
    if args.adopt_trial is not None:
        if not args.adopt_unit or not re.fullmatch(r"bfm-elf3-stage1-[a-zA-Z0-9_-]+(?:\.service)?", args.adopt_unit):
            raise ValueError("Adopting a running job requires its explicit owned ELF3 Stage-I unit")
        trial = args.adopt_trial.resolve()
        info = json.loads((trial / "inputs.json").read_text())
        identity = human_stage1_identity(load_replay_source(info["source"], require_full=True))["sha256"]
        if identity not in {g["canonical_sha256"] for g in groups}:
            raise ValueError("Adopted Stage I does not belong to this expansion")
        adopted[identity] = {"trial": str(trial), "unit": args.adopt_unit}
    protected = {str(p): digest(p) for p in (Path(__file__), args.batch_plan.resolve(), xml,
        ROOT / "scripts/run_elf3_umr_trial.py", ROOT / "scripts/replay_elf3_umr_stage2.py",
        ROOT / "scripts/refine_elf3_umr_trajectory.py", ROOT / "scripts/run_elf3_prepared_batch.py",
        ROOT / "scripts/diagnose_elf3_arm_motion.py",
        ROOT / "scripts/elf3_source_reuse.py", ROOT / "scripts/umr_smplx_source.py")}
    recipe = {"schema": "bfm.elf3_prepared_shape_expansion/1", "groups": groups, "adopted": adopted,
              "protected": protected, "max_seconds": args.max_seconds, "policy_training": False,
              "stage1_points": 4096, "stage1_epochs": 2500, "stage2_contact": "upstream",
              "full_arm_review_required": True,
              "scope": "prepared_source_expansion_not_all_library"}
    if output.exists():
        if not args.resume or output.is_symlink() or json.loads((output / "plan.json").read_text()) != recipe:
            raise ValueError("Resume needs identical source/code/shape recipe and an explicit --resume")
    else:
        output.mkdir(parents=True)
        write_json(output / "plan.json", recipe)
    reports = []
    for index, group in enumerate(groups):
        folder = output / f"shape_{index:03d}"
        folder.mkdir(exist_ok=True)
        final = folder / "group_receipt.json"
        if final.exists():
            saved = json.loads(final.read_text())
            if saved["group"] != group:
                raise ValueError("Saved group identity changed")
            for filename, checksum in saved["artifacts"].items():
                if digest(filename) != checksum:
                    raise ValueError("Saved group artifact changed")
            reports.append(saved)
            print(f"[resume] shape {index}: verified", flush=True)
            continue
        adopted_job = adopted.get(group["canonical_sha256"])
        if adopted_job:
            trial = Path(adopted_job["trial"])
            wait_start = time.monotonic()
            while not (trial / "receipt.json").exists():
                info = subprocess.check_output(["systemctl", "--user", "show", adopted_job["unit"],
                                                "-p", "MainPID", "-p", "ActiveState"], text=True, timeout=10)
                properties = dict(line.split("=", 1) for line in info.splitlines())
                if properties.get("MainPID") == "0" or time.monotonic() - wait_start > 2400:
                    raise RuntimeError("Adopted Stage I stopped without a completed receipt")
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("Expansion overall budget exhausted while awaiting owned Stage I")
                print(f"[waiting] shape {index}: adopted {adopted_job['unit']}", flush=True)
                time.sleep(15)
        else:
            # Preserve interrupted training attempts, never overwrite a partial setup.
            trial = next((p for p in sorted(folder.glob("stage1_*")) if (p / "receipt.json").exists()), None)
            if trial is None:
                trial = next(folder / f"stage1_{i:03d}" for i in range(100) if not (folder / f"stage1_{i:03d}").exists())
                remaining = args.max_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Expansion overall budget exhausted")
                print(f"[learning] shape {index}: 4096 points, 2500 epochs, {len(group['sources'])} sources", flush=True)
                code = child([sys.executable, str(ROOT / "scripts/run_elf3_umr_trial.py"), "run",
                    "--source", group["representative"], "--robot-xml", str(xml), "--assets", str(assets),
                    "--output", str(trial), "--points", "4096", "--epochs", "2500"],
                    folder / f"{trial.name}.log", min(2400, remaining))
                if code != 0:
                    raise RuntimeError(f"Shape {index} Stage-I pipeline failed: {code}; partial artifacts retained")
        validate_trial(trial, group, xml)
        batch = folder / "batch"
        remaining = args.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Expansion overall budget exhausted")
        command = [sys.executable, str(ROOT / "scripts/run_elf3_prepared_batch.py"), "--trial", str(trial),
                   "--sources", *[r["source"] for r in group["sources"]], "--urdf", str(assets / "elf3.urdf"),
                   "--output", str(batch), "--contact", "upstream"]
        if batch.exists():
            command.append("--resume")
        log = next(folder / f"batch_{i:03d}.log" for i in range(100) if not (folder / f"batch_{i:03d}.log").exists())
        print(f"[retargeting] shape {index}: {len(group['sources'])} complete motions", flush=True)
        code = child(command, log, min(6000, remaining))
        if code != 0:
            raise RuntimeError(f"Shape {index} batch failed: {code}; resume will verify completed jobs")
        for filename, checksum in protected.items():
            if digest(filename) != checksum:
                raise ValueError("Expansion code/source changed while running")
        verify_assets(assets)
        summary = json.loads((batch / "summary.json").read_text())
        artifact_paths = list(batch.rglob("*")) + [trial / "inputs.json", trial / "receipt.json",
                         trial / "motion.npz", trial / "setup/bodies.npz", trial / "setup/correspondence.npz"]
        report = {"schema": "bfm.elf3_prepared_shape_result/1", "group": group, "trial": str(trial),
                  "summary": summary, "artifacts": {str(p): digest(p) for p in artifact_paths if p.is_file()},
                  "training_started": False, "arm_review_complete": False}
        write_json(final, report)
        reports.append(report)
        print(f"[done] shape {index}: {summary['by_status']}", flush=True)
    result = {"schema": "bfm.elf3_prepared_expansion_result/1", "shapes": len(reports),
              "motions": sum(len(r["group"]["sources"]) for r in reports),
              "kinematic_candidates": sum(r["summary"]["by_status"].get("kinematic_candidate", 0) for r in reports),
              "quarantined": sum(r["summary"]["by_status"].get("quarantined", 0) for r in reports),
              "arm_review_required": sum(r["summary"]["by_status"].get("arm_review_required", 0) for r in reports),
              "unit": unit, "training_started": False, "whole_library_processed": False}
    if not (output / "summary.json").exists():
        write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, nargs="+", required=True)
    p.add_argument("--batch-plan", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--robot-xml", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--adopt-trial", type=Path)
    p.add_argument("--adopt-unit")
    p.add_argument("--max-seconds", type=int, default=10200)
    p.add_argument("--resume", action="store_true")
    run(p.parse_args())
