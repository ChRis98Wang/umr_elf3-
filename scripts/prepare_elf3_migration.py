#!/usr/bin/env python3
"""Read-only source audit and pinned joint contract for the ELF3 migration.

Writes a NEW local output directory only. Never promotes motions, edits a G1
index, executes downloaded Python, or starts a policy training job.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import shutil
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from elf3_umr_asset import ELF3_COMMIT, digest, verify_assets, write_json
from umr_smplx_source import load_amass, model_file_for_gender

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = "src/bxi_example_py_elf3/bxi_example_py_elf3/policies/joints.py"
DEMO_PATH = "src/bxi_example_py_elf3/bxi_example_py_elf3/bxi_example_demo.py"


def assignment(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise ValueError(f"Missing upstream assignment: {name}")


def parse_policy_parameters(source, demo):
    """Only literal AST values; no imports/eval/exec from untrusted downloaded code."""
    policy = tuple(ast.literal_eval(assignment(source, "ELF3_POLICY_JOINTS").args[0]))
    isaac = tuple(ast.literal_eval(assignment(source, "ELF3_ISAAC_JOINTS").args[0]))
    rows = ast.literal_eval(assignment(source, "ELF3_ISAAC_PARAMETERS").args[1])
    if len(policy) != 29 or len(set(policy)) != 29 or set(policy) != set(isaac):
        raise ValueError("Unexpected upstream ELF3 body layout")
    if tuple(r[0] for r in rows) != isaac:
        raise ValueError("Parameter rows no longer match named Isaac order")
    params = {name: {"default_position": pos, "kp": kp, "kd": kd, "action_scale": scale,
                     "control_role": "body_policy"} for name, pos, kp, kd, scale in rows}
    heads = assignment(demo, "ELF3_COMMAND_DEFAULTS").args[0]
    if not isinstance(heads, ast.Dict):
        raise ValueError("Unexpected upstream head defaults structure")
    for key, node in zip(heads.keys, heads.values):
        name = ast.literal_eval(key)
        if name not in ("head_y_joint", "head_z_joint") or not isinstance(node, ast.Call):
            raise ValueError("Unexpected head default")
        fields = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        if set(fields) != {"position", "kp", "kd"}:
            raise ValueError("Unknown head parameter set")
        params[name] = {"default_position": fields["position"], "kp": fields["kp"], "kd": fields["kd"],
                        "action_scale": 0., "control_role": "separate_head_default"}
    if len(params) != 31:
        raise ValueError("Expected all 31 joints")
    return policy, isaac, params


def contract(assets, output):
    assets, output = assets.resolve(), output.resolve()
    verify_assets(assets)
    output.mkdir(parents=True, exist_ok=False)
    sources, hashes = {}, {}
    for path, filename in ((POLICY_PATH, "upstream_joints.py"), (DEMO_PATH, "upstream_demo.py")):
        url = f"https://raw.githubusercontent.com/bxirobotics/bxi_controller_ros2/{ELF3_COMMIT}/{path}"
        with urllib.request.urlopen(url, timeout=30) as f:
            content = f.read(1024 * 1024 + 1)
        if len(content) > 1024 * 1024:
            raise ValueError("Unexpected oversized upstream source")
        with (output / filename).open("xb") as f:
            f.write(content)
        sources[path] = content.decode()
        hashes[path] = {"sha256": digest(output / filename), "url": url}
    policy, isaac, params = parse_policy_parameters(sources[POLICY_PATH], sources[DEMO_PATH])
    urdf = ET.parse(assets / "elf3.urdf").getroot()
    joint_rows = []
    for j in urdf.findall("joint"):
        if j.get("type") == "fixed":
            continue
        name = j.get("name")
        row = {"name": name, "type": j.get("type"), "parent": j.find("parent").get("link"),
               "child": j.find("child").get("link"),
               "axis": [float(x) for x in j.find("axis").get("xyz").split()],
               "origin": dict(j.find("origin").attrib),
               "limit": {k: float(v) for k, v in j.find("limit").attrib.items()},
               **params[name]}
        if not row["limit"]["lower"] <= row["default_position"] <= row["limit"]["upper"]:
            raise ValueError(f"Upstream default violates supplied URDF: {name}")
        if row["kp"] <= 0 or row["kd"] < 0 or row["action_scale"] < 0:
            raise ValueError("Invalid actuator parameters")
        joint_rows.append(row)
    if len(joint_rows) != 31 or {r["name"] for r in joint_rows} != set(params):
        raise ValueError("URDF and parameter-table joints disagree")
    body_names = ["torso_link", "l_hip_x_link", "l_knee_y_link", "l_ankle_x_link",
                  "r_hip_x_link", "r_knee_y_link", "r_ankle_x_link", "waist_z_link",
                  "l_shoulder_x_link", "l_elbow_y_link", "l_wrist_z_link",
                  "r_shoulder_x_link", "r_elbow_y_link", "r_wrist_z_link"]
    report = {"schema": "bfm.elf3_joint_contract/1", "upstream_commit": ELF3_COMMIT,
              "urdf": str(assets / "elf3.urdf"), "urdf_sha256": digest(assets / "elf3.urdf"),
              "sources": hashes, "root_body": "torso_link", "motion_anchor_body": "torso_link",
              "sparse_pelvis_body": "waist_z_link",
              "urdf_joint_order": [r["name"] for r in joint_rows],
              "upstream_policy_joint_order": policy, "upstream_isaac_joint_order": isaac,
              "body_policy_dof": 29, "separate_head_dof": 2, "joints": joint_rows,
              "tracking_body_names": body_names,
              "authority": {"kinematics_and_hard_limits": "pinned_URDF",
                            "default_position_kp_kd_action_scale": "pinned_repository_explicit_Isaac_policy_table",
                            "head_defaults": "pinned_repository_JointCommandDefaults"},
              "not_claimed": ["upstream_training_recipe_reproduced", "full_PhysX_asset_parity", "hardware_calibration"],
              "unavailable_parameters": ["motor_armature", "motor_friction", "contact_calibration"],
              "training_status": "NOT_STARTED_requires_validated_ELF3_dataset_and_simulation"}
    write_json(output / "joint_contract.json", report)
    print(json.dumps({"joint_contract": str(output / "joint_contract.json"), "joints": len(joint_rows),
                      "all_defaults_within_urdf_limits": True}, indent=2), flush=True)


def inventory(args):
    output, source_root = args.output.resolve(), args.source_root.resolve()
    if not source_root.is_dir():
        raise ValueError("Provide an existing, locally licensed AMASS source directory")
    if output == source_root or source_root in output.parents:
        raise ValueError("Keep inventory output outside the AMASS source directory")
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads(args.previous_inventory.read_text()) if args.previous_inventory else {"rows": []}
    split_lookup = {r["origin_id"]: r["split"] for r in previous["rows"]}
    rg = shutil.which("rg")
    if rg is not None:
        listed = subprocess.check_output([rg, "--files", str(source_root), "-g", "*.npz"], text=True)
        paths = sorted(Path(p) for p in listed.splitlines())
    else:  # User-service PATH need not contain Codex's bundled ripgrep.
        paths = sorted(source_root.rglob("*.npz"))
    if not 1 <= len(paths) <= 50000:
        raise ValueError("Unexpected source inventory size")
    rows, rejects = [], []
    seen = defaultdict(list)
    for i, path in enumerate(paths):
        origin = path.relative_to(source_root).with_suffix("").as_posix()
        try:
            if path.is_symlink() or path.stat().st_size > 512 * 1024 * 1024:
                raise ValueError("Redirected or oversized source")
            before = path.stat()
            raw = load_amass(path)
            source_hash = digest(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Source changed during audit")
            count, fps = len(raw["trans"]), raw["fps"]
            duration = (count - 1) / fps
            frames = int(np.floor(duration * 50 + 1e-9)) + 1
            model_ok, model_reason = True, None
            try:
                model_file_for_gender(args.body_model, raw["gender"])
            except (ValueError, OSError) as exc:
                model_ok, model_reason = False, str(exc)
            row = {"origin_id": origin, "source": str(path), "source_sha256": source_hash,
                   "dataset": origin.split("/")[0], "raw_bytes": after.st_size,
                   "source_frames": count, "source_fps": fps, "duration_s": duration,
                   "target_50hz_frames": frames, "gender": raw["gender"],
                   "shape_betas_sha256": __import__("hashlib").sha256(raw["betas"].tobytes()).hexdigest(),
                   "split": split_lookup.get(origin, "unassigned_new_source"),
                   "body_model_available": model_ok, "body_model_issue": model_reason,
                   "retarget_status": "pending_ELF3_quality_gate"}
            rows.append(row)
            seen[source_hash].append(row)
        except (OSError, ValueError, KeyError, EOFError) as exc:
            rejects.append({"origin_id": origin, "source": str(path), "reason": f"{type(exc).__name__}: {exc}"})
        if i % 250 == 0:
            print(f"[inventory] {i + 1}/{len(paths)} readable={len(rows)} rejected={len(rejects)}", flush=True)
    duplicate_groups = []
    for source_hash, group in seen.items():
        if len(group) > 1:
            duplicate_groups.append({"sha256": source_hash, "origins": [r["origin_id"] for r in group],
                                     "splits": sorted({r["split"] for r in group})})
    unique = [group[0] for group in seen.values()]
    report = {"schema": "bfm.elf3_source_inventory/1", "source_root": str(source_root),
              "previous_inventory_sha256": digest(args.previous_inventory) if args.previous_inventory else None,
              "npz_files": len(paths), "valid_smplx_motion_files": len(rows),
              "unique_source_bytes": len(unique), "rejected_count": len(rejects),
              "by_dataset": dict(Counter(r["dataset"] for r in rows)),
              "by_split": dict(Counter(r["split"] for r in rows)),
              "by_gender": dict(Counter(r["gender"] for r in rows)),
              "unique_duration_hours": sum(r["duration_s"] for r in unique) / 3600,
              "unique_target_frames": sum(r["target_50hz_frames"] for r in unique),
              "unique_motion_output_estimate_bytes_float32_qpos": sum(r["target_50hz_frames"] for r in unique) * 38 * 4,
              "missing_body_model_files": sum(not r["body_model_available"] for r in rows),
              "new_sources_need_split_assignment": sum(r["split"] == "unassigned_new_source" for r in rows),
              "rows": rows, "rejects": rejects, "duplicates": duplicate_groups,
              "data_promotion": False, "policy_training_started": False,
              "batch_strategy": "bounded_one_source_at_a_time; exact_shape_ELF3_correspondence_cache; preserve_full_clock; quarantine_failed_audits"}
    write_json(output / "inventory.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "rejects", "duplicates")}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    c = s.add_parser("contract")
    c.add_argument("--assets", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c = s.add_parser("inventory")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--source-root", type=Path, required=True)
    c.add_argument("--previous-inventory", type=Path,
                   help='Optional earlier inventory; preserve existing train/validation membership')
    c.add_argument("--body-model", type=Path, required=True)
    args = p.parse_args()
    contract(args.assets, args.output) if args.command == "contract" else inventory(args)


if __name__ == "__main__":
    main()
