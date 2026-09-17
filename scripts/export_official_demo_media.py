#!/usr/bin/env python3
"""Export only the three verified self-authored demo videos, never AMASS media."""
import argparse
import json
from pathlib import Path
import shutil

from elf3_umr_asset import digest, write_json
from make_official_demo_sources import DEMO_NAMES
from run_official_elf3 import OFFICIAL_COMMIT, verify_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--preview-prefix", required=True)
    p.add_argument("--output", type=Path, default=Path("docs/media"))
    args = p.parse_args()
    manifest = json.loads((args.sources / "sources.json").read_text())
    generator = Path(__file__).with_name("make_official_demo_sources.py")
    if (manifest.get("schema") != "umr_elf3.authored_demo_sources/1"
            or manifest.get("generator_sha256") != digest(generator)):
        raise ValueError("Missing matching self-authored source generator evidence")
    sources = {r["name"]: r for r in manifest["rows"]}
    rows, copies = [], []
    for name in DEMO_NAMES:
        source = sources[name]
        folder = Path(args.run_prefix + name).resolve()
        preview = Path(args.preview_prefix + name).resolve()
        inputs = verify_inputs(folder)
        receipt = json.loads((folder / "receipt.json").read_text())
        audit = json.loads((folder / "audit.json").read_text())
        video = json.loads((preview / "preview.json").read_text())
        expected = args.sources / (name + ".npz")
        if (source.get("amass_used") is not False or source.get("source_kind") != "authored_parametric_smplx"
                or inputs["clock"].get("source_kind") != "authored_parametric_smplx"
                or Path(inputs["source"]).resolve() != expected.resolve()
                or digest(expected) != source["source_sha256"]
                or receipt.get("status") != "kinematic_result_requires_review"
                or receipt["inputs_sha256"] != digest(folder / "inputs.json")
                or receipt["motion_sha256"] != digest(folder / "motion.npz")
                or video["motion_sha256"] != receipt["motion_sha256"]
                or audit["motion_sha256"] != receipt["motion_sha256"]
                or video.get("mj_step_called") is not False):
            raise ValueError(f"Invalid source, official run or video provenance: {name}")
        files = {}
        for suffix in ("gif", "mp4"):
            src = preview / f"official_elf3.{suffix}"
            if video["files"][src.name] != digest(src) or not 0 < src.stat().st_size < 8 * (1 << 20):
                raise ValueError(f"Changed or oversized demo: {src}")
            filename = f"official_{name}.{suffix}"
            files[filename] = {"sha256": digest(src), "bytes": src.stat().st_size}
            copies.append((src, args.output / filename))
        rows.append({"name": name, "source_kind": "authored_parametric_smplx", "amass_used": False,
                     "source_sha256": source["source_sha256"], "motion_sha256": receipt["motion_sha256"],
                     "official_result_sha256": audit["official_result_sha256"],
                     "frames": audit["frames"], "fps": audit["fps"], "elapsed_s": receipt["elapsed_s"],
                     "qpos_unchanged_from_official_final": True, "official_lqr_filter_enabled": True,
                     "training_approved": False, "policy_inference": False, "files": files,
                     "screening": {k: audit[k] for k in (
                         "joint_limit_violation_frames", "joint_speed_over_urdf_limit_intervals",
                         "foot_visual_mesh_min_z_m", "original_collision_self_penetration_max_m")}})
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    for src, dest in copies:
        shutil.copyfile(src, dest)
    write_json(args.output / "manifest.json", {
        "schema": "umr_elf3.authored_demo_media/1", "upstream_commit": OFFICIAL_COMMIT,
        "source_generator_sha256": digest(generator), "rows": rows,
        "scope": "robot-only research visualization; no AMASS sources, robot meshes or body models included",
        "collision_screen_scope": "enabled original URDF colliders only, not full-body certification",
        "quality_claim": "pipeline execution examples only; not trained policies or physical task success"})
    print(f"Exported {len(rows)} verified self-authored examples to {args.output}")


if __name__ == "__main__":
    main()
