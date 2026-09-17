#!/usr/bin/env python3
"""ELF3-specific, bounded UMR mesh-source pilot; kinematic reference, not a policy.

Uses the existing unofficial pinned UMR network and Mink solver. Human surfaces
may be reused; G1 correspondences and 29-DoF exports are deliberately not reused.
Run in a bounded user service, with CPU/BLAS thread counts explicitly set.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

from elf3_umr_asset import TPOSE, digest, verify_assets, write_json
from umr_smplx_source import (SEGMENTS, SmplxSurfaceHuman, environment_packages,
                              load_prepared_source, model_geometry_fingerprint,
                              verify_umr_checkout)

ROOT = Path(__file__).resolve().parents[1]
UMR = ROOT / "external/umr_trial_20260908"


def elf3_config(xml, points, epochs):
    from umr.config import load_config
    cfg = load_config("g1")
    # Replace the complete morphology block, not merely the XML filename.
    cfg["robot"] = {"name": "elf3_dof31", "xml": str(xml), "tpose_joints": TPOSE,
                    "marker_body_prefixes": [], "marker_body_suffixes": [],
                    "foot_name_keys": ["ankle"],
                    "foot_bodies": ["l_ankle_x_link", "r_ankle_x_link"]}
    cfg["sampling"]["n_points"] = points
    cfg["correspondence"]["epochs"] = epochs
    cfg["correspondence"]["device"] = "cpu"
    cfg["retarget"]["tpose_offset"] = 0.
    # max_velocity in this upstream YAML is not wired to the solver; do not
    # pretend it enforces saved-frame rates. Audit real finite differences.
    cfg["retarget"].pop("max_velocity", None)
    return cfg


def anatomical_metrics(segment, body_ids, body_names):
    from umr.correspondence.evaluate import anatomical_consistency
    # Upstream metric recognizes left_/right_, while ELF3 uses l_/r_. Normalize
    # metric-only labels, never robot names, motion columns or learned targets.
    names = ["left_" + n[2:] if n.startswith("l_") else
             "right_" + n[2:] if n.startswith("r_") else n for n in body_names]
    overall, per_segment, _ = anatomical_consistency(segment, SEGMENTS, body_ids, names)
    return {"overall": overall, "per_segment": per_segment,
            "name_normalization": "metric_only_l_to_left_r_to_right"}


def elf3_mesh_sole_points(model, foot_name_keys=None):
    """Exact mesh convex-hull extreme vertices for the linear floor constraint.

    Pinned UMR's default helper only handles box/sphere soles; the supplied ELF3
    has mesh-only feet. Hull extreme points preserve extrema under any rotation,
    without adding invented collision boxes or changing the original geometry.
    """
    from umr.bodies.surface import geom_mesh_body_local
    bodies, points = [], []
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        if model.geom_group[g] != 1 or model.body(body).name not in ("l_ankle_x_link", "r_ankle_x_link"):
            continue
        vertices = np.asarray(geom_mesh_body_local(model, g).convex_hull.vertices)
        bodies.extend([body] * len(vertices))
        points.extend(vertices)
    if len(set(bodies)) != 2:
        raise ValueError("ELF3 floor constraint requires both actual foot meshes")
    return np.asarray(bodies, dtype=np.int64), np.asarray(points), np.zeros(len(points))


def motion_audit(model, qpos, fps, urdf):
    import mujoco
    from umr.bodies.robot import geom_lowest_z
    if (qpos.ndim != 2 or qpos.shape[1] != 38 or len(qpos) < 2
            or not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Invalid ELF3 trajectory")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Invalid root quaternion")
    joints = [model.joint(j) for j in range(1, model.njnt)]
    names = [j.name for j in joints]
    urdf_root = ET.parse(urdf).getroot()
    source_joints = {j.get("name"): j for j in urdf_root.findall("joint") if j.get("type") != "fixed"}
    if len(joints) != 31 or set(names) != set(source_joints):
        raise ValueError("This audit is only for the complete ELF3")
    angles = qpos[:, [j.qposadr[0] for j in joints]]
    ranges = np.array([j.range for j in joints])
    excess = np.maximum(np.maximum(ranges[:, 0] - angles, angles - ranges[:, 1]), 0.)
    speeds = np.abs(np.diff(angles, axis=0)) * fps
    limits = np.array([float(source_joints[n].find("limit").get("velocity")) for n in names])
    data = mujoco.MjData(model)
    foot_geoms = [g for g in range(model.ngeom) if model.geom_group[g] == 1
                  and model.body(model.geom_bodyid[g]).name in ("l_ankle_x_link", "r_ankle_x_link")]
    body_geoms = [g for g in range(model.ngeom) if model.geom_group[g] == 1]
    feet, body_lows, penetration = [], [], []
    worst_pair = None
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        feet.append(min(geom_lowest_z(model, data, g) for g in foot_geoms))
        body_lows.append(min(geom_lowest_z(model, data, g) for g in body_geoms))
        worst = 0.
        for c in data.contact:
            if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2] and c.dist < 0:
                p = float(-c.dist)
                if p > max(penetration, default=0.) and p > worst:
                    worst_pair = [model.geom(c.geom1).name, model.geom(c.geom2).name]
                worst = max(worst, p)
        penetration.append(worst)
    missing = [l.get("name") for l in urdf_root.findall("link")
               if l.find("visual") is not None and l.find("collision") is None]
    return {
        "frames": len(qpos), "fps": fps, "duration_s": (len(qpos) - 1) / fps,
        "joint_limit_violation_frames": int(np.any(excess > 1e-6, axis=1).sum()),
        "joint_limit_max_excess_rad": float(excess.max()),
        "joint_speed_max_rad_s": float(speeds.max()),
        "joint_speed_over_urdf_limit_intervals": int(np.any(speeds > limits + 1e-6, axis=1).sum()),
        "joint_speed_max_by_name": dict(zip(names, speeds.max(axis=0).tolist())),
        "foot_visual_mesh_min_z_m": float(min(feet)),
        "foot_below_minus_5mm_frames": int((np.array(feet) < -.005).sum()),
        "whole_visual_mesh_min_z_m": float(min(body_lows)),
        "original_collision_self_penetration_max_m": float(max(penetration)),
        "original_collision_self_penetration_over_5mm_frames": int((np.array(penetration) > .005).sum()),
        "worst_original_collision_pair": worst_pair,
        "links_without_original_collision": missing,
        "collision_audit_scope": "only contacts enabled by supplied URDF; convex mesh hulls; NOT full-body certification",
        "root_translation_m": (qpos[-1, :3] - qpos[0, :3]).tolist(),
        "physical_tracking_validated": False, "training_approved": False,
        "no_posthoc_floor_shift_or_smoothing": True,
    }


def run(args):
    import mujoco
    import torch
    from scipy.spatial.transform import Rotation
    from umr.bodies.robot import RobotBody, RobotSpec
    from umr.bodies.surface import SurfacePointCloud, sample_model_surface
    from umr.paths import SetupLayout
    from umr.retarget.binding import LinkBinding
    from umr.retarget.pipeline import UMRRetargeter, select_correspondence_points
    from umr.stages import learn_correspondence
    from unittest.mock import patch

    source, xml, output, assets = [p.resolve() for p in (args.source, args.robot_xml, args.output, args.assets)]
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not 512 <= args.points <= 4096 or not 100 <= args.epochs <= 2500:
        raise ValueError("Pilot bounds: 512..4096 points, 100..2500 epochs")
    commit = verify_umr_checkout(UMR)
    verify_assets(assets)
    torch.set_num_threads(2)
    prepared = load_prepared_source(source)
    # Whole source used when <= 10 s; no silent trimming or best-window selection.
    if len(prepared["times"]) > 501:
        raise ValueError("Choose an explicit <= 10-second prepared source for this pilot")
    cfg = elf3_config(xml, args.points, args.epochs)
    robot = RobotBody(xml, RobotSpec.from_config(cfg.robot))
    if robot.model.nq != 38 or robot.model.body(1).name != "torso_link":
        raise ValueError("Expected ELF3 floating torso, not a G1 robot")
    human = SmplxSurfaceHuman(prepared, robot.height())
    sampled = select_correspondence_points(prepared["segment"], SEGMENTS, args.points,
                                          points=prepared["canonical_points"], method="fps")
    if len(sampled) != args.points:
        raise ValueError("Incorrect Stage I sample count")
    ids = prepared["binding_joint_ids"][sampled]
    rotations = human.data.xmat[ids].reshape(-1, 3, 3)
    canonical = prepared["canonical_points"][sampled] * human.scale
    local_pos = np.einsum("nji,nj->ni", rotations, canonical - human.data.xpos[ids])
    local_normal = np.einsum("nji,nj->ni", rotations, prepared["canonical_normals"][sampled])
    human_cloud = SurfacePointCloud(canonical, prepared["canonical_normals"][sampled], ids,
                                   np.full(len(ids), -1, dtype=np.int64), local_pos, local_normal,
                                   prepared["segment"][sampled], SEGMENTS)
    output.mkdir(parents=True)
    source_hash, xml_hash = digest(source), digest(xml)
    source_scripts = [Path(__file__), ROOT / "scripts/elf3_umr_asset.py", ROOT / "scripts/umr_smplx_source.py"]
    script_hashes = {str(p): digest(p) for p in source_scripts}
    geometry = model_geometry_fingerprint(robot.model)
    write_json(output / "inputs.json", {
        "source": str(source), "source_sha256": source_hash,
        "robot_xml": str(xml), "robot_xml_sha256": xml_hash,
        "umr_commit": commit, "scripts": script_hashes, "config": cfg,
        "robot_geometry": geometry, "source_metadata": prepared["metadata"],
    })
    print(f"[ELF3] 31 DoF, height={robot.height():.4f}m, source={len(prepared['times'])} frames", flush=True)
    robot_cloud = sample_model_surface(robot.model, robot.data, len(ids), geom_ids=robot.surface_geoms(),
                                       oversample=cfg.sampling["oversample"], cull_margin=cfg.sampling["cull_margin"],
                                       seed=0, segment_names=SEGMENTS)
    setup = SetupLayout(output / "setup").ensure()
    np.savez(setup.bodies, stamp=digest(output / "inputs.json"), robot_xml=str(xml),
             **human_cloud.to_dict("human_"), **robot_cloud.to_dict("robot_"))
    learn_correspondence(cfg, setup, epochs=args.epochs, device="cpu")
    with np.load(setup.correspondence, allow_pickle=True) as corr:
        # Only read locally produced trusted Stage I debug metadata with pickle.
        binding = LinkBinding.from_dict(corr)
        segment = np.array(corr["inherited_segment"])
        stage1 = dict(corr["stage1_metrics"])
    stage1.pop("anatomical_consistency", None)  # G1 name convention invalid here.
    stage1["elf3_anatomical_consistency"] = anatomical_metrics(segment, binding.body_ids, robot.body_names)

    class Elf3Retargeter(UMRRetargeter):
        def human_targets(self, frame):
            p, n = self.human.targets(frame)
            return p[sampled], n[sampled]

        def initialize_root(self, frame):
            self.robot.set_tpose()
            self.human.set_tpose()
            # ELF3 torso root is NOT the human pelvis/G1 root. Use chest joint
            # 9 plus the canonical shape offset for an initialization only.
            offset = self.robot.data.body("torso_link").xpos - self.human.data.xpos[9]
            self.human.set_frame(frame)
            forward = self.human.data.xmat[0].reshape(3, 3)[:, 2]
            if np.linalg.norm(forward[:2]) < .1:
                raise ValueError("Initial human heading undefined")
            r = Rotation.from_euler("z", np.arctan2(forward[1], forward[0]))
            q = self.robot.model.key_qpos[0].copy()
            q[:3] = self.human.data.xpos[9] + r.apply(offset)
            q[3:7] = r.as_quat(scalar_first=True)
            self.robot.set_qpos(q)

    kwargs = {k: cfg.retarget[k] for k in (
        "n_selected", "point_selection", "tpose_offset", "iterations", "dt", "damping", "solver",
        "trust_region", "trust_region_radius", "floor_height", "floor_band", "floor_margin",
        "contact_threshold", "contact_weight", "posture_cost", "self_collision")}
    # Scoped dependency injection for mesh-only soles. Restored even on exception;
    # the pinned repository and its on-disk files are never edited.
    with patch("umr.retarget.pipeline.sole_sample_points", elf3_mesh_sole_points):
        solver = Elf3Retargeter(robot, human, human_body_ids=ids, human_local_pos=local_pos,
                               human_local_normal=local_normal, robot_body_ids=binding.body_ids,
                               robot_local_pos=binding.local_pos, robot_local_normal=binding.local_normal,
                               segment=segment, segment_names=SEGMENTS, **kwargs)
    solver.initialize_root(0)
    warm = solver.solve_frame(0, iterations=30)
    start = time.monotonic()
    rows = []
    for frame in range(len(prepared["times"])):
        row = solver.solve_frame(frame)
        row["qpos"] = np.array(row["qpos"], copy=True)
        rows.append(row)
        if frame % 50 == 0:
            print(f"[stage2] {frame + 1}/{len(prepared['times'])} error={row['point_error'] * 1000:.1f}mm failures={row['failures']}", flush=True)
    qpos = np.stack([r["qpos"] for r in rows])
    names = [robot.model.joint(j).name for j in range(1, robot.model.njnt)]
    fps = float(prepared["fps"])
    # Save raw output even if its engineering audit fails. No G1 training export.
    np.savez_compressed(output / "motion.npz", qpos=qpos, fps=fps, times=prepared["times"],
                        dof_names=np.array(names), root_body="torso_link", quaternion_order="wxyz",
                        source_surface_indices=sampled,
                        point_error=np.array([r["point_error"] for r in rows]),
                        solve_failures=np.array([r["failures"] for r in rows]))
    audit = motion_audit(robot.model, qpos, fps, assets / "elf3.urdf")
    verify_assets(assets)
    if (digest(source) != source_hash or digest(xml) != xml_hash
            or any(digest(p) != h for p, h in script_hashes.items())
            or model_geometry_fingerprint(mujoco.MjModel.from_xml_path(str(xml))) != geometry
            or verify_umr_checkout(UMR) != commit):
        raise ValueError("Protected inputs changed during pilot")
    receipt = {"schema": "bfm.elf3_umr_trial/1", "status": "kinematic_trial_completed",
               "upstream_status": "pinned_unofficial_UMR_reimplementation", "umr_commit": commit,
               "root_body": "torso_link", "dof": 31, "frames": len(qpos),
               "stage1": stage1, "stage1_points": args.points, "stage1_epochs": args.epochs,
               "warmup_solver_failures": warm["failures"],
               "stage2_solver_failures": sum(r["failures"] for r in rows),
               "stage2_and_audit_seconds": time.monotonic() - start,
               "point_error_mean_m": float(np.mean([r["point_error"] for r in rows])),
               "human_scale": human.scale, "human_constant_ground_offset_m": human.ground_offset,
               "floor_reference": "actual_ELF3_foot_mesh_convex_hull_extreme_vertices",
               "audit": audit, "protected_inputs_rechecked": True,
               "policy_inference": False, "promoted_to_training": False,
               "motion_sha256": digest(output / "motion.npz"),
               "environment": environment_packages(("numpy", "scipy", "mujoco", "torch", "mink", "trimesh"))}
    write_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def preview(args):
    """Render raw qpos without physics stepping; label every frame accordingly."""
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw
    output = args.output.resolve()
    inputs = json.loads((output / "inputs.json").read_text())
    receipt = json.loads((output / "receipt.json").read_text())
    if digest(output / "motion.npz") != receipt["motion_sha256"]:
        raise ValueError("Motion changed since audit")
    if digest(inputs["robot_xml"]) != inputs["robot_xml_sha256"]:
        raise ValueError("Robot XML changed")
    model = mujoco.MjModel.from_xml_path(inputs["robot_xml"])
    if model_geometry_fingerprint(model) != inputs["robot_geometry"]:
        raise ValueError("Robot geometry changed")
    with np.load(output / "motion.npz", allow_pickle=False) as z:
        qpos, fps = z["qpos"], float(z["fps"])
    mp4, gif = output / "elf3_umr_preview.mp4", output / "elf3_umr_preview.gif"
    if mp4.exists() or gif.exists():
        raise FileExistsError("Preview files already exist")
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.2, 135, -15
    camera.lookat[:] = np.r_[qpos[:, :2].mean(axis=0), .85]
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    data = mujoco.MjData(model)
    frames = []
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        with imageio.get_writer(str(mp4), fps=fps, codec="libx264", quality=7,
                                ffmpeg_params=["-threads", "2", "-movflags", "+faststart"]) as writer:
            for i, q in enumerate(qpos):
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                im = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, 960, 53), fill=(20, 23, 30))
                draw.text((14, 9), "ELF3 31 DoF | UMR neural correspondence + IK | KINEMATIC ONLY", fill="white")
                draw.text((14, 31), f"AMASS motion reference | {i / fps:.2f}s | NOT a trained policy / NOT object manipulation", fill=(255, 210, 100))
                writer.append_data(np.asarray(im))
                if i % 5 == 0:
                    frames.append(im.resize((640, 480)))
                if i in {0, len(qpos) // 2, len(qpos) - 1}:
                    im.save(output / f"frame_{i:04d}.png")
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=round(5000 / fps), loop=0)
    write_json(output / "preview_receipt.json", {"kind": "kinematic_qpos_replay", "fps": fps,
               "frames": len(qpos), "mp4_sha256": digest(mp4), "gif_sha256": digest(gif),
               "policy_inference": False, "physics_step": False})
    print(mp4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--robot-xml", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--points", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=1500)
    p = sub.add_parser("preview")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl" if args.command == "preview" else "glfw")
    sys.path.insert(0, str(UMR))
    verify_umr_checkout(UMR)
    run(args) if args.command == "run" else preview(args)


if __name__ == "__main__":
    main()
