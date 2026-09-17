#!/usr/bin/env python3
"""Isolated official UMR -> ELF3 adapter. No policy training or legacy solver.

Prepare a complete 50 Hz SMPL-X clip, call the unmodified pinned upstream,
and retain its final qpos verbatim. All generated/licensed artifacts stay local.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from elf3_umr_asset import TPOSE, digest, validate_urdf_fk, verify_assets, write_json

OFFICIAL_COMMIT = "e24fc070030dc0bb0b2c024ecb9f795a3995d725"
OFFICIAL_URL = "https://github.com/hanyang9/UMR"
SEGMENTATION_SHA256 = "bb69c10801205c9cfb5353fdeb1b9cc5ade53d14c265c3339421cdde8b9c91e7"


def verify_official(checkout):
    checkout = Path(checkout).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args], text=True, timeout=30).strip()
    if git("rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise ValueError("Official UMR revision differs from the reviewed pin")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Official tracked source has local modifications")
    for name in ("scripts/humanoid_retarget_pipeline.py", "humanoid_retarget_defaults.json",
                 "assets/smplx_parts_segm.pkl"):
        if not (checkout / name).is_file():
            raise FileNotFoundError(checkout / name)
    if digest(checkout / "assets/smplx_parts_segm.pkl") != SEGMENTATION_SHA256:
        raise ValueError("Fetch the pinned segmentation with git lfs pull; a pointer is not the asset")
    return checkout


def resample_source(source):
    """Only numeric/text NPZ fields; SO(3) interpolation, no crop/pose repair.

    AMASS can contain unrelated pickled marker labels. Never load those fields.
    The last output timestamp is at most one 50 Hz interval before the last input.
    """
    selected = {"poses", "trans", "betas", "gender", "surface_model_type",
                "mocap_frame_rate", "mocap_framerate", "fps", "source_kind"}
    with zipfile.ZipFile(source) as archive:
        names = [e.filename for e in archive.infolist()]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate NPZ members")
        total = sum(e.file_size for e in archive.infolist() if e.filename.removesuffix(".npy") in selected)
        if total > 512 * (1 << 20):
            raise ValueError("Source exceeds the bounded single-clip allocation")
    with np.load(source, allow_pickle=False) as z:
        if str(z["surface_model_type"].item()).lower() != "smplx":
            raise ValueError("This adapter requires an explicit AMASS SMPL-X source")
        poses, trans, betas = (np.asarray(z[k], dtype=np.float64) for k in ("poses", "trans", "betas"))
        gender = str(z["gender"].item()).lower()
        source_kind = str(z["source_kind"].item()) if "source_kind" in z else "amass_smplx"
        rate_key = next((k for k in ("mocap_frame_rate", "mocap_framerate", "fps") if k in z), None)
        if rate_key is None:
            raise ValueError("Source frame rate is required; no implicit 30 Hz")
        rate = float(np.asarray(z[rate_key]).item())
    if (poses.ndim != 2 or poses.shape[1] != 165 or not 3 <= len(poses) <= 2000000
            or trans.shape != (len(poses), 3) or betas.ndim != 1 or len(betas) < 10
            or gender not in {"neutral", "male", "female"}
            or not np.isfinite(rate) or not 0 < rate <= 1000
            or any(not np.isfinite(a).all() for a in (poses, trans, betas))):
        raise ValueError("Invalid complete SMPL-X pose/translation/shape/clock")
    duration = (len(poses) - 1) / rate
    count = math.floor(duration * 50 + 1e-9) + 1
    if not 2 <= count <= 90001:
        raise ValueError("Full motion exceeds frame budget (never truncated)")
    times = np.arange(count, dtype=np.float64) / 50
    source_times = np.arange(len(poses), dtype=np.float64) / rate
    # Floating point roundoff at an exact endpoint is not extrapolation.
    query = np.minimum(times, source_times[-1])
    sampled = np.empty((count, 165), dtype=np.float64)
    for joint in range(55):
        section = slice(joint * 3, joint * 3 + 3)
        sampled[:, section] = Slerp(source_times, Rotation.from_rotvec(poses[:, section]))(query).as_rotvec()
    translation = np.column_stack([np.interp(query, source_times, trans[:, i]) for i in range(3)])
    fields = {"poses": sampled, "trans": translation, "betas": betas,
              "gender": np.asarray(gender), "mocap_frame_rate": np.asarray(50.),
              "surface_model_type": np.asarray("smplx"), "output_up": np.asarray("z"),
              "source_kind": np.asarray(source_kind)}
    evidence = {"source_frames": len(poses), "source_fps": rate, "source_duration_s": duration,
                "frames": count, "fps": 50., "duration_s": float(times[-1]),
                "tail_residual_s": float(duration - times[-1]), "gender": gender, "source_kind": source_kind,
                "rotation_interpolation": "SO3_shortest_arc_slerp",
                "world_frame": "unchanged_AMASS_z_up", "full_duration": True,
                "betas_used_by_upstream": 10,
                "upstream_hand_face_behavior": "official SMPL-X path uses body pose and neutral hands/face"}
    return fields, evidence


def robot_config(xml, urdf):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(xml))
    tree = ET.parse(urdf).getroot()
    fk = validate_urdf_fk(tree, model)
    joints = {j.get("name"): j for j in tree.findall("joint") if j.get("type") != "fixed"}
    names = [model.joint(i).name for i in range(1, model.njnt)]
    for name, angle in TPOSE.items():
        limits = joints[name].find("limit")
        if not float(limits.get("lower")) <= angle <= float(limits.get("upper")):
            raise ValueError(f"T-pose violates original joint limits: {name}")
    cfg = {"name": "elf3_dof31", "xml": str(Path(xml).resolve()),
           "point_cloud_center": "body:waist_z_link", "tpose_qpos": dict(TPOSE),
           "xml_policy": {"add_freejoint_root": False},
           "joint_limits": {n: [float(joints[n].find("limit").get(k)) for k in ("lower", "upper")]
                            for n in names}, "visual_geom_policy": "visual"}
    return cfg, {"joint_names": names, "nq": model.nq, "nv": model.nv,
                 "root_body": "torso_link", "point_cloud_center_body": "waist_z_link", "urdf_fk": fk}


def prepare(args):
    checkout = verify_official(args.upstream)
    assets = Path(args.assets).resolve()
    verify_assets(assets)
    source, xml, body_model, out = [Path(p).absolute() for p in
                                    (args.source, args.robot_xml, args.body_model, args.output)]
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if body_model.suffix.lower() != ".npz" or not body_model.is_file():
        raise ValueError("Use your locally licensed SMPL-X .npz body model")
    fields, clock = resample_source(source)
    # The supplied model is neutral; never apply it to male/female data silently.
    if clock["gender"] != "neutral":
        raise ValueError("This initial ELF3 integration is limited to neutral SMPL-X")
    cfg_robot, robot = robot_config(xml, assets / "elf3.urdf")
    out.mkdir(parents=True)
    np.savez_compressed(out / "source_50hz.npz", **fields)
    cfg = {"robot": cfg_robot, "smplx_model_dir": str(body_model),
           "correspondence": {
               "dataset": {"out": str(out / "correspondence_dataset.npz"),
                           "smpl_models": [{"type": "smplx", "dir": str(body_model), "genders": ["neutral"]}]},
               "train": {"out_dir": str(out / "correspondence"), "device": "cuda"}},
           "motion": {"data": str(out / "source_50hz.npz"), "seq_key": "source_50hz",
                      "start": 0, "end": -1, "stride": 1, "max_frames": 0},
           # This is the official batch initialization mode, not our former optimizer.
           "solver": {"trajectory_warm_start_mode": "bidirectional"},
           "retarget": {"out": str(out / "official_result.npz"), "smplx_device": "cuda",
                        "smplx_batch_size": 32, "smplx_batch_size_max": 32},
           "view": {"enabled": False}}
    write_json(out / "official_config.json", cfg)
    paths = [source.resolve(), xml.resolve(), body_model.resolve(), assets / "elf3.urdf",
             assets / "provenance.json", out / "source_50hz.npz", out / "official_config.json",
             checkout / "humanoid_retarget_defaults.json", checkout / "assets/smplx_parts_segm.pkl",
             Path(__file__).resolve()]
    write_json(out / "inputs.json", {
        "schema": "umr_elf3.official_input/1", "backend": "official_hanyang9_UMR",
        "upstream": str(checkout), "upstream_url": OFFICIAL_URL, "upstream_commit": OFFICIAL_COMMIT,
        "assets": str(assets), "robot_xml": str(xml.resolve()), "source": str(source.resolve()),
        "protected_sha256": {str(p): digest(p) for p in paths}, "clock": clock, "robot": robot,
        "recipe": "official_defaults_4096_points_500_epochs_plus_batch_bidirectional",
        "official_lqr_filter_enabled": True, "legacy_refinement": False,
        "training_approved": False, "redistribution_permission": "UNCONFIRMED_DO_NOT_UPLOAD"})
    print(json.dumps({"prepared": str(out), "frames": clock["frames"], "robot": robot}, indent=2))


def verify_inputs(folder):
    inputs = json.loads((folder / "inputs.json").read_text())
    if inputs.get("schema") != "umr_elf3.official_input/1" or inputs.get("upstream_commit") != OFFICIAL_COMMIT:
        raise ValueError("Invalid official run inputs")
    verify_official(inputs["upstream"])
    verify_assets(Path(inputs["assets"]))
    for name, expected in inputs["protected_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Protected input changed: {name}")
    return inputs


def basic_audit(qpos, model, urdf, fps=50.):
    """Read-only checks. No clipping, smoothing, height correction or approval."""
    import mujoco
    if (qpos.ndim != 2 or qpos.shape[1] != 38 or len(qpos) < 2 or not np.isfinite(qpos).all()
            or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Invalid finite ELF3 qpos with wxyz root quaternion")
    names = [model.joint(i).name for i in range(1, model.njnt)]
    joints = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint") if j.get("type") != "fixed"}
    limits = np.asarray([[float(joints[n].find("limit").get(k)) for k in ("lower", "upper", "velocity")]
                         for n in names])
    angles = qpos[:, [model.joint(n).qposadr[0] for n in names]]
    violation = np.maximum(limits[:, 0] - angles, angles - limits[:, 1])
    speed = np.abs(np.diff(angles, axis=0)) * fps
    feet = []
    for g in range(model.ngeom):
        if model.geom_group[g] == 1 and model.body(model.geom_bodyid[g]).name in ("l_ankle_x_link", "r_ankle_x_link"):
            m = model.geom_dataid[g]
            start, count = model.mesh_vertadr[m], model.mesh_vertnum[m]
            feet.append((g, model.mesh_vert[start:start+count].copy()))
    if len(feet) != 2:
        raise ValueError("Expected the two actual ELF3 visual foot meshes")
    data = mujoco.MjData(model)
    foot_z, penetration, wrists = [], [], []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        foot_z.append(min(float((v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min())
                          for g, v in feet))
        penetration.append(max([0.] + [-float(c.dist) for c in data.contact
                                      if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]]))
        wrists.append([data.body(n).xpos.copy() for n in ("l_wrist_z_link", "r_wrist_z_link")])
    wrist_speed = np.linalg.norm(np.diff(wrists, axis=0), axis=2) * fps
    return {"frames": len(qpos), "fps": fps,
            "joint_limit_violation_frames": int(np.any(violation > 1e-6, axis=1).sum()),
            "joint_limit_max_excess_rad": float(max(0, violation.max())),
            "joint_speed_over_urdf_limit_intervals": int(np.any(speed > limits[:, 2] + 1e-6, axis=1).sum()),
            "joint_speed_max_by_name": dict(zip(names, speed.max(axis=0).tolist())),
            "foot_visual_mesh_min_z_m": float(min(foot_z)),
            "foot_below_minus_5mm_frames": int((np.asarray(foot_z) < -.005).sum()),
            "original_collision_self_penetration_max_m": float(max(penetration)),
            "original_collision_self_penetration_over_5mm_frames": int((np.asarray(penetration) > .005).sum()),
            "wrist_world_speed_p95_m_s": np.percentile(wrist_speed, 95, axis=0).tolist(),
            "wrist_world_speed_max_m_s": wrist_speed.max(axis=0).tolist(),
            "collision_scope": "original URDF enabled convex-hull contacts only; incomplete body coverage",
            "quality_status": "not_approved_requires_review", "training_approved": False,
            "physical_tracking_validated": False}


def collect_result(folder, inputs):
    import mujoco
    result = folder / "official_result.npz"
    model = mujoco.MjModel.from_xml_path(inputs["robot_xml"])
    # Upstream stores joint names as object dtype. Read numeric fields without
    # pickle; joint order is taken from the protected model, not an untrusted pickle.
    with np.load(result, allow_pickle=False) as z:
        qpos = z["qpos"].copy()
        ids = z["frame_ids"].copy()
        fps = float(z["fps"].item())
        if (fps != 50 or qpos.shape != (inputs["clock"]["frames"], model.nq)
                or not np.array_equal(ids, np.arange(len(qpos)))
                or str(z["robot_name"].item()) != "elf3_dof31"
                or Path(str(z["robot_xml"].item())).resolve() != Path(inputs["robot_xml"]).resolve()
                or Path(str(z["source_data"].item())).resolve() != folder / "source_50hz.npz"):
            raise ValueError("Official output robot/source/shape/clock mismatch")
    report = basic_audit(qpos, model, Path(inputs["assets"]) / "elf3.urdf")
    # Export only changes the container schema, not qpos values or dtype.
    with (folder / "motion.npz").open("xb") as stream:
        np.savez_compressed(stream, qpos=qpos, times=np.arange(len(qpos)) / fps, fps=np.asarray(fps),
                            dof_names=np.asarray(inputs["robot"]["joint_names"]),
                            root_body=np.asarray("torso_link"), quaternion_order=np.asarray("wxyz"))
    report.update(official_result_sha256=digest(result), motion_sha256=digest(folder / "motion.npz"),
                  upstream_commit=OFFICIAL_COMMIT, qpos_unchanged=True,
                  official_lqr_filter_enabled=True, legacy_refinement=False)
    write_json(folder / "audit.json", report)
    return report


def run(args):
    folder = Path(args.prepared).resolve()
    inputs = verify_inputs(folder)
    if not 60 <= args.timeout <= 7200:
        raise ValueError("Timeout must be 60..7200 seconds")
    if any((folder / n).exists() for n in ("run.log", "receipt.json", "official_result.npz")):
        raise FileExistsError("A run was already attempted; prepare a new output directory")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight failed; no silent CPU substitution")
    # Actually exercise the installed wheel on this GPU before expensive work.
    (torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")).sum().item()
    command = [sys.executable, "-u", str(Path(inputs["upstream"]) / "scripts/humanoid_retarget_pipeline.py"),
               "--config", str(folder / "official_config.json"), "--skip-view"]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", MUJOCO_GL="egl")
    started = time.monotonic()
    process = None
    receipt = {"schema": "umr_elf3.official_run/1", "backend": "official_hanyang9_UMR",
               "upstream_commit": OFFICIAL_COMMIT, "status": "failed", "training_approved": False,
               "command": command, "inputs_sha256": digest(folder / "inputs.json"),
               "environment": {p: importlib.metadata.version(p) for p in
                               ("torch", "numpy", "scipy", "mujoco", "smplx", "trimesh", "clarabel")}}
    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (folder / "run.log").open("x") as log:
            process = subprocess.Popen(command, cwd=inputs["upstream"], env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=args.timeout)
        receipt["returncode"] = code
        if code:
            raise RuntimeError(f"Official pipeline exited {code}; see {folder / 'run.log'}")
        verify_inputs(folder)
        report = collect_result(folder, inputs)
        receipt.update(status="kinematic_result_requires_review", frames=report["frames"],
                       motion_sha256=report["motion_sha256"])
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Kill only the process group created by this invocation, including any
        # descendants that outlived their pipeline parent. Never pkill by name.
        for s in previous:
            signal.signal(s, signal.SIG_IGN)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        receipt["elapsed_s"] = time.monotonic() - started
        write_json(folder / "receipt.json", receipt)
        for s, handler in previous.items():
            signal.signal(s, handler)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Generate config and complete 50 Hz source; no solving")
    for name in ("upstream", "source", "robot-xml", "assets", "body-model", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.set_defaults(fn=prepare)
    r = commands.add_parser("run", help="Run official learning and retargeting, then read-only audit")
    r.add_argument("--prepared", type=Path, required=True)
    r.add_argument("--timeout", type=int, default=1800)
    r.set_defaults(fn=run)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
