#!/usr/bin/env python3
"""Opt-in adapter for a *pinned, unofficial* UMR implementation.

This module neither installs nor imports UMR into the ScaleBFM environment.
``retarget-bvh`` runs Stage II in an explicitly selected Python with an existing,
matching Stage I setup. ``convert`` validates its artifact and emits the native
ScaleRetarget kinematic pickle. Neither command promotes data or trains a policy.
AMASS/SMPL-X is deliberately unsupported: it requires a real source adapter.

Examples (output directories must not exist):
  python scripts/umr_backend.py convert --umr-repo /path/to/pinned/umr \
    --input /path/to/motion.npz --source /path/to/walk.bvh \
    --origin-id xsens/actor/walk --split validation --output-dir /new/output
  python scripts/umr_backend.py retarget-bvh --python /path/to/umr/python \
    --umr-repo /path/to/pinned/umr --setup-dir /path/to/setup \
    --source /path/to/walk.bvh --origin-id xsens/actor/walk \
    --split validation --duration 5 --output-dir /new/output

Pickles and UMR setup files are trusted local artifacts, not a safe interchange
format for untrusted uploads. Pickle conversion requires --trusted-pickle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


UMR_COMMIT = "0aa1855fe4f65a73681ffbd1d9f95ab1c2bad9ca"
UMR_REMOTE = "https://github.com/longchengzhuo/Unified-Motion-Retargeting"
UMR_ROBOT_XML = "assets/robots/g1_description/xml/g1_29dof_rev_1_0.xml"
UMR_CONFIG = "configs/g1_29dof_rev_1_0.yaml"
ADAPTER_VERSION = "umr-bvh-kinematic-v1"
G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _names(values: Sequence[str], label: str) -> list[str]:
    names = list(values)
    if len(names) != 29 or any(not isinstance(n, str) or not n for n in names):
        raise ValueError(f"{label} must contain 29 nonempty joint names")
    if len(set(names)) != 29 or set(names) != set(G1_JOINT_NAMES):
        raise ValueError(f"{label} must match the exact, unique G1 29-DoF joint name set")
    return names


def convert_umr_payload(
    payload: Mapping[str, Any], *, source_joint_names: Sequence[str] | None = None,
    target_joint_names: Sequence[str] = G1_JOINT_NAMES,
) -> dict[str, Any]:
    """Validate and copy UMR fields; no grounding, rotation or resampling.

    qpos is [xyz, quaternion wxyz, named joints]. Native output quaternion is
    xyzw. Small floating-point norm drift (<= 1e-4) is normalized; malformed
    quaternions are rejected. If redundant fields exist they must agree.
    """
    import numpy as np

    target = _names(target_joint_names, "target_joint_names")
    declared = payload.get("dof_names")
    if source_joint_names is None and declared is None:
        raise ValueError("Missing joint names; never infer qpos order from its shape")
    names = _names(source_joint_names if source_joint_names is not None else declared,
                   "source_joint_names")
    if declared is not None and _names(declared, "dof_names") != names:
        raise ValueError("Declared dof_names disagree with source_joint_names")
    try:
        fps_array = np.asarray(payload["fps"])
        if fps_array.ndim != 0 or fps_array.dtype.kind not in "iuf":
            raise ValueError("fps must be a numeric scalar")
        fps = float(fps_array)
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Missing or invalid fps") from error
    if not math.isfinite(fps) or not 0 < fps <= 1000:
        raise ValueError("fps must be finite and in (0, 1000]")

    def array(key: str) -> Any:
        try:
            raw = np.asarray(payload[key])
            if raw.dtype.kind not in "iuf":
                raise ValueError(f"{key} must be a real numeric array")
            value = raw.astype(np.float64, copy=True)
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError(f"Missing or invalid {key}") from error
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values")
        return value

    if "qpos" in payload:
        qpos = array("qpos")
        if qpos.ndim != 2 or qpos.shape[0] < 2 or qpos.shape[1] != 36:
            raise ValueError("qpos must have shape (T >= 2, 36)")
        root_pos, root_rot, dof = qpos[:, :3].copy(), qpos[:, [4, 5, 6, 3]], qpos[:, 7:].copy()
    else:
        root_pos, root_rot = array("root_trans"), array("root_rot")
        dof = array("dof")
    if root_pos.ndim != 2 or root_pos.shape[0] < 2 or root_pos.shape[1] != 3:
        raise ValueError("root_trans must have shape (T >= 2, 3)")
    frames = len(root_pos)
    if root_rot.shape != (frames, 4) or dof.shape != (frames, 29):
        raise ValueError("root_rot/dof shape or frame counts do not match")
    for key, expected in (("root_trans", root_pos), ("root_rot", root_rot),
                          ("dof", dof), ("dof_full", dof)):
        if key in payload:
            value = array(key)
            if value.shape != expected.shape or not np.allclose(value, expected, rtol=0, atol=1e-7):
                raise ValueError(f"Redundant {key} disagrees with qpos/motion fields")
    norms = np.linalg.norm(root_rot, axis=1)
    if not np.allclose(norms, 1, rtol=0, atol=1e-4):
        raise ValueError("Root quaternions must be normalized (tolerance 1e-4)")
    root_rot /= norms[:, None]
    order = [names.index(name) for name in target]
    return {"fps": fps, "root_pos": root_pos.copy(), "root_rot": root_rot.copy(),
            "dof_pos": dof[:, order].copy()}


def verify_repository(repo: Path) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("--umr-repo must be the third-party repository root")
    commit = git("rev-parse", "HEAD")
    if commit != UMR_COMMIT:
        raise ValueError(f"UMR pin mismatch: expected {UMR_COMMIT}, got {commit}")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("UMR repository is dirty; preserve it and use a clean pinned checkout")
    return {"path": str(repo), "commit": commit, "remote": git("remote", "get-url", "origin"),
            "upstream": UMR_REMOTE, "implementation": "unofficial independent reproduction",
            "config_sha256": sha256(repo / UMR_CONFIG),
            "robot_xml_sha256": sha256(repo / UMR_ROBOT_XML)}


def xml_joint_names(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None or root.findall(".//include"):
        raise ValueError("Expected the pinned, self-contained robot XML")
    joints = [joint for joint in worldbody.iter("joint") if joint.get("type", "hinge") != "free"]
    if any(joint.get("type", "hinge") != "hinge" for joint in joints):
        raise ValueError("Only 29 named hinge joints are supported")
    return _names([joint.get("name", "") for joint in joints], "robot XML joints")


def load_artifact(path: Path, repo: Path, *, trusted_pickle: bool = False) -> dict[str, Any]:
    """Read only safe NPZ members; pickles require an explicit trust opt-in."""
    import numpy as np
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as source:
            keys = ("qpos", "fps", "dof_names", "robot_xml", "bvh_path", "start", "duration")
            result = {key: source[key].copy() for key in keys if key in source}
        if "dof_names" not in result:
            expected_xml = (repo / UMR_ROBOT_XML).resolve(strict=True)
            if "robot_xml" not in result or Path(str(result["robot_xml"])).resolve() != expected_xml:
                raise ValueError("NPZ without dof_names must identify the exact pinned robot XML")
            result["dof_names"] = xml_joint_names(expected_xml)
        return result
    if path.suffix.lower() == ".pkl":
        if not trusted_pickle:
            raise ValueError("Pickle can execute code: use --trusted-pickle only for your own artifacts")
        with path.open("rb") as stream:
            result = pickle.load(stream)
        if not isinstance(result, dict):
            raise ValueError("UMR pickle must contain a dictionary")
        return result
    raise ValueError("Input must be an UMR .npz or trusted .pkl, not AMASS data")


def _save_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save_motion(path: Path, motion: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            pickle.dump(motion, stream, protocol=4)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publication without replacing any existing artifact.
        os.link(temporary, path)
    finally:
        if created and temporary.exists():
            temporary.unlink()


def run_bounded(command: list[str], *, cwd: Path, timeout: float) -> None:
    """Own a process group and reap it on timeout/interruption; never daemonize."""
    environment = os.environ.copy()
    # Select the explicit venv, not imports injected by the calling Isaac/Conda
    # shell. Retain normal device/proxy/thread configuration without printing it.
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True, env=environment)
    try:
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        # Includes descendants if the direct child already exited. The new
        # session/group belongs to this invocation, never another user's job.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _worker(args: argparse.Namespace) -> None:
    """Runs only in the selected UMR interpreter; validates setup before IK."""
    import importlib.metadata
    import numpy as np
    repo = args.umr_repo.resolve(strict=True)
    verify_repository(repo)
    versions = {}
    for package in ("numpy", "scipy", "mujoco", "mink", "torch", "qpsolvers", "clarabel"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    _save_json(args.output_dir / "worker_environment.json",
               {"executable": sys.executable, "python": sys.version, "packages": versions})
    sys.path.insert(0, str(repo))
    from umr.bodies import bvh as bvh_module
    from umr.bodies.skeletons import get_skeleton
    from umr.config import load_config
    from umr.paths import SetupLayout, ClipLayout
    from umr.stages import _digest, _fingerprint, retarget_motion

    cfg = load_config("unitree_g1")
    skeleton = get_skeleton(args.human)
    setup = SetupLayout(args.setup_dir.resolve(strict=True))
    expected_bodies_stamp = _fingerprint(
        "bodies/2", _digest(repo / UMR_ROBOT_XML), skeleton.name,
        bvh_module.skeleton_signature(args.source, strip_prefix=skeleton.strip_prefix),
        cfg.robot, cfg.source, cfg.sampling, int(cfg.sampling["n_points"]),
    )
    with np.load(setup.bodies, allow_pickle=False) as bodies:
        if str(bodies["stamp"]) != expected_bodies_stamp:
            raise ValueError("Setup does not match this BVH skeleton, actor dimensions or pinned config")
        if (Path(str(bodies["human_xml"])).resolve() != setup.human_xml or
                Path(str(bodies["robot_xml"])).resolve() != (repo / UMR_ROBOT_XML).resolve()):
            raise ValueError("Setup XML paths do not match the selected setup and pinned robot")
    expected_corr_stamp = _fingerprint("correspondence/2", expected_bodies_stamp,
                                      cfg.correspondence, int(cfg.correspondence["epochs"]))
    with np.load(setup.correspondence, allow_pickle=False) as correspondence:
        if str(correspondence["stamp"]) != expected_corr_stamp:
            raise ValueError("Stage I correspondence does not match the pinned setup/config")
    clip = ClipLayout(args.output_dir / "umr")
    if clip.motion.exists() or clip.motion_pkl.exists():
        raise FileExistsError("Worker refuses to overwrite an existing UMR artifact")
    retarget_motion(cfg, args.source, setup, clip, skeleton=skeleton, tgt_fps=args.fps,
                    interpolation="slerp", start=args.start, duration=args.duration,
                    export_pkl=True)


def _check_origin(origin_id: str) -> None:
    if not origin_id.strip() or any(ord(char) < 32 for char in origin_id):
        raise ValueError("origin-id must be a nonempty stable source motion identity")
    if origin_id.startswith("/") or ".." in origin_id.split("/"):
        raise ValueError("origin-id must be a relative source identity, not a local path")


def execute(args: argparse.Namespace) -> Path:
    _check_origin(args.origin_id)
    repo = args.umr_repo.resolve(strict=True)
    source = args.source.resolve(strict=True)
    if source.suffix.lower() != ".bvh":
        raise ValueError("Only BVH is supported; AMASS/SMPL-X needs a separate source adapter")
    repository = verify_repository(repo)
    if args.command == "retarget-bvh":
        if not (math.isfinite(args.start) and args.start >= 0 and
                math.isfinite(args.duration) and args.duration > 0 and
                math.isfinite(args.fps) and 0 < args.fps <= 1000 and
                math.isfinite(args.timeout) and 0 < args.timeout <= 86400):
            raise ValueError("Require finite start >= 0, duration > 0, 0 < fps <= 1000 and 0 < timeout <= 86400")
        # Resolving a venv's python symlink to /usr/bin/python would silently
        # select the wrong environment. Preserve the executable's venv path.
        args.python = args.python.absolute()
        if not args.python.is_file() or not os.access(args.python, os.X_OK):
            raise ValueError("--python must be an existing executable in the isolated UMR environment")
        args.setup_dir = args.setup_dir.resolve(strict=True)
        protected = [source, repo / UMR_CONFIG, repo / UMR_ROBOT_XML,
                     *[args.setup_dir / name for name in ("human.xml", "bodies.npz", "correspondence.npz")]]
    else:
        args.input = args.input.resolve(strict=True)
        protected = [source, args.input, repo / UMR_CONFIG, repo / UMR_ROBOT_XML]
    before = {str(path): sha256(path) for path in protected}
    output = args.output_dir.absolute()
    # mkdir(exist_ok=False) also refuses empty directories and symlink targets.
    output.mkdir(parents=True, exist_ok=False)
    args.output_dir, args.source, args.umr_repo = output, source, repo
    receipt: dict[str, Any] = {
        "schema_version": 1, "adapter": ADAPTER_VERSION, "adapter_sha256": sha256(Path(__file__)),
        "status": "RUNNING", "started_unix": time.time(), "command": args.command,
        "repository": repository, "origin_id": args.origin_id, "split": args.split,
        "source": str(source), "source_sha256": before[str(source)],
        "artifact_origin_verification": (
            "generated_by_this_invocation" if args.command == "retarget-bvh" else
            "declared_metadata_only; current pin/source hash does not attest historical generation"),
        "split_provenance": "caller supplied; must match the canonical origin-based split manifest",
        "quality_accepted": False, "training_started": False, "data_promoted": False,
        "protected_inputs_sha256": before,
        "limitations": ["BVH only; not an AMASS/SMPL-X surface adapter",
                        "Contract validation is not physics or tracking acceptance",
                        "No ground-height correction, resampling or world-frame rotation applied",
                        "Unofficial MIT code; robot assets and motion data have separate terms"],
    }
    receipt_path = output / "provenance.json"
    _save_json(receipt_path, receipt)
    try:
        if args.command == "retarget-bvh":
            command = [str(args.python), str(Path(__file__).resolve()), "_worker",
                       "--umr-repo", str(repo), "--source", str(source),
                       "--setup-dir", str(args.setup_dir), "--output-dir", str(output),
                       "--human", args.human, "--start", str(args.start),
                       "--duration", str(args.duration), "--fps", str(args.fps)]
            receipt["worker"] = {"argv": command, "timeout_seconds": args.timeout}
            _save_json(receipt_path, receipt)
            run_bounded(command, cwd=repo, timeout=args.timeout)
            receipt["worker"]["environment"] = json.loads(
                (output / "worker_environment.json").read_text(encoding="utf-8"))
            artifact = output / "umr" / "motion.npz"
        else:
            artifact = args.input
        payload = load_artifact(artifact, repo, trusted_pickle=getattr(args, "trusted_pickle", False))
        declared_source = payload.get("bvh_path", payload.get("source_file"))
        if declared_source is not None and Path(str(declared_source)).resolve() != source:
            raise ValueError("Artifact source path does not match --source")
        motion = convert_umr_payload(payload)
        if {str(path): sha256(path) for path in protected} != before:
            raise RuntimeError("Protected source/setup/config changed; reject this artifact")
        verify_repository(repo)
        destination = output / "motion.pkl"
        _save_motion(destination, motion)
        receipt.update({"status": "CONVERTED_NOT_QUALITY_ACCEPTED",
                        "input_artifact": str(artifact), "input_artifact_sha256": sha256(artifact),
                        "output": str(destination), "output_sha256": sha256(destination),
                        "frames": len(motion["root_pos"]), "fps": motion["fps"],
                        "joint_names": list(G1_JOINT_NAMES), "root_quaternion_order": "xyzw",
                        "world_convention": "UMR world preserved, meters, Z up",
                        "quaternion_normalization_tolerance": 1e-4,
                        "joint_limit_validation": "not performed; required in downstream quality gate",
                        "clip_start_seconds": float(payload["start"]) if "start" in payload else None,
                        "clip_requested_duration_seconds": float(payload["duration"]) if "duration" in payload else None,
                        "source_joint_names": list(payload["dof_names"]),
                        "protected_inputs_unchanged": True})
        return destination
    except BaseException as error:
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["elapsed_seconds"] = time.time() - receipt["started_unix"]
        _save_json(receipt_path, receipt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("convert", "retarget-bvh", "_worker"):
        command = commands.add_parser(name)
        command.add_argument("--umr-repo", required=True, type=Path)
        command.add_argument("--source", required=True, type=Path)
        command.add_argument("--output-dir", required=True, type=Path)
        if name != "_worker":
            command.add_argument("--origin-id", required=True)
            command.add_argument("--split", required=True, choices=("train", "validation", "test"))
        if name == "convert":
            command.add_argument("--input", required=True, type=Path)
            command.add_argument("--trusted-pickle", action="store_true")
        else:
            command.add_argument("--setup-dir", required=True, type=Path)
            command.add_argument("--human", choices=("xsens", "fzmotion"), default="xsens")
            command.add_argument("--fps", type=float, default=50.0)
            command.add_argument("--start", type=float, default=0.0)
            command.add_argument("--duration", required=True, type=float)
        if name == "retarget-bvh":
            command.add_argument("--python", required=True, type=Path)
            command.add_argument("--timeout", type=float, default=1200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "_worker":
        _worker(args)
    else:
        print(execute(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
