#!/usr/bin/env python3
"""Bounded, full-clip SMPL-X surface preparation; no retargeting or promotion.

The historical 30-second adapter stays byte-for-byte frozen. This module reuses
its geometry, material sampling and interpolation functions, but owns an explicit
full-sequence clock, chunked interpolation/inference and fail-closed budgets.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

import numpy as np

CORE_PATH = Path(__file__).with_name("umr_smplx_source.py")
CORE_SHA256 = "d2b41fe5cffaed03637d3e37cbdf9fef1f4c30996d636b6146483bf5e9b66b94"
if hashlib.sha256(CORE_PATH.read_bytes()).hexdigest() != CORE_SHA256:
    raise ValueError("Frozen historical SMPL-X adapter changed")
_spec = importlib.util.spec_from_file_location("umr_full_source_v5_frozen_core", CORE_PATH)
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)

FULL_SCHEMA = "bfm.smplx_full_sequence/5"
MIB = 1 << 20
GIB = 1 << 30
MOTION_FIELDS = ("root_orient", "pose_body", "pose_hand", "pose_jaw", "pose_eye")


@dataclass(frozen=True)
class Limits:
    """Per-origin limits, never a request to crop an origin to fit."""

    max_output_frames: int = 90001
    max_source_frames: int = 2000000
    max_array_bytes: int = 2 * GIB
    max_working_bytes: int = 4 * GIB
    max_disk_bytes: int = 3 * GIB
    max_raw_bytes: int = 512 * MIB
    min_free_disk_bytes: int = GIB

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("Every full-source budget must be a positive integer")
        if self.max_output_frames < 2 or self.max_source_frames < 3:
            raise ValueError("Frame budgets cannot hold a motion")


def unit_guard():
    """The caller owns a finite, process-group-cleaned CPU systemd service."""
    matches = re.findall(r"(?:^|/)(bfm-umr-(?:full-source|refresh)-[A-Za-z0-9_-]+\.service)(?:/|$)",
                         Path("/proc/self/cgroup").read_text(), re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError("Run preparation in an owned finite bfm-umr-*.service")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Full-source preparation is CPU-only: CUDA_VISIBLE_DEVICES must be empty")
    output = subprocess.check_output(["systemctl", "--user", "show", matches[0],
        "-p", "KillMode", "-p", "Restart", "-p", "RuntimeMaxUSec", "-p", "TimeoutStopUSec",
        "-p", "MemoryMax", "-p", "TasksMax"], text=True, timeout=10)
    props = dict(line.split("=", 1) for line in output.splitlines())
    if (props.get("KillMode") != "control-group" or props.get("Restart") != "no"
            or any(props.get(key) in (None, "", "0", "infinity")
                   for key in ("RuntimeMaxUSec", "TimeoutStopUSec", "MemoryMax", "TasksMax"))):
        raise RuntimeError("Finite runtime/stop/memory/tasks and entire-group cleanup required")
    return matches[0]


def full_sampling_grid(count: int, source_fps: float, fps=50., *, limits=None):
    """All supported 50 Hz timestamps from zero; residual tail is < one tick."""
    limits = limits or Limits()
    if (type(count) is not int or not 3 <= count <= limits.max_source_frames
            or isinstance(source_fps, bool) or not np.isfinite(source_fps)
            or not 0 < source_fps <= 1000 or isinstance(fps, bool) or fps != 50.):
        raise ValueError("Invalid full-source frame count or actual 50 Hz clock")
    duration = (count - 1) / float(source_fps)
    output_count = math.floor(duration * fps + 1e-9) + 1
    if not 2 <= output_count <= limits.max_output_frames:
        raise ValueError(f"Full source needs {output_count} frames; frame budget exceeded (no truncation)")
    times = np.arange(output_count, dtype=np.float64) / fps
    fractional = np.clip(times * source_fps, 0, count - 1)
    nearest = np.rint(fractional)
    fractional = np.where(np.abs(fractional - nearest) < 1e-9, nearest, fractional)
    lower = np.floor(fractional).astype(np.int64)
    upper = np.minimum(lower + 1, count - 1)
    return times, lower, upper, fractional - lower


def estimate_budget(frames: int, points: int, source_bytes=0, chunk_size=16, *, limits=None):
    """Conservative allocation/disk envelope, checked before mesh or arrays exist.

    The 1 GiB fixed inference allowance and 2 MiB per chunk-frame are conservative
    planning estimates, not an OS memory guarantee; the owning service also caps
    RSS. No full-sequence vertex mesh or full-sequence quaternion copy is made.
    """
    limits = limits or Limits()
    if (type(frames) is not int or not 2 <= frames <= limits.max_output_frames
            or type(points) is not int or not 512 <= points <= 8192
            or type(chunk_size) is not int or not 1 <= chunk_size <= 64
            or type(source_bytes) is not int or not 0 <= source_bytes <= limits.max_raw_bytes):
        raise ValueError("Invalid or over-budget full-source allocation parameters")
    # Moving surface float32, 55 joint positions/rotations float32, five 8-byte
    # clock/proof/foot arrays. Canonical float64 samples and integer identities.
    arrays = frames * (points * 24 + 55 * 12 * 4 + 5 * 8) + points * 96 + MIB
    disk = arrays + arrays // 100 + MIB  # compressed ZIP may be marginally larger
    working = arrays + source_bytes * 3 + GIB + chunk_size * 2 * MIB
    if arrays > limits.max_array_bytes:
        raise ValueError(f"Full surface arrays need {arrays} bytes; array budget exceeded (no truncation)")
    if disk > limits.max_disk_bytes:
        raise ValueError(f"Full surface archive needs up to {disk} bytes; disk budget exceeded")
    if working > limits.max_working_bytes:
        raise ValueError(f"Full source working estimate {working} bytes exceeds memory budget")
    return {"surface_array_estimate_bytes": arrays, "archive_upper_estimate_bytes": disk,
            "working_estimate_bytes": working, "source_loaded_bytes": source_bytes,
            "chunk_size": chunk_size, "limits": asdict(limits)}


def inspect_npz(path, *, max_uncompressed_bytes, selected=None):
    """Inspect NPY headers before allocation; never unpickle unrelated AMASS data."""
    layouts = {}
    total = 0
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError("Duplicate NPZ members are forbidden")
        for entry in entries:
            name = entry.filename.removesuffix(".npy")
            if selected is not None and name not in selected:
                continue
            if entry.filename != name + ".npy" or "/" in name or "\\" in name:
                raise ValueError("Unexpected NPZ member path")
            total += entry.file_size
            if total > max_uncompressed_bytes:
                raise ValueError("NPZ uncompressed bytes exceed allocation budget")
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported NPY header version")
                if dtype.hasobject or dtype.kind not in "biufUS":
                    raise ValueError("Numeric/plain-string NPZ arrays only; no pickle")
                nbytes = math.prod(shape) * dtype.itemsize
                if nbytes > max_uncompressed_bytes or stream.tell() + nbytes != entry.file_size:
                    raise ValueError("NPY shape/payload disagrees or exceeds allocation budget")
            layouts[name] = {"shape": shape, "dtype": dtype, "bytes": nbytes}
    return layouts, total


def interpolated_chunk(raw, lower, upper, alpha):
    """Reuse the pinned short-arc interpolation on only the needed source span."""
    first, last = int(lower.min()), int(upper.max()) + 1
    fields = {key: core.interpolate_rotvec(raw[key][first:last], lower - first, upper - first, alpha)
              for key in MOTION_FIELDS if key in raw}
    fields["trans"] = ((1 - alpha[:, None]) * raw["trans"][lower]
                       + alpha[:, None] * raw["trans"][upper])
    return fields


def inference_chunks(count, chunk_size):
    """Preserve the historical first-30s inference batch, then continue globally.

    Torch float32 GEMM may change last-frame roundoff when a batch contains 13
    versus 14 frames; surface normals amplify that roundoff. Keeping the first
    1501-frame partition identical preserves the old 30s prefix numerically.
    This is an inference allocation boundary, NOT a motion/floor/heading reset.
    """
    if type(count) is not int or count < 2 or type(chunk_size) is not int or not 1 <= chunk_size <= 64:
        raise ValueError("Invalid inference partition")
    boundaries = [0, min(count, 1501)]
    if count > 1501:
        boundaries.append(count)
    for begin, end in zip(boundaries, boundaries[1:]):
        for first in range(begin, end, chunk_size):
            yield first, min(first + chunk_size, end)


def sequence_metadata(source_count, source_fps, times, budget):
    source_duration = (source_count - 1) / source_fps
    return {"schema": FULL_SCHEMA, "source_frames": source_count,
            "source_duration": source_duration, "output_frames": len(times),
            "last_source_frame_index": source_count - 1,
            "unsampled_tail_seconds": float(source_duration - times[-1]),
            "full_sequence": True, "cropped": False, "clock_origin_seconds": 0.,
            "heading_policy": "one_global_first_frame_heading",
            "xy_policy": "one_global_first_pelvis_offset",
            "floor_policy": "one_global_full_sequence_foot_percentile_in_SmplxSurfaceHuman",
            "material_policy": "one_canonical_face_barycentric_identity_for_all_frames",
            "inference_partition": "legacy_1501_frame_prefix_then_full_tail_no_state_reset",
            "budget": budget, "frozen_core_sha256": CORE_SHA256}


def _atomic_archive(output, arrays, max_bytes):
    """Publish a completed file exclusively; remove only our private temp on error."""
    class BoundedFile:
        def __init__(self, stream): self.stream = stream
        def write(self, data):
            if self.stream.tell() + len(data) > max_bytes:
                raise ValueError("Actual NPZ write exceeds disk budget")
            return self.stream.write(data)
        def __getattr__(self, name): return getattr(self.stream, name)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".umr-full-v5-", suffix=".partial", dir=output.parent,
                                         delete=False) as stream:
            temp_path = Path(stream.name)
            np.savez_compressed(BoundedFile(stream), **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_path, output)  # refuses every pre-existing output, including symlinks
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def prepare_source(source: Path, body_model: Path, output: Path, *, fps=50., points=4096,
                   chunk_size=16, seed=0, limits=None) -> dict:
    """Prepare the entire clip or fail explicitly; never select a shorter window."""
    unit_guard()
    adapter_digest = core.sha256(Path(__file__))
    limits = limits or Limits()
    source, output = Path(source).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    keys = {*MOTION_FIELDS, "trans", "betas", "gender", "surface_model_type", "mocap_frame_rate"}
    _, source_bytes = inspect_npz(source, max_uncompressed_bytes=limits.max_raw_bytes, selected=keys)
    source_digest = core.sha256(source)
    raw = core.load_amass(source)
    times, lower, upper, alpha = full_sampling_grid(len(raw["trans"]), raw["fps"], fps, limits=limits)
    budget = estimate_budget(len(times), points, source_bytes, chunk_size, limits=limits)
    if type(seed) is not int or seed < 0:
        raise ValueError("Sampling seed must be a nonnegative integer")
    existing_parent = output.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < budget["archive_upper_estimate_bytes"] + limits.min_free_disk_bytes:
        raise ValueError("Insufficient free disk for full source plus reserved headroom")
    import smplx
    import torch
    from scipy.spatial.transform import Rotation
    model_path = core.model_file_for_gender(body_model, raw["gender"])
    model_digest = core.sha256(model_path)
    body = smplx.create(str(model_path), model_type="smplx", gender=raw["gender"], ext="npz",
                        use_pca=False, flat_hand_mean=True, num_betas=len(raw["betas"]), batch_size=1)
    if body.num_betas != len(raw["betas"]):
        raise ValueError("Model does not support every source beta; refusing silent truncation")
    parents = body.parents.detach().cpu().numpy()
    if len(parents) != 55:
        raise ValueError("Expected 55 SMPL-X kinematic joints")
    faces = np.asarray(body.faces, dtype=np.int64)
    skinning = body.lbs_weights.detach().cpu().numpy()
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32)

    def infer(fields):
        count = len(fields["root_orient"])
        hand = fields.get("pose_hand", np.zeros((count, 90)))
        eye = fields.get("pose_eye", np.zeros((count, 6)))
        return body(betas=tensor(raw["betas"])[None].expand(count, -1),
                    global_orient=tensor(fields["root_orient"]), body_pose=tensor(fields["pose_body"]),
                    transl=tensor(fields["trans"]), left_hand_pose=tensor(hand[:, :45]),
                    right_hand_pose=tensor(hand[:, 45:]),
                    jaw_pose=tensor(fields.get("pose_jaw", np.zeros((count, 3)))),
                    leye_pose=tensor(eye[:, :3]), reye_pose=tensor(eye[:, 3:]),
                    expression=torch.zeros((count, body.num_expression_coeffs)),
                    return_verts=True, return_full_pose=True)

    with torch.inference_mode():
        rest = infer({"root_orient": np.zeros((1, 3)), "pose_body": np.zeros((1, 63)),
                      "trans": np.zeros((1, 3))})
    rest_vertices = rest.vertices[0].detach().cpu().numpy() @ core.CANONICAL_ROTATION.T
    rest_joints = rest.joints[0, :55].detach().cpu().numpy() @ core.CANONICAL_ROTATION.T
    rest_rot = core.CANONICAL_ROTATION @ core.global_joint_rotations(rest.full_pose.detach().cpu().numpy(), parents)[0]
    canonical_offset = np.array([-rest_joints[0, 0], -rest_joints[0, 1], -rest_vertices[:, 2].min()])
    rest_vertices += canonical_offset
    rest_joints += canonical_offset
    actor_height = float(np.ptp(rest_vertices[:, 2]))
    if not .8 < actor_height < 2.5:
        raise ValueError("Unexpected canonical human height; check model/units")
    sample = core.sample_canonical_surface(rest_vertices, faces, skinning, points, seed)
    first_fields = interpolated_chunk(raw, lower[:1], upper[:1], alpha[:1])
    forward = Rotation.from_rotvec(first_fields["root_orient"][0]).apply([0., 0., 1.])
    if np.linalg.norm(forward[:2]) < .1:
        raise ValueError("Cannot infer full clip horizontal heading from first pelvis; no interval substitution")
    heading = Rotation.from_euler("z", -np.arctan2(forward[1], forward[0])).as_matrix()
    sequence_points = np.empty((len(times), points, 3), dtype=np.float32)
    sequence_normals = np.empty_like(sequence_points)
    joint_positions = np.empty((len(times), 55, 3), dtype=np.float32)
    joint_rotations = np.empty((len(times), 55, 3, 3), dtype=np.float32)
    foot_low = np.empty(len(times), dtype=np.float64)
    foot_vertex = np.isin(np.argmax(skinning, axis=1), [7, 8, 10, 11])
    if not foot_vertex.any():
        raise ValueError("No SMPL-X foot vertices found")
    for first, last in inference_chunks(len(times), chunk_size):
        fields = interpolated_chunk(raw, lower[first:last], upper[first:last], alpha[first:last])
        with torch.inference_mode():
            posed = infer(fields)
        vertices = posed.vertices.detach().cpu().numpy() @ heading.T
        pos, nrm = core.barycentric_transport(vertices, faces, sample["face_indices"], sample["barycentric"])
        sequence_points[first:last], sequence_normals[first:last] = pos, nrm
        joint_positions[first:last] = posed.joints[:, :55].detach().cpu().numpy() @ heading.T
        joint_rotations[first:last] = heading @ core.global_joint_rotations(posed.full_pose.detach().cpu().numpy(), parents)
        foot_low[first:last] = vertices[:, foot_vertex, 2].min(axis=1)
    xy_offset = -joint_positions[0, 0, :2].astype(np.float64)
    sequence_points[:, :, :2] += xy_offset
    joint_positions[:, :, :2] += xy_offset
    duration = (len(raw["trans"]) - 1) / raw["fps"]
    metadata = {"schema": core.SCHEMA, "source_file": str(source), "source_sha256": source_digest,
                "body_model": str(model_path), "body_model_sha256": model_digest,
                "source_gender": raw["gender"], "shape_policy": "source_betas_all_static",
                "num_betas": len(raw["betas"]), "flat_hand_mean": True, "expression": "zero",
                "optional_pose_fields_used": [key for key in MOTION_FIELDS[2:] if key in raw],
                "optional_pose_fields_missing_zeroed": [key for key in MOTION_FIELDS[2:] if key not in raw],
                "source_fps": raw["fps"], "target_fps": fps, "start": 0., "requested_duration": duration,
                "actual_duration": float(times[-1]), "frames": len(times), "points": points,
                "canonical_rotation": core.CANONICAL_ROTATION.tolist(), "posed_world_assumption": "AMASS Z-up metres",
                "posed_heading_rotation": heading.tolist(), "posed_xy_offset": xy_offset.tolist(),
                "actor_height": actor_height, "sampling_seed": seed,
                "surface_transport": "same_face_barycentric_on_posed_smplx_mesh",
                "normal_transport": "oriented_posed_triangle_cross_product",
                "license_note": "Local derived licensed AMASS/SMPL-X data; not approved for redistribution",
                "adapter_sha256": adapter_digest,
                "environment": core.environment_packages(("numpy", "scipy", "torch", "smplx")),
                "preprocessing_device": "cpu",
                "numeric_threads": {"torch_num_threads": torch.get_num_threads(),
                                    "torch_num_interop_threads": torch.get_num_interop_threads(),
                                    "environment": {key: os.environ.get(key) for key in
                                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
                "full_sequence": sequence_metadata(len(raw["trans"]), raw["fps"], times, budget)}
    arrays = dict(metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False), **sample,
                  sequence_points=sequence_points, sequence_normals=sequence_normals,
                  canonical_joint_positions=rest_joints, canonical_joint_rotations=rest_rot,
                  joint_positions=joint_positions, joint_rotations=joint_rotations, foot_low=foot_low,
                  times=times, sample_lower=lower, sample_upper=upper, sample_alpha=alpha,
                  segment_names=np.array(core.SEGMENTS), parents=parents, betas=raw["betas"],
                  actor_height=actor_height, fps=fps)
    validate_values({key: np.asarray(value) for key, value in arrays.items()}, limits=limits)
    if (core.sha256(source) != source_digest or core.sha256(model_path) != model_digest
            or core.sha256(CORE_PATH) != CORE_SHA256 or core.sha256(Path(__file__)) != adapter_digest):
        raise ValueError("Source/model/core changed during full preprocessing")
    _atomic_archive(output, arrays, limits.max_disk_bytes)
    return metadata


def validate_values(values, *, limits=None):
    """Old numeric/geometry checks plus exact full-clock, shape and origin checks."""
    limits = limits or Limits()
    metadata = json.loads(core.scalar_text(values["metadata_json"], "metadata_json"))
    if metadata.get("schema") != core.SCHEMA or metadata.get("full_sequence", {}).get("schema") != FULL_SCHEMA:
        raise ValueError("Expected explicit full-sequence v5 metadata")
    count, points = metadata.get("frames"), metadata.get("points")
    estimate_budget(count, points, limits=limits)
    shapes = {"canonical_points": (points, 3), "canonical_normals": (points, 3),
              "sequence_points": (count, points, 3), "sequence_normals": (count, points, 3),
              "canonical_joint_positions": (55, 3), "canonical_joint_rotations": (55, 3, 3),
              "joint_positions": (count, 55, 3), "joint_rotations": (count, 55, 3, 3),
              "face_indices": (points,), "barycentric": (points, 3), "segment": (points,),
              "binding_joint_ids": (points,), "times": (count,), "foot_low": (count,),
              "sample_lower": (count,), "sample_upper": (count,), "sample_alpha": (count,),
              "parents": (55,), "fps": (), "actor_height": ()}
    if set(values) != {*shapes, "betas", "segment_names", "metadata_json"}:
        raise ValueError("Prepared source has missing or unexpected fields")
    if sum(np.asarray(value).nbytes for value in values.values()) > limits.max_array_bytes:
        raise ValueError("Actual prepared arrays exceed allocation budget")
    for key, shape in shapes.items():
        value = np.asarray(values[key])
        if value.shape != shape or value.dtype.kind not in "iuf":
            raise ValueError(f"Invalid prepared shape/type: {key}")
        for first in range(0, max(1, len(value) if value.ndim else 1), 64):
            chunk = value[first:first + 64] if value.ndim else value
            if not np.isfinite(chunk).all():
                raise ValueError(f"Invalid nonfinite prepared field: {key}")
    for key in ("sequence_points", "sequence_normals", "joint_positions", "joint_rotations"):
        if values[key].dtype != np.dtype("float32"):
            raise ValueError(f"Prepared moving arrays must retain frozen float32 dtype: {key}")
    full = metadata["full_sequence"]
    if (full.get("full_sequence") is not True or full.get("cropped") is not False
            or full.get("clock_origin_seconds") != 0. or metadata.get("start") != 0.
            or full.get("output_frames") != count or float(values["fps"]) != 50.
            or metadata.get("target_fps") != 50. or full.get("frozen_core_sha256") != CORE_SHA256):
        raise ValueError("Full sequence identity/clock contract disagrees")
    expected = full_sampling_grid(full.get("source_frames"), metadata.get("source_fps"), limits=limits)
    for key, expected_array in zip(("times", "sample_lower", "sample_upper", "sample_alpha"), expected):
        if not np.array_equal(values[key], expected_array):
            raise ValueError(f"Incomplete or relabelled full-sequence sampling proof: {key}")
    duration = (full["source_frames"] - 1) / metadata["source_fps"]
    tail = duration - float(values["times"][-1])
    if (not -1e-10 <= tail < .02 + 1e-10
            or full.get("source_duration") != duration or metadata.get("requested_duration") != duration
            or metadata.get("actual_duration") != float(values["times"][-1])
            or full.get("unsampled_tail_seconds") != tail
            or full.get("last_source_frame_index") != full["source_frames"] - 1):
        raise ValueError("Full-sequence final boundary disagrees; truncation forbidden")
    for key in ("canonical_normals", "sequence_normals"):
        for first in range(0, len(values[key]), 64):
            if not np.allclose(np.linalg.norm(values[key][first:first + 64], axis=-1), 1., atol=1e-5):
                raise ValueError("Prepared surface normals must be unit length")
    for key, maximum in (("binding_joint_ids", 55), ("segment", len(core.SEGMENTS))):
        value = values[key]
        if value.dtype.kind not in "iu" or np.any(value < 0) or np.any(value >= maximum):
            raise ValueError(f"Invalid prepared {key}")
    for key in ("face_indices", "sample_lower", "sample_upper"):
        if values[key].dtype.kind not in "iu" or np.any(values[key] < 0):
            raise ValueError(f"Invalid integer indices: {key}")
    barycentric = values["barycentric"]
    if (np.any(barycentric < -1e-9) or np.any(barycentric > 1 + 1e-9)
            or not np.allclose(barycentric.sum(axis=1), 1., atol=1e-7, rtol=0)):
        raise ValueError("Invalid prepared barycentric weights")
    for key in ("canonical_joint_rotations", "joint_rotations"):
        for first in range(0, len(values[key]), 64):
            rotation = values[key][first:first + 64].astype(np.float64)
            if (not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-5, rtol=0)
                    or not np.allclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0)):
                raise ValueError(f"Prepared {key} must be proper rotation matrices")
    parents = values["parents"]
    if (parents.dtype.kind not in "iu" or parents[0] != -1
            or any(not 0 <= int(parents[index]) < index for index in range(1, 55))):
        raise ValueError("Invalid prepared SMPL-X hierarchy")
    if not .8 < float(values["actor_height"]) < 2.5 or metadata.get("actor_height") != float(values["actor_height"]):
        raise ValueError("Invalid prepared actor height")
    if list(values["segment_names"]) != core.SEGMENTS:
        raise ValueError("Unknown segment conventions")
    betas = values["betas"]
    if (type(metadata.get("num_betas")) is not int or betas.shape != (metadata["num_betas"],)
            or not 1 <= len(betas) <= 300 or betas.dtype.kind not in "fiu" or not np.isfinite(betas).all()
            or metadata.get("shape_policy") != "source_betas_all_static"
            or metadata.get("flat_hand_mean") is not True or metadata.get("expression") != "zero"
            or metadata.get("source_gender") not in ("male", "female", "neutral")
            or metadata.get("canonical_rotation") != core.CANONICAL_ROTATION.tolist()):
        raise ValueError("Invalid full-source canonical shape identity")
    heading = np.asarray(metadata.get("posed_heading_rotation"), dtype=float)
    xy = np.asarray(metadata.get("posed_xy_offset"), dtype=float)
    if (heading.shape != (3, 3) or xy.shape != (2,) or not np.isfinite(xy).all()
            or not np.allclose(heading.T @ heading, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(heading), 1., atol=1e-6, rtol=0)
            or not np.allclose(heading[2], [0., 0., 1.], atol=1e-6, rtol=0)
            or not np.allclose(values["joint_positions"][0, 0, :2], 0., atol=1e-6, rtol=0)):
        raise ValueError("Invalid one-global-heading/XY normalization")
    return metadata


def load_prepared_source(path: Path, *, limits=None) -> dict:
    limits = limits or Limits()
    path = Path(path)
    if path.stat().st_size > limits.max_disk_bytes:
        raise ValueError("Prepared source archive exceeds disk budget")
    inspect_npz(path, max_uncompressed_bytes=limits.max_array_bytes)
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    values["metadata"] = validate_values(values, limits=limits)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--body-model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    for name, value in asdict(Limits()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=value)
    args = parser.parse_args()
    limits = Limits(**{key: getattr(args, key) for key in asdict(Limits())})
    unit_guard()
    if args.command == "prepare":
        if args.body_model is None or args.output is None:
            parser.error("prepare requires --body-model and --output")
        result = prepare_source(args.source, args.body_model, args.output, points=args.points,
                                chunk_size=args.chunk_size, seed=args.seed, limits=limits)
    else:
        result = load_prepared_source(args.source, limits=limits)["metadata"]
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
