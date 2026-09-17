#!/usr/bin/env python3
"""Full SMPL-X source with an audited, t0-only anatomical heading fallback.

This is an explicit controlled fork of v5's preparation orchestration. Frozen
v5/core and their consumers are untouched. No later-frame search or truncation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import umr_full_source_v5 as v5

core = v5.core
V5_SHA = "5a8a788ba9d42efb4db257e821554c9ba11a1363fdc4972d1792057e1aff7f23"
SCHEMA = "bfm.smplx_t0_heading_reference/6"
PROTOCOL = ROOT / "docs/UMR_HEADING_RECOVERY_V6_20260912.md"
THRESHOLD = .1
if core.sha256(Path(v5.__file__)) != V5_SHA:
    raise ValueError("Frozen full-source v5 changed")


def heading_reference(root_rotvec):
    """SMPL-X +Z=anterior, +X=left; recover a yaw from left x world-up if needed.

    When ||forward_xy|| < .1, orthogonality guarantees
    ||left_xy|| >= sqrt(1-.1**2). This branch needs no other time sample.
    The branch is selected ONCE, not per output frame.
    """
    from scipy.spatial.transform import Rotation
    value = np.asarray(root_rotvec, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Heading needs one finite t0 pelvis axis-angle")
    rotation = Rotation.from_rotvec(value)
    forward = rotation.apply([0., 0., 1.])
    left = rotation.apply([1., 0., 0.])
    norm = float(np.linalg.norm(forward[:2]))
    left_norm = float(np.linalg.norm(left[:2]))
    fallback = norm < THRESHOLD
    if fallback:
        if left_norm < np.sqrt(1. - THRESHOLD**2) - 1e-12:
            raise ValueError("Pelvis axes violate the orthogonal fallback bound")
        reference = np.array([left[1], -left[0], 0.])
        branch = "t0_left_cross_world_up"
    else:
        reference = forward
        branch = "legacy_t0_forward_projection"
    # Identical expression/order to frozen v5 on its accepted branch.
    heading = Rotation.from_euler("z", -np.arctan2(reference[1], reference[0])).as_matrix()
    proof = {"schema": SCHEMA, "branch": branch, "fallback_used": fallback,
             "threshold_forward_xy": THRESHOLD, "first_sample_root_rotvec": value.tolist(),
             "first_forward_world": forward.tolist(), "first_left_world": left.tolist(),
             "first_forward_xy_norm": norm, "first_left_xy_norm": left_norm,
             "reference_world_xy": reference[:2].tolist(), "world_up": [0., 0., 1.],
             "selection_time_seconds": 0., "selection_frame_index": 0,
             "per_frame_switching": False, "future_frames_inspected": 0,
             "reference_is_locomotion_direction": False,
             "frozen_v5_sha256": V5_SHA,
             "stage_ii_legacy_initializer_compatible": not fallback}
    return heading, proof


def model_left_axis_evidence(rest_joints, joint_names):
    if list(joint_names[:3]) != ["pelvis", "left_hip", "right_hip"]:
        raise ValueError("Installed SMPL-X joint-name convention differs")
    joints = np.asarray(rest_joints, dtype=float)
    if joints.shape != (55, 3) or not np.isfinite(joints).all():
        raise ValueError("Invalid model rest-joint evidence")
    lateral = joints[1] - joints[2]
    distance = float(np.linalg.norm(lateral))
    cosine = float(lateral[0] / distance) if distance > 0 else 0.
    if not distance > .05 or not cosine > .8:
        raise ValueError("Actual gender/shape model does not support +X anatomical-left convention")
    return {"joint_names_first_three": list(joint_names[:3]),
            "left_hip_minus_right_hip_rest_model_xyz_m": lateral.tolist(),
            "hip_separation_m": distance, "positive_x_alignment_cosine": cosine,
            "declared_anatomical_left_local_axis": [1., 0., 0.]}


def validate_values(values, *, limits=None):
    metadata = v5.validate_values(values, limits=limits)
    proof = metadata.get("heading_reference", {})
    if proof.get("schema") != SCHEMA:
        raise ValueError("Missing explicit v6 heading proof")
    heading, expected = heading_reference(proof.get("first_sample_root_rotvec"))
    if any(proof.get(key) != value for key, value in expected.items()):
        raise ValueError("V6 heading evidence disagrees with its actual t0 pose")
    if not np.array_equal(heading, np.asarray(metadata["posed_heading_rotation"])):
        raise ValueError("V6 heading proof does not match the global source rotation")
    from scipy.spatial.transform import Rotation
    first_rotation = heading @ Rotation.from_rotvec(proof["first_sample_root_rotvec"]).as_matrix()
    if not np.allclose(values["joint_rotations"][0, 0], first_rotation, atol=1e-6, rtol=0):
        raise ValueError("First stored pelvis orientation does not match the heading proof")
    evidence = metadata.get("model_axis_evidence", {})
    if (evidence.get("joint_names_first_three") != ["pelvis", "left_hip", "right_hip"]
            or evidence.get("declared_anatomical_left_local_axis") != [1., 0., 0.]):
        raise ValueError("Missing actual model anatomical-axis evidence")
    # The original model rest frame is recovered from unchanged canonical joints;
    # canonical translation cancels in the left-right difference.
    recovered = np.asarray(values["canonical_joint_positions"]) @ core.CANONICAL_ROTATION
    rebuilt = model_left_axis_evidence(recovered, evidence["joint_names_first_three"])
    for key in ("hip_separation_m", "positive_x_alignment_cosine", "left_hip_minus_right_hip_rest_model_xyz_m"):
        if not np.allclose(evidence.get(key), rebuilt[key], atol=1e-12, rtol=0):
            raise ValueError("Model anatomical-axis evidence differs from canonical geometry")
    return metadata


def prepare_source(source: Path, body_model: Path, output: Path, *, fps=50., points=4096,
                   chunk_size=16, seed=0, limits=None):
    v5.unit_guard()
    limits = limits or v5.Limits()
    source, output = Path(source).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    keys = {*v5.MOTION_FIELDS, "trans", "betas", "gender", "surface_model_type", "mocap_frame_rate"}
    _, source_bytes = v5.inspect_npz(source, max_uncompressed_bytes=limits.max_raw_bytes, selected=keys)
    frozen = {str(path): core.sha256(path) for path in
              (source, Path(__file__).resolve(), Path(v5.__file__).resolve(), v5.CORE_PATH, PROTOCOL)}
    raw = core.load_amass(source)
    times, lower, upper, alpha = v5.full_sampling_grid(len(raw["trans"]), raw["fps"], fps, limits=limits)
    budget = v5.estimate_budget(len(times), points, source_bytes, chunk_size, limits=limits)
    if type(seed) is not int or seed < 0:
        raise ValueError("Sampling seed must be a nonnegative integer")
    existing = output.parent
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < budget["archive_upper_estimate_bytes"] + limits.min_free_disk_bytes:
        raise ValueError("Insufficient full-source disk headroom")
    first = v5.interpolated_chunk(raw, lower[:1], upper[:1], alpha[:1])
    heading, heading_proof = heading_reference(first["root_orient"][0])
    import smplx
    import smplx.joint_names as names
    import smplx.body_models as body_models
    import torch
    for path in (Path(names.__file__).resolve(), Path(body_models.__file__).resolve()):
        frozen[str(path)] = core.sha256(path)
    model_path = core.model_file_for_gender(body_model, raw["gender"])
    frozen[str(model_path)] = core.sha256(model_path)
    body = smplx.create(str(model_path), model_type="smplx", gender=raw["gender"], ext="npz",
                        use_pca=False, flat_hand_mean=True, num_betas=len(raw["betas"]), batch_size=1)
    if body.num_betas != len(raw["betas"]):
        raise ValueError("Cannot truncate any actual source beta")
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
    model_joints = rest.joints[0, :55].detach().cpu().numpy()
    axis_evidence = model_left_axis_evidence(model_joints, names.JOINT_NAMES)
    axis_evidence.update(joint_names_file=str(Path(names.__file__).resolve()),
                         joint_names_sha256=frozen[str(Path(names.__file__).resolve())],
                         body_model_implementation=str(Path(body_models.__file__).resolve()),
                         body_model_implementation_sha256=frozen[str(Path(body_models.__file__).resolve())])
    rest_vertices = rest.vertices[0].detach().cpu().numpy() @ core.CANONICAL_ROTATION.T
    rest_joints = model_joints @ core.CANONICAL_ROTATION.T
    rest_rot = core.CANONICAL_ROTATION @ core.global_joint_rotations(rest.full_pose.detach().cpu().numpy(), parents)[0]
    offset = np.array([-rest_joints[0, 0], -rest_joints[0, 1], -rest_vertices[:, 2].min()])
    rest_vertices += offset
    rest_joints += offset
    actor_height = float(np.ptp(rest_vertices[:, 2]))
    if not .8 < actor_height < 2.5:
        raise ValueError("Unexpected canonical human height")
    sample = core.sample_canonical_surface(rest_vertices, faces, skinning, points, seed)
    sequence_points = np.empty((len(times), points, 3), dtype=np.float32)
    sequence_normals = np.empty_like(sequence_points)
    joint_positions = np.empty((len(times), 55, 3), dtype=np.float32)
    joint_rotations = np.empty((len(times), 55, 3, 3), dtype=np.float32)
    foot_low = np.empty(len(times), dtype=np.float64)
    foot_vertex = np.isin(np.argmax(skinning, axis=1), [7, 8, 10, 11])
    if not foot_vertex.any():
        raise ValueError("No SMPL-X foot vertices found")
    for begin, end in v5.inference_chunks(len(times), chunk_size):
        fields = v5.interpolated_chunk(raw, lower[begin:end], upper[begin:end], alpha[begin:end])
        with torch.inference_mode():
            posed = infer(fields)
        vertices = posed.vertices.detach().cpu().numpy() @ heading.T
        pos, nrm = core.barycentric_transport(vertices, faces, sample["face_indices"], sample["barycentric"])
        sequence_points[begin:end], sequence_normals[begin:end] = pos, nrm
        joint_positions[begin:end] = posed.joints[:, :55].detach().cpu().numpy() @ heading.T
        joint_rotations[begin:end] = heading @ core.global_joint_rotations(posed.full_pose.detach().cpu().numpy(), parents)
        foot_low[begin:end] = vertices[:, foot_vertex, 2].min(axis=1)
    xy_offset = -joint_positions[0, 0, :2].astype(np.float64)
    sequence_points[:, :, :2] += xy_offset
    joint_positions[:, :, :2] += xy_offset
    duration = (len(raw["trans"]) - 1) / raw["fps"]
    metadata = {"schema": core.SCHEMA, "source_file": str(source), "source_sha256": frozen[str(source)],
                "body_model": str(model_path), "body_model_sha256": frozen[str(model_path)],
                "source_gender": raw["gender"], "shape_policy": "source_betas_all_static",
                "num_betas": len(raw["betas"]), "flat_hand_mean": True, "expression": "zero",
                "optional_pose_fields_used": [key for key in v5.MOTION_FIELDS[2:] if key in raw],
                "optional_pose_fields_missing_zeroed": [key for key in v5.MOTION_FIELDS[2:] if key not in raw],
                "source_fps": raw["fps"], "target_fps": fps, "start": 0., "requested_duration": duration,
                "actual_duration": float(times[-1]), "frames": len(times), "points": points,
                "canonical_rotation": core.CANONICAL_ROTATION.tolist(), "posed_world_assumption": "AMASS Z-up metres",
                "posed_heading_rotation": heading.tolist(), "posed_xy_offset": xy_offset.tolist(),
                "actor_height": actor_height, "sampling_seed": seed,
                "surface_transport": "same_face_barycentric_on_posed_smplx_mesh",
                "normal_transport": "oriented_posed_triangle_cross_product",
                "license_note": "Local derived licensed AMASS/SMPL-X data; not approved for redistribution",
                "adapter_sha256": frozen[str(Path(__file__).resolve())], "input_code_protocol_sha256": frozen,
                "environment": core.environment_packages(("numpy", "scipy", "torch", "smplx")),
                "preprocessing_device": "cpu", "heading_reference": heading_proof, "model_axis_evidence": axis_evidence,
                "numeric_threads": {"torch_num_threads": torch.get_num_threads(),
                    "torch_num_interop_threads": torch.get_num_interop_threads(),
                    "environment": {key: os.environ.get(key) for key in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
                "full_sequence": v5.sequence_metadata(len(raw["trans"]), raw["fps"], times, budget),
                "not_yet_verified": ["v6 Stage II root initialization", "retargeted motion quality", "physics", "training"]}
    arrays = dict(metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False), **sample,
                  sequence_points=sequence_points, sequence_normals=sequence_normals,
                  canonical_joint_positions=rest_joints, canonical_joint_rotations=rest_rot,
                  joint_positions=joint_positions, joint_rotations=joint_rotations, foot_low=foot_low,
                  times=times, sample_lower=lower, sample_upper=upper, sample_alpha=alpha,
                  segment_names=np.array(core.SEGMENTS), parents=parents, betas=raw["betas"],
                  actor_height=actor_height, fps=fps)
    validate_values({key: np.asarray(value) for key, value in arrays.items()}, limits=limits)
    if any(core.sha256(Path(path)) != digest for path, digest in frozen.items()):
        raise ValueError("Full v6 source/model/code/protocol changed during preparation")
    v5._atomic_archive(output, arrays, limits.max_disk_bytes)
    return metadata


def load_prepared_source(path, *, limits=None):
    values = v5.load_prepared_source(path, limits=limits)
    values.pop("metadata")
    values["metadata"] = validate_values(values, limits=limits)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--body-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_source(args.source, args.body_model, args.output)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
