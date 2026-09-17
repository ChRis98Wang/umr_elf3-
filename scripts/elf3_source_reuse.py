"""Strict human-shape identity for reusing one ELF3 Stage-I correspondence.

This is only the human portion of the cache contract. The caller must separately
verify the robot geometry, sampling, learned artifacts, pinned UMR and runtime.
Actor folder names never authorize reuse. Full-source validation retains the
global clock and rejects cropped/relabeled sequences, including clips over 30s.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from umr_smplx_source import canonical_array_fingerprint, canonical_json

STATIC_ARRAYS = ("canonical_points", "canonical_normals", "canonical_joint_positions",
                 "canonical_joint_rotations", "face_indices", "barycentric", "segment",
                 "binding_joint_ids", "parents", "segment_names", "betas", "actor_height")
SHAPE_FIELDS = ("schema", "source_gender", "body_model_sha256", "shape_policy", "num_betas",
                "flat_hand_mean", "expression", "canonical_rotation", "sampling_seed", "points")


def human_stage1_identity(prepared):
    metadata = prepared["metadata"]
    missing = (set(STATIC_ARRAYS) - set(prepared)) | (set(SHAPE_FIELDS) - set(metadata))
    if missing:
        raise ValueError(f"Missing canonical cache identity fields: {sorted(missing)}")
    identity = {"schema": "bfm.elf3_human_stage1_identity/1",
                "shape_metadata": {k: metadata[k] for k in SHAPE_FIELDS},
                "static_arrays": {k: canonical_array_fingerprint(prepared[k]) for k in STATIC_ARRAYS}}
    return {**identity, "sha256": hashlib.sha256(canonical_json(identity)).hexdigest()}


def require_same_canonical(reference, candidate):
    before, after = human_stage1_identity(reference), human_stage1_identity(candidate)
    if before != after:
        raise ValueError("Stage I reuse rejected: actual canonical shape/samples differ; new learning required")
    return after


def load_replay_source(path, *, require_full=False):
    # Full loader inspects ZIP headers before allocating the moving surface.
    from umr_full_source_v5 import inspect_npz, Limits, load_prepared_source as full_load
    from umr_smplx_source import load_prepared_source as pilot_load
    path = Path(path)
    limits = Limits()
    if path.stat().st_size > limits.max_disk_bytes:
        raise ValueError("Prepared archive exceeds the bounded per-source budget")
    inspect_npz(path, max_uncompressed_bytes=limits.max_array_bytes)
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata_json"].item()))
    if "full_sequence" in metadata:
        return full_load(path)
    if require_full:
        raise ValueError("New-source execution requires proven full-sequence metadata, not a selected window")
    return pilot_load(path)
