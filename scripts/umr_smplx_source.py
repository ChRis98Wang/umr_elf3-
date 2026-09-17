#!/usr/bin/env python3
"""Experimental, true-mesh AMASS/SMPL-X source bridge for pinned unofficial UMR.

``prepare`` runs in the existing SMPL-X environment; ``retarget`` runs in the
isolated UMR environment. The interchange NPZ contains no Python pickle objects.
Surface samples keep the *same* triangle and barycentric coordinates throughout
the clip, including pose-dependent SMPL-X skin deformation. No BVH conversion or
procedural human replaces the SMPL-X mesh. This is an experimental data producer,
not an automatic replacement/training/promotion command.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

SCHEMA = "bfm.smplx_surface_source/1"
UMR_COMMIT = "0aa1855fe4f65a73681ffbd1d9f95ab1c2bad9ca"
CANONICAL_SETUP_ALGORITHM = "bfm.smplx_surface_canonical_stage1/1"
SETUP_CACHE_SCHEMA = "bfm.umr_smplx_setup_cache/1"
SETUP_ARTIFACTS = ("bodies.npz", "correspondence.npz")
# SMPL-X rest coordinates: X toward the left hip, Y up, Z forward. Robot: X forward, Y left,
# Z up. This is a proper rotation, not a reflection; only canonical rest uses it.
CANONICAL_ROTATION = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
SEGMENTS = [
    "pelvis", "torso", "chest", "neck", "head",
    "l_clavicle", "l_upperarm", "l_forearm", "l_hand",
    "r_clavicle", "r_upperarm", "r_forearm", "r_hand",
    "l_thigh", "l_shin", "l_foot", "l_toe",
    "r_thigh", "r_shin", "r_foot", "r_toe",
]
JOINT_SEGMENTS = [
    "pelvis", "l_thigh", "r_thigh", "torso", "l_shin", "r_shin",
    "torso", "l_foot", "r_foot", "chest", "l_toe", "r_toe", "neck",
    "l_clavicle", "r_clavicle", "head", "l_upperarm", "r_upperarm",
    "l_forearm", "r_forearm", "l_hand", "r_hand", "head", "head", "head",
] + ["l_hand"] * 15 + ["r_hand"] * 15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_packages(packages) -> dict:
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"executable": sys.executable, "python": sys.version, "packages": versions}


def verify_umr_checkout(repo: Path) -> str:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != Path(repo).resolve():
        raise ValueError("Expected the actual UMR repository root")
    commit = git("rev-parse", "HEAD")
    if commit != UMR_COMMIT or git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Expected the clean pinned unofficial UMR checkout, including no untracked files")
    return commit


def model_geometry_fingerprint(model) -> dict:
    """Hash loaded geometry/topology/dynamics, not merely the referring XML.

    This catches an edited STL/OBJ even when XML bytes and filenames are unchanged.
    Reloading from the source XML after the run checks the on-disk assets as well.
    """
    fields = (
        "mesh_vert", "mesh_face", "mesh_normal", "mesh_texcoord", "mesh_vertadr", "mesh_vertnum",
        "mesh_faceadr", "mesh_facenum", "mesh_normaladr", "mesh_normalnum", "mesh_texcoordadr",
        "mesh_texcoordnum", "mesh_pos", "mesh_quat", "mesh_scale",
        "body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat",
        "jnt_type", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited", "jnt_qposadr",
        "dof_armature", "dof_damping", "dof_frictionloss",
        "geom_type", "geom_bodyid", "geom_dataid", "geom_size", "geom_pos", "geom_quat",
        "geom_contype", "geom_conaffinity", "geom_group", "geom_friction", "geom_solref", "geom_solimp",
        "geom_margin", "geom_gap", "qpos0", "key_qpos",
    )
    arrays = {}
    for name in fields:
        if not hasattr(model, name):
            continue
        value = np.ascontiguousarray(getattr(model, name))
        arrays[name] = {"shape": list(value.shape), "dtype": value.dtype.str,
                        "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if not {"mesh_vert", "mesh_face", "geom_pos", "jnt_range"} <= set(arrays):
        raise ValueError("Robot model lacks required loaded geometry arrays")
    encoded = json.dumps(arrays, sort_keys=True, separators=(",", ":")).encode()
    return {"schema": "bfm.loaded_mujoco_geometry/1", "sha256": hashlib.sha256(encoded).hexdigest(),
            "arrays": arrays}


def canonical_json(value) -> bytes:
    """Stable JSON, rejecting NaN and unserializable configuration values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_array_fingerprint(value) -> dict:
    array = np.asarray(value)
    # Upstream SurfacePointCloud uses object arrays for segment names. Hash their
    # actual strings, never process-dependent Python object pointer bytes.
    if array.dtype.hasobject:
        if not all(isinstance(item, str) for item in array.ravel()):
            raise ValueError("Canonical setup cannot fingerprint arbitrary Python objects")
        array = array.astype(str)
    if array.dtype.kind not in "biufcUS" or (array.dtype.kind in "fcu" and not np.isfinite(array).all()):
        raise ValueError("Canonical setup array has unsupported or nonfinite values")
    array = np.ascontiguousarray(array)
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def canonical_setup_identity(prepared: dict, *, scale: float, robot_geometry: dict,
                             robot_xml_sha256: str, robot_config: dict, sampling_config: dict,
                             correspondence_config: dict, robot_cloud: dict, epochs: int,
                             device: str, runtime: dict, umr_commit=UMR_COMMIT,
                             algorithm=CANONICAL_SETUP_ALGORITHM) -> dict:
    """Identity of Stage I's actual canonical inputs, independent of the motion.

    Source paths/hashes, frame times, moving vertices, heading/floor normalization,
    and Stage II settings are deliberately excluded. Actual static shape and
    sampled/bound canonical arrays are included, not merely an actor-directory ID.
    Runtime/package/device details are conservative numerical provenance: sharing
    across clips is supported, silently sharing across environments is not.
    """
    metadata = prepared["metadata"]
    shape_fields = ("source_gender", "body_model_sha256", "shape_policy", "num_betas",
                    "flat_hand_mean", "expression", "canonical_rotation", "sampling_seed", "points")
    missing = set(shape_fields) - set(metadata)
    if missing or "betas" not in prepared:
        raise ValueError(f"Canonical setup needs explicit model/shape metadata: {sorted(missing)}")
    betas = np.asarray(prepared["betas"])
    if (betas.ndim != 1 or betas.size != metadata["num_betas"] or not betas.size
            or betas.dtype.kind not in "fiu" or not np.isfinite(betas).all()):
        raise ValueError("Canonical setup requires every actual static source beta")
    if not np.isfinite(scale) or not .3 < scale < 2.:
        raise ValueError("Invalid canonical setup scale")
    source_keys = ("canonical_points", "canonical_normals", "canonical_joint_positions",
                   "canonical_joint_rotations", "face_indices", "barycentric", "segment",
                   "binding_joint_ids", "parents", "segment_names", "betas", "actor_height")
    # Absolute XML location is incidental. Its content plus loaded mesh/topology,
    # joint conventions, and RobotSpec configuration determine the robot setup.
    robot_spec = {key: value for key, value in robot_config.items() if key != "xml"}
    identity = {"schema": CANONICAL_SETUP_ALGORITHM, "adapter_algorithm": algorithm,
                "umr_commit": umr_commit, "source_schema": metadata["schema"],
                "source_shape": {key: metadata[key] for key in shape_fields},
                "source_canonical_arrays": {key: canonical_array_fingerprint(prepared[key])
                                             for key in source_keys},
                "scale": float(scale), "robot_xml_sha256": robot_xml_sha256,
                "loaded_robot_geometry": robot_geometry, "robot_config": robot_spec,
                "sampled_robot_arrays": {key: canonical_array_fingerprint(value)
                                          for key, value in sorted(robot_cloud.items())},
                "sampling_config": sampling_config, "correspondence_config": correspondence_config,
                "epochs": int(epochs), "requested_device": device, "runtime": runtime}
    # Detach nested mutable configuration dictionaries before handing the identity
    # to a potentially long-running builder.
    return json.loads(canonical_json(identity))


def canonical_setup_key(identity: dict) -> str:
    return hashlib.sha256(canonical_json(identity)).hexdigest()


def validate_setup_arrays(entry: Path, identity: dict) -> None:
    """Inspect required numeric fields without unpickling upstream debug metadata."""
    count = int(identity["source_shape"]["points"])
    key = canonical_setup_key(identity)
    # This is the fixed commit's documented Stage I stamp, not our cache key.
    upstream_parts = ("correspondence/2", key, identity["correspondence_config"], identity["epochs"])
    expected_corr = hashlib.sha1(json.dumps(upstream_parts, sort_keys=True, default=str,
                                           ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    expected = {
        "bodies.npz": {"human_points": (count, 3), "human_normals": (count, 3),
                       "robot_points": (count, 3), "robot_normals": (count, 3)},
        "correspondence.npz": {"bind_body_ids": (count,), "bind_local_pos": (count, 3),
                               "bind_local_normal": (count, 3), "bind_snap_distance": (count,),
                               "bind_world_pos": (count, 3), "inherited_segment": (count,)}}
    try:
        for name, shapes in expected.items():
            with np.load(entry / name, allow_pickle=False) as arrays:
                if scalar_text(arrays["stamp"], "stamp") != (key if name == "bodies.npz" else expected_corr):
                    raise ValueError(f"Canonical setup array stamp mismatch: {name}")
                for field, shape in shapes.items():
                    value = arrays[field]
                    if value.shape != shape or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                        raise ValueError(f"Invalid canonical setup numeric array: {name}:{field}")
                    if field in ("bind_body_ids", "inherited_segment"):
                        if value.dtype.kind not in "iu" or np.any(value < 0):
                            raise ValueError(f"Invalid canonical setup integer binding: {field}")
                        if field == "inherited_segment" and np.any(value >= len(SEGMENTS)):
                            raise ValueError("Invalid canonical setup inherited segment")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Canonical setup cache NPZ validation failed: {exc}") from exc


def validate_setup_cache_entry(entry: Path, identity: dict) -> dict:
    """Fail closed on partial, redirected, empty, or changed cache artifacts."""
    entry = Path(entry)
    if entry.is_symlink() or not entry.is_dir():
        raise ValueError(f"Canonical setup cache entry must be a real directory: {entry}")
    expected_names = {*SETUP_ARTIFACTS, "manifest.json"}
    if {path.name for path in entry.iterdir()} != expected_names:
        raise ValueError(f"Canonical setup cache is incomplete or contains unexpected files: {entry}")
    manifest_path = entry / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Canonical setup manifest must be a regular file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Canonical setup cache manifest is unreadable") from exc
    key = canonical_setup_key(identity)
    if (manifest.get("schema") != SETUP_CACHE_SCHEMA or manifest.get("key_sha256") != key
            or manifest.get("identity") != identity
            or set(manifest.get("artifacts", {})) != set(SETUP_ARTIFACTS)):
        raise ValueError("Canonical setup manifest does not match requested semantic identity")
    artifacts = {}
    for name in SETUP_ARTIFACTS:
        path = entry / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Canonical setup cache artifact is missing, empty, or redirected: {name}")
        actual = {"sha256": sha256(path), "bytes": path.stat().st_size}
        if actual != manifest["artifacts"][name]:
            raise ValueError(f"Canonical setup cache artifact integrity mismatch: {name}")
        artifacts[name] = actual
    validate_setup_arrays(entry, identity)
    return {"key_sha256": key, "manifest_sha256": sha256(manifest_path), "artifacts": artifacts}


@contextlib.contextmanager
def exclusive_setup_lock(cache_root: Path, key: str):
    """Crash-released Linux flock; contention fails quickly instead of hanging."""
    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(Path(cache_root) / f".{key}.lock", flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Canonical setup {key} is being built; retry after its owner finishes") from exc
        yield
    finally:
        os.close(descriptor)


def obtain_cached_setup(cache_root: Path, identity: dict, build, verify_inputs, *, provenance=None):
    """Build privately and atomically publish once, or validate an immutable hit.

    ``build(directory, key)`` writes exactly bodies.npz and correspondence.npz.
    It is never called on a finished entry. Failed partial work is removed only
    from this call's mkdtemp directory; any pre-existing invalid entry is retained
    and rejected so cache corruption cannot silently become a recomputation.
    """
    cache_root = Path(cache_root)
    if cache_root.is_symlink():
        raise ValueError("Canonical setup cache root must not be a symlink")
    cache_root.mkdir(parents=True, exist_ok=True)
    identity = json.loads(canonical_json(identity))
    key = canonical_setup_key(identity)
    entry = cache_root / key
    with exclusive_setup_lock(cache_root, key):
        verify_inputs()
        if entry.exists() or entry.is_symlink():
            evidence = validate_setup_cache_entry(entry, identity)
            return entry, {"enabled": True, "hit": True, "entry": str(entry), **evidence}
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}.building-", dir=cache_root))
        try:
            build(temporary, key)
            verify_inputs()
            if {path.name for path in temporary.iterdir()} != set(SETUP_ARTIFACTS):
                raise ValueError("Canonical setup builder did not create exactly both required artifacts")
            artifacts = {}
            for name in SETUP_ARTIFACTS:
                path = temporary / name
                if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                    raise ValueError(f"Canonical setup builder emitted an empty or redirected artifact: {name}")
                artifacts[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
            manifest = {"schema": SETUP_CACHE_SCHEMA, "key_sha256": key, "identity": identity,
                        "artifacts": artifacts, "producer": provenance or {}}
            with (temporary / "manifest.json").open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            evidence = validate_setup_cache_entry(temporary, identity)
            for path in temporary.iterdir():
                path.chmod(0o444)
            temporary.chmod(0o555)
            # The per-key exclusive lock serializes all publishers. Refuse even
            # an empty pre-existing entry; rename publishes the full manifest and
            # both files as a single directory operation, never a partial setup.
            if entry.exists() or entry.is_symlink():
                raise FileExistsError(entry)
            temporary.rename(entry)
            return entry, {"enabled": True, "hit": False, "entry": str(entry), **evidence}
        finally:
            if temporary.exists():
                temporary.chmod(0o700)
                shutil.rmtree(temporary)


def export_scalebfm_motion(path: Path, qpos, fps: float, joint_names) -> dict:
    """Use the common named-joint/quaternion converter, preserving original UMR PKL."""
    import pickle
    # CLI execution places scripts/ on sys.path; explicit addition also supports
    # importlib-based users/tests without making scripts a required package.
    module_dir = str(Path(__file__).resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    from umr_backend import convert_umr_payload
    motion = convert_umr_payload({"qpos": qpos, "fps": fps, "dof_names": list(joint_names)})
    with Path(path).open("xb") as stream:
        pickle.dump(motion, stream, protocol=4)
    return {"path": str(path), "sha256": sha256(Path(path)), "frames": len(motion["root_pos"]),
            "fps": motion["fps"], "root_quaternion_order": "xyzw",
            "converter_sha256": sha256(Path(__file__).with_name("umr_backend.py")),
            "contract": "root_pos/root_rot_xyzw/dof_pos/fps", "quality_accepted": False}


def save_npz_exclusive(path: Path, **arrays) -> None:
    """Do not overwrite a prior artifact, including a symlink."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        if np.asarray(value).dtype.hasobject:
            raise ValueError(f"Interchange array must not require pickle: {key}")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def scalar_text(value, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be one plain string")
    value = array.reshape(()).item()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_amass(path: Path) -> dict:
    """Load only numeric/plain-string fields, never unpickle unrelated markers."""
    with np.load(path, allow_pickle=False) as src:
        required = ("root_orient", "pose_body", "trans", "betas", "gender",
                    "surface_model_type", "mocap_frame_rate")
        missing = set(required) - set(src.files)
        if missing:
            raise ValueError(f"Missing AMASS fields: {sorted(missing)}")
        out = {key: np.array(src[key], copy=True) for key in required}
        for key in ("pose_hand", "pose_jaw", "pose_eye"):
            if key in src:
                out[key] = np.array(src[key], copy=True)
    out["gender"] = scalar_text(out["gender"], "gender").lower()
    if out["gender"] not in ("male", "female", "neutral"):
        raise ValueError("Unknown gender; no implicit neutral-model fallback")
    if scalar_text(out["surface_model_type"], "surface_model_type").lower() != "smplx":
        raise ValueError("This adapter requires AMASS SMPL-X, not SMPL/SMPL-H")
    rate = np.asarray(out["mocap_frame_rate"])
    if rate.size != 1:
        raise ValueError("Frame rate must be scalar")
    out["fps"] = float(rate.reshape(()))
    if not np.isfinite(out["fps"]) or not 0 < out["fps"] <= 1000:
        raise ValueError("Invalid source frame rate")
    count = len(out["root_orient"])
    if count < 3:
        raise ValueError("AMASS motion needs at least 3 frames")
    for key, width in (("root_orient", 3), ("pose_body", 63), ("trans", 3),
                       ("pose_hand", 90), ("pose_jaw", 3), ("pose_eye", 6)):
        if key not in out:
            continue
        value = out[key]
        if value.shape != (count, width) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key}: expected finite ({count}, {width})")
    betas = out["betas"]
    if betas.ndim == 2:
        if betas.shape[0] not in (1, count) or not np.allclose(betas, betas[:1], atol=1e-8, rtol=0):
            raise ValueError("Time-varying shape needs a different canonical setup")
        betas = betas[0]
    if betas.ndim != 1 or not 1 <= len(betas) <= 300 or not np.isfinite(betas).all():
        raise ValueError("Invalid static SMPL-X betas")
    out["betas"] = np.asarray(betas, dtype=np.float32)
    return out


def model_file_for_gender(path: Path, gender: str) -> Path:
    """A file with another gender name is not made valid by smplx.create(gender=...)."""
    path = Path(path).resolve()
    if path.is_dir():
        candidates = [path / f"SMPLX_{gender.upper()}.npz",
                      path / "smplx" / f"SMPLX_{gender.upper()}.npz"]
        found = [item for item in candidates if item.is_file()]
        if len(found) != 1:
            raise ValueError("Expected exactly one gender-specific SMPL-X NPZ")
        path = found[0]
    if not path.is_file() or path.suffix.lower() != ".npz":
        raise ValueError("Use an existing licensed SMPL-X NPZ model")
    match = re.match(r"^SMPLX_(NEUTRAL|MALE|FEMALE)(?:[_.]|$)", path.name.upper())
    if not match or match[1].lower() != gender:
        raise ValueError(f"Model filename must identify source gender {gender}; refusing substitution")
    return path


def sampling_grid(count: int, source_fps: float, fps: float, start: float, duration: float):
    if not all(np.isfinite(x) for x in (source_fps, fps, start, duration)):
        raise ValueError("Sampling parameters must be finite")
    if count < 3 or source_fps <= 0 or fps <= 0 or start < 0 or not 0 < duration <= 30:
        raise ValueError("A bounded 0 < duration <= 30 seconds is required")
    end = min(start + duration, (count - 1) / source_fps)
    if end - start < 1 / fps:
        raise ValueError("Requested interval contains fewer than two output frames")
    times = start + np.arange(int(np.floor((end - start) * fps + 1e-9)) + 1) / fps
    # Real fixed-rate samples; never relabel a stretched linspace as the target fps.
    fractional = np.clip(times * source_fps, 0, count - 1)
    nearest = np.rint(fractional)
    fractional = np.where(np.abs(fractional - nearest) < 1e-9, nearest, fractional)
    lower = np.floor(fractional).astype(np.int64)
    upper = np.minimum(lower + 1, count - 1)
    return times, lower, upper, fractional - lower


def interpolate_rotvec(value, lower, upper, alpha):
    from scipy.spatial.transform import Rotation
    value = np.asarray(value).reshape(len(value), -1, 3)
    q = Rotation.from_rotvec(value.reshape(-1, 3)).as_quat().reshape(*value.shape[:-1], 4)
    a, b = q[lower], q[upper]
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    theta = np.arccos(np.clip(np.abs(dot), 0., 1.))
    sin = np.sin(theta)
    near = sin < 1e-7
    divisor = np.where(near, 1., sin)
    t = alpha[:, None, None]
    interp = np.where(near, (1 - t) * a + t * b,
                      np.sin((1 - t) * theta) / divisor * a + np.sin(t * theta) / divisor * b)
    interp /= np.linalg.norm(interp, axis=-1, keepdims=True)
    return Rotation.from_quat(interp.reshape(-1, 4)).as_rotvec().reshape(len(alpha), -1)


def global_joint_rotations(full_pose: np.ndarray, parents: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    full_pose = np.asarray(full_pose).reshape(len(full_pose), -1, 3)
    if full_pose.shape[1] != len(parents) or parents[0] != -1:
        raise ValueError("SMPL-X full pose and joint hierarchy disagree")
    local = Rotation.from_rotvec(full_pose.reshape(-1, 3)).as_matrix().reshape(
        len(full_pose), len(parents), 3, 3)
    world = local.copy()
    for index in range(1, len(parents)):
        parent = int(parents[index])
        if not 0 <= parent < index:
            raise ValueError("Joint hierarchy must be parent-before-child")
        world[:, index] = world[:, parent] @ local[:, index]
    return world


def barycentric_transport(vertices, faces, face_indices, barycentric):
    """Exact material surface positions and oriented triangle normals (not joint FK)."""
    vertices = np.asarray(vertices)
    triangles = vertices[..., np.asarray(faces)[np.asarray(face_indices)], :]
    weights = np.asarray(barycentric)
    if weights.shape != (len(face_indices), 3) or not np.isfinite(weights).all():
        raise ValueError("Invalid barycentric weights")
    if np.any(weights < -1e-9) or not np.allclose(weights.sum(axis=1), 1., atol=1e-7):
        raise ValueError("Barycentric weights must be inside triangles and sum to one")
    points = np.einsum("...nvc,nv->...nc", triangles, weights)
    cross = np.cross(triangles[..., 1, :] - triangles[..., 0, :],
                     triangles[..., 2, :] - triangles[..., 0, :])
    length = np.linalg.norm(cross, axis=-1, keepdims=True)
    if not np.isfinite(points).all() or not np.isfinite(length).all() or np.any(length < 1e-12):
        raise ValueError("Degenerate/nonfinite sampled SMPL-X triangle")
    return points, cross / length


def sample_canonical_surface(vertices, faces, skinning_weights, count: int, seed: int = 0):
    """Area-weighted stratified samples; every represented anatomy gets >= 8 points."""
    if not 512 <= count <= 8192:
        raise ValueError("Use between 512 and 8192 surface samples")
    vertices, faces, skinning_weights = map(np.asarray, (vertices, faces, skinning_weights))
    if skinning_weights.shape != (len(vertices), len(JOINT_SEGMENTS)):
        raise ValueError("Expected full 55-joint SMPL-X skinning weights")
    membership = np.zeros((55, len(SEGMENTS)))
    membership[np.arange(55), [SEGMENTS.index(x) for x in JOINT_SEGMENTS]] = 1.
    triangle_weights = skinning_weights[faces].mean(axis=1)
    face_segment = np.argmax(triangle_weights @ membership, axis=1)
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1) * .5
    masses = np.bincount(face_segment, weights=area, minlength=len(SEGMENTS))
    active = np.flatnonzero(masses > 1e-12)
    if not len(active) or count < 8 * len(active):
        raise ValueError("Insufficient nondegenerate canonical surface coverage")
    quota = np.zeros(len(SEGMENTS), dtype=int)
    quota[active] = 8
    expected = (count - quota.sum()) * masses / masses.sum()
    quota += np.floor(expected).astype(int)
    for segment in np.argsort(-(expected - np.floor(expected)))[:count - quota.sum()]:
        quota[segment] += 1
    rng = np.random.default_rng(seed)
    selections, labels = [], []
    for segment in active:
        pool = np.flatnonzero((face_segment == segment) & (area > 1e-12))
        selections.append(rng.choice(pool, quota[segment], p=area[pool] / area[pool].sum()))
        labels.append(np.full(quota[segment], segment, dtype=np.int64))
    face_indices = np.concatenate(selections)
    # sqrt mapping produces a uniform area distribution on a triangle.
    u, v = rng.random((2, count))
    u = np.sqrt(u)
    weights = np.column_stack((1 - u, u * (1 - v), u * v))
    points, normals = barycentric_transport(vertices, faces, face_indices, weights)
    joint_weights = np.einsum("nvj,nv->nj", skinning_weights[faces[face_indices]], weights)
    return {"face_indices": face_indices, "barycentric": weights,
            "canonical_points": points, "canonical_normals": normals,
            "segment": np.concatenate(labels), "binding_joint_ids": np.argmax(joint_weights, axis=1)}


def prepare_source(source: Path, body_model: Path, output: Path, *, start=0., duration=3.,
                   fps=50., points=4096, chunk_size=16, seed=0) -> dict:
    import smplx
    import torch
    from scipy.spatial.transform import Rotation

    source, output = Path(source).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not 1 <= chunk_size <= 64 or fps != 50.:
        raise ValueError("Current pilot requires 50 Hz and 1 <= chunk size <= 64")
    raw = load_amass(source)
    model_path = model_file_for_gender(body_model, raw["gender"])
    source_digest, model_digest = sha256(source), sha256(model_path)
    body = smplx.create(str(model_path), model_type="smplx", gender=raw["gender"],
                        ext="npz", use_pca=False, flat_hand_mean=True,
                        num_betas=len(raw["betas"]), batch_size=1)
    if body.num_betas != len(raw["betas"]):
        raise ValueError("Model does not support every source beta; refusing silent truncation")
    parents = body.parents.detach().cpu().numpy()
    if len(parents) != 55:
        raise ValueError("Expected 55 SMPL-X kinematic joints")
    faces = np.asarray(body.faces, dtype=np.int64)
    skinning = body.lbs_weights.detach().cpu().numpy()
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32)

    def infer(fields):
        count = len(fields["root_orient"])
        hand = fields.get("pose_hand", np.zeros((count, 90)))
        eye = fields.get("pose_eye", np.zeros((count, 6)))
        return body(betas=tensor(raw["betas"])[None].expand(count, -1),
                    global_orient=tensor(fields["root_orient"]),
                    body_pose=tensor(fields["pose_body"]), transl=tensor(fields["trans"]),
                    left_hand_pose=tensor(hand[:, :45]), right_hand_pose=tensor(hand[:, 45:]),
                    jaw_pose=tensor(fields.get("pose_jaw", np.zeros((count, 3)))),
                    leye_pose=tensor(eye[:, :3]), reye_pose=tensor(eye[:, 3:]),
                    expression=torch.zeros((count, body.num_expression_coeffs)),
                    return_verts=True, return_full_pose=True)

    with torch.inference_mode():
        rest = infer({"root_orient": np.zeros((1, 3)), "pose_body": np.zeros((1, 63)),
                      "trans": np.zeros((1, 3))})
    rest_vertices = rest.vertices[0].detach().cpu().numpy() @ CANONICAL_ROTATION.T
    rest_joints = rest.joints[0, :55].detach().cpu().numpy() @ CANONICAL_ROTATION.T
    rest_rot = CANONICAL_ROTATION @ global_joint_rotations(
        rest.full_pose.detach().cpu().numpy(), parents)[0]
    # Normalize canonical translation only. The moving pelvis is SMPL-X output
    # joint 0, not transl: shape changes the regressed pelvis offset.
    canonical_offset = np.array([-rest_joints[0, 0], -rest_joints[0, 1], -rest_vertices[:, 2].min()])
    rest_vertices += canonical_offset
    rest_joints += canonical_offset
    actor_height = float(np.ptp(rest_vertices[:, 2]))
    if not .8 < actor_height < 2.5:
        raise ValueError("Unexpected canonical human height; check model/units")
    sample = sample_canonical_surface(rest_vertices, faces, skinning, points, seed)
    times, lower, upper, alpha = sampling_grid(len(raw["trans"]), raw["fps"], fps, start, duration)
    target_fields = {}
    for key in ("root_orient", "pose_body", "pose_hand", "pose_jaw", "pose_eye"):
        if key in raw:
            target_fields[key] = interpolate_rotvec(raw[key], lower, upper, alpha)
    target_fields["trans"] = ((1 - alpha[:, None]) * raw["trans"][lower]
                              + alpha[:, None] * raw["trans"][upper])
    forward = Rotation.from_rotvec(target_fields["root_orient"][0]).apply([0., 0., 1.])
    if np.linalg.norm(forward[:2]) < .1:
        raise ValueError("Cannot infer horizontal heading from first pelvis pose; choose another pilot interval")
    heading = Rotation.from_euler("z", -np.arctan2(forward[1], forward[0])).as_matrix()
    sequence_points = np.empty((len(times), points, 3), dtype=np.float32)
    sequence_normals = np.empty_like(sequence_points)
    joint_positions = np.empty((len(times), 55, 3), dtype=np.float32)
    joint_rotations = np.empty((len(times), 55, 3, 3), dtype=np.float32)
    foot_vertex = np.isin(np.argmax(skinning, axis=1), [7, 8, 10, 11])
    if not foot_vertex.any():
        raise ValueError("No SMPL-X foot vertices found")
    foot_low = np.empty(len(times), dtype=np.float64)
    for first in range(0, len(times), chunk_size):
        last = min(first + chunk_size, len(times))
        with torch.inference_mode():
            posed = infer({key: value[first:last] for key, value in target_fields.items()})
        vertices = posed.vertices.detach().cpu().numpy() @ heading.T
        pos, nrm = barycentric_transport(vertices, faces, sample["face_indices"], sample["barycentric"])
        sequence_points[first:last], sequence_normals[first:last] = pos, nrm
        joint_positions[first:last] = posed.joints[:, :55].detach().cpu().numpy() @ heading.T
        joint_rotations[first:last] = heading @ global_joint_rotations(posed.full_pose.detach().cpu().numpy(), parents)
        foot_low[first:last] = vertices[:, foot_vertex, 2].min(axis=1)
    xy_offset = -joint_positions[0, 0, :2].astype(np.float64)
    sequence_points[:, :, :2] += xy_offset
    joint_positions[:, :, :2] += xy_offset
    metadata = {
        "schema": SCHEMA, "source_file": str(source), "source_sha256": source_digest,
        "body_model": str(model_path), "body_model_sha256": model_digest,
        "source_gender": raw["gender"], "shape_policy": "source_betas_all_static",
        "num_betas": len(raw["betas"]), "flat_hand_mean": True, "expression": "zero",
        "optional_pose_fields_used": [key for key in ("pose_hand", "pose_jaw", "pose_eye") if key in raw],
        "optional_pose_fields_missing_zeroed": [key for key in ("pose_hand", "pose_jaw", "pose_eye") if key not in raw],
        "source_fps": raw["fps"], "target_fps": fps, "start": start, "requested_duration": duration,
        "actual_duration": float(times[-1] - times[0]), "frames": len(times), "points": points,
        "canonical_rotation": CANONICAL_ROTATION.tolist(), "posed_world_assumption": "AMASS Z-up metres",
        "posed_heading_rotation": heading.tolist(), "posed_xy_offset": xy_offset.tolist(),
        "actor_height": actor_height, "sampling_seed": seed,
        "surface_transport": "same_face_barycentric_on_posed_smplx_mesh",
        "normal_transport": "oriented_posed_triangle_cross_product",
        "license_note": "Local derived licensed AMASS/SMPL-X data; not approved for redistribution",
        "adapter_sha256": sha256(Path(__file__)),
        "environment": environment_packages(("numpy", "scipy", "torch", "smplx")),
        "preprocessing_device": "cpu",
    }
    if sha256(source) != source_digest or sha256(model_path) != model_digest:
        raise ValueError("Source/model changed during preprocessing")
    save_npz_exclusive(output, metadata_json=json.dumps(metadata, sort_keys=True),
                       **sample, sequence_points=sequence_points, sequence_normals=sequence_normals,
                       canonical_joint_positions=rest_joints, canonical_joint_rotations=rest_rot,
                       joint_positions=joint_positions, joint_rotations=joint_rotations,
                       foot_low=foot_low, times=times, sample_lower=lower, sample_upper=upper,
                       sample_alpha=alpha, segment_names=np.array(SEGMENTS), parents=parents,
                       betas=raw["betas"], actor_height=actor_height, fps=fps)
    return metadata


def load_prepared_source(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        values = {key: np.array(data[key], copy=True) for key in data.files}
    metadata = json.loads(scalar_text(values["metadata_json"], "metadata_json"))
    if metadata.get("schema") != SCHEMA:
        raise ValueError("Unsupported source schema")
    count, points = metadata["frames"], metadata["points"]
    if (type(count) is not int or type(points) is not int or not 2 <= count <= 1501
            or not 512 <= points <= 8192):
        raise ValueError("Prepared source exceeds the bounded pilot contract")
    shapes = {"canonical_points": (points, 3), "canonical_normals": (points, 3),
              "sequence_points": (count, points, 3), "sequence_normals": (count, points, 3),
              "canonical_joint_positions": (55, 3), "canonical_joint_rotations": (55, 3, 3),
              "joint_positions": (count, 55, 3), "joint_rotations": (count, 55, 3, 3),
              "face_indices": (points,), "barycentric": (points, 3), "segment": (points,),
              "binding_joint_ids": (points,), "times": (count,), "foot_low": (count,),
              "sample_lower": (count,), "sample_upper": (count,), "sample_alpha": (count,),
              "parents": (55,)}
    for key, shape in shapes.items():
        if (values[key].shape != shape or values[key].dtype.kind not in "iuf"
                or not np.isfinite(values[key]).all()):
            raise ValueError(f"Invalid prepared source {key}")
    for key in ("canonical_normals", "sequence_normals"):
        if not np.allclose(np.linalg.norm(values[key], axis=-1), 1., atol=1e-5):
            raise ValueError("Prepared surface normals must be unit length")
    if count < 2 or float(values["fps"]) != 50. or not np.allclose(np.diff(values["times"]), .02, atol=1e-8):
        raise ValueError("Prepared source must use an actual 50 Hz clock")
    for key, maximum in (("binding_joint_ids", 55), ("segment", len(SEGMENTS))):
        if (values[key].dtype.kind not in "iu" or np.any(values[key] < 0)
                or np.any(values[key] >= maximum)):
            raise ValueError(f"Invalid prepared {key}")
    for key in ("face_indices", "sample_lower", "sample_upper"):
        if values[key].dtype.kind not in "iu" or np.any(values[key] < 0):
            raise ValueError(f"Invalid integer indices: {key}")
    if (np.any(values["sample_upper"] < values["sample_lower"])
            or np.any(values["sample_upper"] > values["sample_lower"] + 1)
            or np.any(values["sample_alpha"] < 0) or np.any(values["sample_alpha"] > 1)):
        raise ValueError("Invalid source interpolation indices/weights")
    barycentric = values["barycentric"]
    if (np.any(barycentric < -1e-9) or np.any(barycentric > 1 + 1e-9)
            or not np.allclose(barycentric.sum(axis=1), 1., atol=1e-7, rtol=0)):
        raise ValueError("Invalid prepared barycentric weights")
    for key in ("canonical_joint_rotations", "joint_rotations"):
        rotation = values[key].astype(np.float64)
        if (not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-5, rtol=0)
                or not np.allclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0)):
            raise ValueError(f"Prepared {key} must be proper rotation matrices")
    parents = values["parents"]
    if (parents.dtype.kind not in "iu" or parents[0] != -1
            or any(not 0 <= int(parents[index]) < index for index in range(1, 55))):
        raise ValueError("Invalid prepared SMPL-X hierarchy")
    if (np.asarray(values["actor_height"]).shape != () or not .8 < float(values["actor_height"]) < 2.5):
        raise ValueError("Invalid prepared actor height")
    if list(values["segment_names"]) != SEGMENTS:
        raise ValueError("Unknown segment conventions")
    values["metadata"] = metadata
    return values


class SmplxSurfaceHuman:
    """Minimal UMR human protocol. Moving targets are NOT rigid-link transported."""

    def __init__(self, source: dict, robot_height: float):
        self.source = source
        self.scale = float(robot_height) / float(source["actor_height"])
        if not np.isfinite(self.scale) or not .3 < self.scale < 2.:
            raise ValueError("Invalid human-to-robot height scaling")
        self.ground_offset = -float(np.percentile(source["foot_low"], 1)) * self.scale
        self.body_ids = np.arange(55, dtype=np.int64)
        self.data = SimpleNamespace(xpos=None, xmat=None)
        self.set_tpose()

    def set_tpose(self):
        self.data.xpos = self.source["canonical_joint_positions"] * self.scale
        self.data.xmat = self.source["canonical_joint_rotations"].reshape(55, 9)

    def set_frame(self, frame: int):
        self.data.xpos = self.source["joint_positions"][frame].astype(np.float64) * self.scale
        self.data.xpos[:, 2] += self.ground_offset
        self.data.xmat = self.source["joint_rotations"][frame].reshape(55, 9)

    def targets(self, frame: int):
        self.set_frame(frame)
        points = self.source["sequence_points"][frame].astype(np.float64) * self.scale
        points[:, 2] += self.ground_offset
        return points, self.source["sequence_normals"][frame].astype(np.float64)


def retargeter_class(base):
    """Dependency injection keeps preprocessing/tests independent of Mink/UMR."""
    class SmplxMeshRetargeter(base):
        def __init__(self, *args, **kwargs):
            if float(kwargs.get("tpose_offset", 0.)) != 0.:
                raise ValueError("Rigid T-pose offset compensation is not defined for deformable SMPL-X")
            kwargs["tpose_offset"] = 0.
            super().__init__(*args, **kwargs)

        def human_targets(self, frame):
            return self.human.targets(frame)

        def initialize_root(self, frame):
            """Use SMPL-X local +Z as forward, not UMR's BVH local +X.

            The source exposes genuine SMPL-X joint matrices so that canonical
            link bindings stay consistent. Converting only this heading query
            avoids corrupting those matrices or applying the canonical basis a
            second time. Root XY and initial target height follow stock UMR.
            """
            from scipy.spatial.transform import Rotation
            self.human.set_frame(frame)
            pelvis = int(self.human.body_ids[0])
            hips = self.human.data.xpos[pelvis]
            smplx_rotation = self.human.data.xmat[pelvis].reshape(3, 3)
            # R_robot = R_smplx @ CANONICAL_ROTATION.T; its first column is
            # exactly SMPL-X R[:, 2]. In an aligned rest pose R_smplx=C,
            # R_robot=I and the correct robot yaw is zero, not +90 degrees.
            forward = smplx_rotation[:, 2]
            if not np.isfinite(forward).all() or np.linalg.norm(forward[:2]) < .1:
                raise ValueError("SMPL-X pelvis does not define a horizontal initial heading")
            yaw = float(np.arctan2(forward[1], forward[0]))
            q = self.robot.model.key_qpos[0].copy() if self.robot.model.nkey else self.robot.q.copy()
            q[:2] = hips[:2]
            q[3:7] = Rotation.from_euler("z", yaw).as_quat(scalar_first=True)
            self.robot.set_qpos(q)
    return SmplxMeshRetargeter


def retarget_source(source: Path, umr_root: Path, output: Path, *, robot_xml: Path,
                    epochs=2500, device="cuda", iterations=6, setup_cache: Path | None = None) -> dict:
    source, umr_root, output, robot_xml = map(lambda p: Path(p).resolve(), (source, umr_root, output, robot_xml))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not 1 <= epochs <= 10000 or not 1 <= iterations <= 20:
        raise ValueError("Bounded pilot epochs/iterations required")
    if setup_cache is not None:
        requested_cache = Path(setup_cache)
        if requested_cache.is_symlink():
            raise ValueError("Canonical setup cache root must not be a symlink")
        setup_cache = requested_cache.resolve()
        local_root = Path(__file__).resolve().parents[1] / "local"
        if setup_cache == local_root or not setup_cache.is_relative_to(local_root):
            raise ValueError("Pilot setup cache must use a dedicated directory below this repository's local/")
    commit = verify_umr_checkout(umr_root)
    sys.path.insert(0, str(umr_root))
    from umr.bodies.robot import RobotBody, RobotSpec
    from umr.bodies.surface import SurfacePointCloud, sample_model_surface
    from umr.config import load_config
    from umr.paths import SetupLayout
    from umr.retarget.binding import LinkBinding
    from umr.retarget.export import save_motion_pkl
    from umr.retarget.pipeline import UMRRetargeter
    from umr.stages import learn_correspondence
    import mujoco

    prepared = load_prepared_source(source)
    cfg = load_config("g1")
    cfg["robot"]["xml"] = str(robot_xml)
    cfg["retarget"]["tpose_offset"] = 0.
    cfg["retarget"]["iterations"] = iterations
    robot_hash, source_hash = sha256(robot_xml), sha256(source)
    adapter_hash = sha256(Path(__file__))
    robot = RobotBody(robot_xml, RobotSpec.from_config(cfg.robot))
    geometry = model_geometry_fingerprint(robot.model)
    # This bridge is specifically for the shared 29-hinge G1 target. Do not
    # silently export another morphology under the same pipeline name.
    joint_ids = [joint for joint in range(robot.model.njnt)
                 if int(robot.model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_HINGE)]
    if robot.model.nq != 36 or len(joint_ids) != 29:
        raise ValueError("Expected a free-base G1 with exactly 29 hinge joints")
    if [int(robot.model.jnt_qposadr[joint]) for joint in joint_ids] != list(range(7, 36)):
        raise ValueError("Unexpected G1 joint address layout")
    joint_names = np.array([mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                            for joint in joint_ids], dtype=str)
    human = SmplxSurfaceHuman(prepared, robot.height())
    ids = prepared["binding_joint_ids"].astype(np.int64)
    rotations = human.data.xmat[ids].reshape(-1, 3, 3)
    canonical = prepared["canonical_points"] * human.scale
    local_points = np.einsum("nji,nj->ni", rotations, canonical - human.data.xpos[ids])
    local_normals = np.einsum("nji,nj->ni", rotations, prepared["canonical_normals"])
    cloud = SurfacePointCloud(canonical, prepared["canonical_normals"], ids,
                              np.full(len(ids), -1, dtype=np.int64), local_points, local_normals,
                              prepared["segment"].astype(np.int64), SEGMENTS)
    robot_cloud = sample_model_surface(robot.model, robot.data, len(ids), geom_ids=robot.surface_geoms(),
                                       oversample=float(cfg.sampling["oversample"]),
                                       cull_margin=float(cfg.sampling["cull_margin"]),
                                       seed=int(cfg.sampling["seed"]), segment_names=SEGMENTS)
    runtime = environment_packages(("numpy", "scipy", "mujoco", "mink", "torch", "trimesh"))
    identity = canonical_setup_identity(
        prepared, scale=human.scale, robot_geometry=geometry, robot_xml_sha256=robot_hash,
        robot_config=dict(cfg.robot), sampling_config=dict(cfg.sampling),
        correspondence_config=dict(cfg.correspondence), robot_cloud=robot_cloud.to_dict(),
        epochs=epochs, device=device, runtime=runtime, umr_commit=commit)

    def verify_inputs():
        if (sha256(robot_xml) != robot_hash or sha256(source) != source_hash
                or sha256(Path(__file__)) != adapter_hash):
            raise ValueError("Immutable source, robot, or adapter changed during retargeting")
        after = model_geometry_fingerprint(mujoco.MjModel.from_xml_path(str(robot_xml)))
        if after != geometry or verify_umr_checkout(umr_root) != commit:
            raise ValueError("Robot loaded geometry or pinned UMR changed during retargeting")

    def build_setup(directory, stamp):
        new_setup = SetupLayout(directory).ensure()
        # Stage I reads these canonical arrays and robot_xml only. No fake
        # human.xml/BVH is introduced; Stage 0/II are the true-mesh adapters.
        np.savez(new_setup.bodies, stamp=stamp, robot_xml=str(robot_xml),
                 **cloud.to_dict("human_"), **robot_cloud.to_dict("robot_"))
        learn_correspondence(cfg, new_setup, epochs=epochs, device=device)

    output.mkdir(parents=True, exist_ok=False)
    if setup_cache is None:
        setup = SetupLayout(output / "setup").ensure()
        build_setup(setup.root, canonical_setup_key(identity))
        verify_inputs()
        cache_evidence = {"enabled": False, "hit": False, "entry": str(setup.root),
                          "key_sha256": canonical_setup_key(identity)}
    else:
        setup_directory, cache_evidence = obtain_cached_setup(
            setup_cache, identity, build_setup, verify_inputs,
            provenance={"adapter_sha256": adapter_hash, "prepared_source_sha256": source_hash,
                        "source_sha256": prepared["metadata"]["source_sha256"]})
        setup = SetupLayout(setup_directory)
        print(f"[canonical setup] {'HIT' if cache_evidence['hit'] else 'MISS'} "
              f"{cache_evidence['key_sha256']}", flush=True)
    with np.load(setup.correspondence, allow_pickle=True) as corr:
        binding = LinkBinding.from_dict(corr, "bind_")
        segment = np.array(corr["inherited_segment"])
    retargeter = retargeter_class(UMRRetargeter)(
        robot, human, human_body_ids=ids, human_local_pos=local_points, human_local_normal=local_normals,
        robot_body_ids=binding.body_ids, robot_local_pos=binding.local_pos,
        robot_local_normal=binding.local_normal, segment=segment, segment_names=SEGMENTS,
        **{key: cfg.retarget[key] for key in (
            "n_selected", "point_selection", "tpose_offset", "iterations", "dt", "damping", "solver",
            "trust_region", "trust_region_radius", "floor_height", "floor_band", "floor_margin",
            "contact_threshold", "contact_weight", "posture_cost", "self_collision")})
    result = retargeter.run(np.arange(len(prepared["times"])), float(prepared["fps"]))
    if not np.isfinite(result.qpos).all() or not np.allclose(np.linalg.norm(result.qpos[:, 3:7], axis=1), 1., atol=1e-5):
        raise ValueError("Retarget output is nonfinite or has invalid root quaternions")
    receipt = {"schema": "bfm.umr_smplx_trial/1", "experimental": True, "promoted_to_training": False,
               "source": prepared["metadata"], "prepared_source_sha256": source_hash,
               "umr_commit": commit, "upstream_status": "unofficial_reimplementation",
               "robot_xml": str(robot_xml), "robot_xml_sha256": robot_hash,
               "loaded_robot_geometry": geometry,
               "adapter_sha256": adapter_hash, "config": cfg,
               "canonical_setup_algorithm": CANONICAL_SETUP_ALGORITHM,
               "setup_cache": cache_evidence,
               "epochs": epochs, "device": device, "scale": human.scale,
               "ground_offset": human.ground_offset, "ground_rule": "constant_negative_1pct_actual_source_foot_vertex_min",
               "frames": len(result.qpos), "fps": result.fps, "point_error_mean_m": float(result.point_error.mean()),
               "solve_failures": result.solve_failures, "timings": result.timings,
               "tpose_offset": 0., "material_surface_transport": True,
               "root_heading_convention": "yaw_of_smplx_pelvis_local_positive_z_in_world",
               "environment": environment_packages(("numpy", "scipy", "mujoco", "mink", "torch",
                                                      "qpsolvers", "clarabel", "trimesh")),
               "not_yet_verified": ["physical_tracking", "joint_velocity_limits", "all_eight_masks", "training_improvement"]}
    verify_inputs()
    if setup_cache is not None:
        after_cache = validate_setup_cache_entry(setup.root, identity)
        if any(cache_evidence[key] != value for key, value in after_cache.items()):
            raise ValueError("Canonical setup cache changed during Stage II")
        cache_evidence["rechecked_after_retarget"] = True
    receipt["protected_inputs_rechecked"] = True
    receipt["scalebfm_output"] = export_scalebfm_motion(output / "scalebfm_motion.pkl",
                                                       result.qpos, result.fps, joint_names)
    save_npz_exclusive(output / "motion.npz", qpos=result.qpos, fps=result.fps, dof_names=joint_names,
                       frame_indices=result.frame_indices, point_error=result.point_error,
                       normal_error=result.normal_error, contact_count=result.contact_count,
                       floor_rows=result.floor_rows, metadata_json=json.dumps(receipt, sort_keys=True))
    save_motion_pkl(output / "motion.pkl", robot.model, result.qpos, result.fps,
                    extra={"source_file": prepared["metadata"]["source_file"], "source_kind": "smplx_mesh",
                           "source_sha256": prepared["metadata"]["source_sha256"], "experimental": True})
    with (output / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Run in existing SMPL-X environment")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--body-model", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--start", type=float, default=0.)
    prepare.add_argument("--duration", type=float, default=3.)
    prepare.add_argument("--fps", type=float, default=50.)
    prepare.add_argument("--points", type=int, default=4096)
    prepare.add_argument("--chunk-size", type=int, default=16)
    prepare.add_argument("--seed", type=int, default=0)
    retarget = commands.add_parser("retarget", help="Run in isolated pinned UMR environment")
    retarget.add_argument("--source", type=Path, required=True)
    retarget.add_argument("--umr-root", type=Path, required=True)
    retarget.add_argument("--output", type=Path, required=True)
    retarget.add_argument("--robot-xml", type=Path, required=True)
    retarget.add_argument("--epochs", type=int, default=2500)
    retarget.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    retarget.add_argument("--iterations", type=int, default=6)
    retarget.add_argument("--setup-cache", type=Path,
                          help="Optional immutable canonical Stage I cache; dedicated directory under local/")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    result = prepare_source(**args) if command == "prepare" else retarget_source(**args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
