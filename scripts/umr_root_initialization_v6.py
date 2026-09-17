"""Explicit t0-only Stage II root-yaw recovery; frozen UMR is never patched."""
from __future__ import annotations

import numpy as np

from scripts import umr_heading_source_v6 as source_v6

SCHEMA = "bfm.umr_root_initialization/6"


def root_reference(rotation, metadata):
    """Use the stored source-world pelvis axes, not the pre-alignment raw yaw.

    The dtype/order of the ordinary branch test matches frozen core. Metadata
    proof is required only for the formerly rejected branch; callers must use
    the strict source loader before constructing the solver.
    """
    from scipy.spatial.transform import Rotation
    matrix = np.asarray(rotation)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=2e-5, rtol=0)
            or not np.isclose(np.linalg.det(matrix), 1., atol=2e-5, rtol=0)):
        raise ValueError("Require one proper stored t0 pelvis rotation")
    forward, left = matrix[:, 2], matrix[:, 0]
    fallback = bool(np.linalg.norm(forward[:2]) < .1)
    if fallback:
        proof = metadata.get("heading_reference", {})
        heading, rebuilt = source_v6.heading_reference(proof.get("first_sample_root_rotvec"))
        if proof != rebuilt or not rebuilt["fallback_used"]:
            raise ValueError("Degenerate root requires the exact v6 t0 heading proof")
        if not np.array_equal(heading, np.asarray(metadata.get("posed_heading_rotation"))):
            raise ValueError("Source global heading differs from its t0 proof")
        raw = Rotation.from_rotvec(rebuilt["first_sample_root_rotvec"]).as_matrix()
        if not np.allclose(matrix, heading @ raw, atol=1e-6, rtol=0):
            raise ValueError("Observed source-world root differs from its v6 proof")
        if np.linalg.norm(left[:2]) < np.sqrt(.99) - 2e-5:
            raise ValueError("Stored left axis violates the orthogonal fallback bound")
        reference = np.array([left[1], -left[0]], dtype=matrix.dtype)
    else:
        reference = forward[:2]
    return {
        "schema": SCHEMA, "frame_index": 0, "call_count": 1,
        "branch": "t0_left_cross_world_up" if fallback else "legacy_super",
        "fallback_used": fallback, "super_delegated": not fallback,
        "yaw_rad": float(np.arctan2(reference[1], reference[0])),
        "observed_forward_world": forward.tolist(), "observed_left_world": left.tolist(),
        "reference_world_xy": reference.tolist(),
        "source_heading_branch": metadata.get("heading_reference", {}).get("branch", "legacy_t0_forward_projection"),
    }


def initialized_retargeter_class(base):
    """Compose ABOVE core's SMPL-X class and BELOW frozen v3 controls.

    Ordinary initialization delegates directly to super with no extra human
    set_frame or robot writes. Fallback changes only the rejected yaw query.
    Root Z and all articulation values still come from the original seed.
    """
    class InitializedRetargeter(base):
        def initialize_root(self, frame):
            from scipy.spatial.transform import Rotation
            if (type(frame) is not int or frame != 0
                    or getattr(self, "root_initialization", None) is not None):
                raise ValueError("v6 permits exactly one initialization at source frame zero")
            pelvis = int(self.human.body_ids[0])
            stored = self.human.source["joint_rotations"][0, pelvis]
            trace = root_reference(stored, self.human.source["metadata"])
            self.root_initialization = trace
            seed = self.robot.model.key_qpos[0].copy() if self.robot.model.nkey else self.robot.q.copy()
            if trace["fallback_used"]:
                self.human.set_frame(frame)
                observed = self.human.data.xmat[pelvis].reshape(3, 3)
                if not np.array_equal(observed, stored):
                    raise ValueError("Actual human root differs from the strictly loaded source")
                q = seed.copy()
                q[:2] = self.human.data.xpos[pelvis, :2]
                q[3:7] = Rotation.from_euler("z", trace["yaw_rad"]).as_quat(scalar_first=True)
                self.robot.set_qpos(q)
            else:
                super().initialize_root(frame)
            actual = self.robot.q.copy()
            # Check the actual target provider stayed in the original lying/
            # crawling pose. No source array or target is rotated/uprighted here.
            expected_positions = self.human.source["joint_positions"][0].astype(np.float64) * self.human.scale
            expected_positions[:, 2] += self.human.ground_offset
            unchanged = (np.array_equal(self.human.data.xpos, expected_positions)
                and np.array_equal(self.human.data.xmat, self.human.source["joint_rotations"][0].reshape(55, 9)))
            trace.update(qpos_seed=seed.tolist(), qpos_initialized=actual.tolist(),
                root_z_unchanged=bool(actual[2] == seed[2]),
                joints_unchanged=bool(np.array_equal(actual[7:], seed[7:])),
                human_targets_unchanged=bool(unchanged))
            if not all(trace[k] for k in ("root_z_unchanged", "joints_unchanged", "human_targets_unchanged")):
                raise ValueError("Initialization changed the seed Z, articulation or human targets")
    return InitializedRetargeter
