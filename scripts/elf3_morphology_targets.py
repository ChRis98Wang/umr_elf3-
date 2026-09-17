"""Morphology-aware targets retaining the actual SMPL-X surface deformation.

Stage I supplies the robot surface correspondence, not a hand-written joint
pose. Transport only its canonical human-to-robot offset in the associated
human bone frame; the moving human surface itself is never rigidified. This
is an explicit local Stage-II experiment, not an upstream UMR default.
"""
from __future__ import annotations

import numpy as np


class MorphologyTargets:
    def __init__(self, human_points, robot_points, canonical_rotations):
        hp, rp, rotation = (np.asarray(v, dtype=np.float64)
                            for v in (human_points, robot_points, canonical_rotations))
        if (hp.ndim != 2 or hp.shape[1] != 3 or not len(hp) or rp.shape != hp.shape
                or rotation.shape != (len(hp), 3, 3)
                or not all(np.isfinite(v).all() for v in (hp, rp, rotation))):
            raise ValueError("Canonical correspondence must contain finite paired XYZ and rotations")
        if (not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=2e-5)
                or not np.allclose(np.linalg.det(rotation), 1., atol=2e-5)):
            raise ValueError("Canonical bone rotations must belong to SO(3)")
        self.local_offset = np.einsum("nji,nj->ni", rotation, rp - hp)

    def positions(self, moving_surface, moving_rotations):
        points, rotation = (np.asarray(v, dtype=np.float64)
                            for v in (moving_surface, moving_rotations))
        if (points.shape != self.local_offset.shape
                or rotation.shape != (len(points), 3, 3)
                or not np.isfinite(points).all() or not np.isfinite(rotation).all()):
            raise ValueError("Moving surface/rotations do not match canonical correspondence")
        return points + np.einsum("nij,nj->ni", rotation, self.local_offset)
