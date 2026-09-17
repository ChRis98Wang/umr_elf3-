"""Checked mesh distances for offline ELF3 optimization (no model mutation).

MuJoCo 3.9 can return exact zero with noncoincident witnesses. Never use that
inconsistent result as a contact gradient. The fallback below only accepts a
positive distance after matching convex-combination and separating-plane
bounds. It does not invent penetration depths or alter mesh geometry.
"""
from itertools import combinations

import numpy as np


def closest_simplex(vertices):
    """Closest point of a <=4 point simplex, including all its boundary faces."""
    best = None
    for size in range(1, len(vertices)+1):
        for ids in combinations(range(len(vertices)), size):
            points = vertices[list(ids)]
            if size == 1:
                weights = np.ones(1)
            else:
                edges = (points[1:]-points[0]).T
                tail = np.linalg.lstsq(edges, -points[0], rcond=None)[0]
                weights = np.r_[1.-tail.sum(), tail]
            if weights.min() < -1e-12:
                continue
            weights = np.maximum(weights, 0.)
            weights /= weights.sum()
            point = weights @ points
            norm2 = point @ point
            if best is None or norm2 < best[0]:
                best = norm2, np.asarray(ids), weights, point
    return best[1:]


def certified_separated_distance(a, b, tolerance=1e-9, max_iterations=128):
    """GJK distance; return witnesses only when primal/dual bounds agree.

    The upper bound is a distance between convex combinations of mesh vertices.
    The lower bound is the gap along a separating plane over ALL mesh vertices.
    An intersecting or unresolved pair raises, rather than becoming free space.
    """
    direction = a.mean(axis=0)-b.mean(axis=0)
    if np.linalg.norm(direction) < 1e-12:
        direction = np.array([1., 0., 0.])
    va, vb = [], []
    for _ in range(max_iterations):
        pa, pb = a[np.argmin(a@direction)], b[np.argmax(b@direction)]
        va.append(pa); vb.append(pb)
        aa, bb = np.asarray(va), np.asarray(vb)
        ids, weights, closest = closest_simplex(aa-bb)
        upper = np.linalg.norm(closest)
        if upper < 1e-10:
            raise ValueError('Inconsistent mesh distance: fallback could not certify separation')
        normal = closest/upper
        lower = float(np.min(a@normal)-np.max(b@normal))
        if lower > 0. and upper-lower <= tolerance:
            return upper, np.r_[weights@aa[ids], weights@bb[ids]]
        keep = weights > 1e-12
        va, vb = list(aa[ids[keep]]), list(bb[ids[keep]])
        if len(va) >= 4:
            raise ValueError('Unresolved tetrahedral mesh distance')
        direction = closest
    raise ValueError('Mesh separation bounds did not converge')


class CheckedMeshDistance:
    def __init__(self, model):
        self.model = model
        self.recoveries = 0
        self.vertices = {}

    def world_vertices(self, data, geom):
        import mujoco
        if self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError('Inconsistent non-mesh distance is unsupported')
        mesh = int(self.model.geom_dataid[geom])
        if mesh not in self.vertices:
            start, count = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
            self.vertices[mesh] = self.model.mesh_vert[start:start+count].astype(float)
        return self.vertices[mesh]@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]

    def __call__(self, data, a, b, cutoff, fromto):
        import mujoco
        distance = mujoco.mj_geomDistance(self.model, data, a, b, cutoff, fromto)
        if distance == 0. and np.linalg.norm(fromto[3:]-fromto[:3]) > 1e-6:
            distance, witnesses = certified_separated_distance(
                self.world_vertices(data, a), self.world_vertices(data, b))
            fromto[:] = witnesses
            self.recoveries += 1
        return distance
