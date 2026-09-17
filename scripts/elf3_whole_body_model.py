"""Continuous full-body FK model with an explicit nonsingular SO(3) chart.

37 coordinates: root translation, local exponential rotation, 31 hinge angles.
No quaternion averaging, output clipping, time warping or frozen-frame repair.
This is offline kinematic optimization, not an actor/policy or dynamics solver.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from diagnose_elf3_arm_motion import traces
from elf3_mesh_distance import CheckedMeshDistance
from refine_elf3_umr_trajectory import BoundSurface
from run_elf3_umr_trial import elf3_mesh_sole_points


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def right_jacobian(v):
    """SO(3) right Jacobian: Exp(v+dv) = Exp(v) Exp(Jr(v) dv)."""
    theta = np.linalg.norm(v)
    k = skew(v)
    if theta < 1e-5:
        return np.eye(3) - (.5-theta**2/24.)*k + (1./6.-theta**2/120.)*(k@k)
    return np.eye(3) - (1.-np.cos(theta))/theta**2*k + (theta-np.sin(theta))/theta**3*(k@k)


class FullBodyChart:
    def __init__(self, qpos):
        qpos = np.asarray(qpos)
        if qpos.ndim != 2 or qpos.shape[1] != 38 or not np.isfinite(qpos).all():
            raise ValueError("Finite full ELF3 qpos required")
        if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., atol=1e-5, rtol=0):
            raise ValueError("Unit wxyz root quaternions required")
        roots = Rotation.from_quat(qpos[:, 3:7], scalar_first=True)
        self.reference = roots[0]
        relative = (self.reference.inv() * roots).as_quat(scalar_first=True)
        if relative[0, 0] < 0:
            relative[0] *= -1
        for i in range(1, len(relative)):
            if relative[i] @ relative[i-1] < 0:
                relative[i] *= -1
        sine = np.linalg.norm(relative[:, 1:], axis=1)
        angle = 2*np.arctan2(sine, relative[:, 0])
        # A single chart cannot honestly represent arbitrarily many turns.
        # Reject here; never crop a source or force a pi-wrap into the fit.
        if np.any(angle >= 1.75*np.pi):
            raise ValueError("Root requires multiple rotation charts; not supported by this bounded pilot")
        phi = relative[:, 1:] * np.divide(angle, sine, out=np.full_like(angle, 2.), where=sine>1e-10)[:, None]
        self.original = np.c_[qpos[:, :3], phi, qpos[:, 7:]]

    def qpos(self, coordinates):
        coordinates = np.asarray(coordinates)
        orientation = self.reference * Rotation.from_rotvec(coordinates[:, 3:6])
        return np.c_[coordinates[:, :3], orientation.as_quat(scalar_first=True), coordinates[:, 6:]]

    def pullback(self, coordinates, velocity_gradient):
        gradient = velocity_gradient.copy()
        for i, phi in enumerate(coordinates[:, 3:6]):
            gradient[i, 3:6] = right_jacobian(phi).T @ velocity_gradient[i, 3:6]
        return gradient


def temporal_prior(coordinates, original):
    count = len(coordinates)
    correction_weights = np.r_[np.full(3, 50.), np.full(3, .1), np.full(31, .05)]
    acceleration_weights = np.r_[np.full(3, 100.), np.full(3, 5.), np.full(31, 5.)]
    delta = coordinates - original
    loss = np.sum(correction_weights * delta**2) / count
    grad = 2*correction_weights*delta / count
    acceleration = np.diff(coordinates, n=2, axis=0)
    loss += np.sum(acceleration_weights * acceleration**2) / count
    da = 2*acceleration_weights*acceleration / count
    grad[:-2] += da
    grad[1:-1] -= 2*da
    grad[2:] += da
    return float(loss), grad


class WholeBodyObjective:
    def __init__(self, model, qpos, surface, original_pairs, visual_pairs, basis, clearance=.001):
        import mujoco
        if not 0 <= clearance <= .003:
            raise ValueError("Explicit 0..3mm geometric soft-clearance experiment required")
        self.model, self.base, self.surface, self.basis = model, qpos.copy(), surface, basis
        self.chart = FullBodyChart(qpos)
        self.data = mujoco.MjData(model)
        self.distance = CheckedMeshDistance(model)
        self.visual_pairs = set(visual_pairs)
        self.pairs = list(dict.fromkeys([*original_pairs, *visual_pairs]))
        self.clearance = clearance
        bodies, points, _ = elf3_mesh_sole_points(model)
        self.feet = BoundSurface(model, bodies, points)
        saved, _ = traces(model, qpos)
        self.wrist_bodies = [model.body(s+'_wrist_z_link').id for s in ('l', 'r')]
        self.tips = BoundSurface(model, self.wrist_bodies, saved['tool_local_points'])
        self.wrist_rotations = saved['wrist_rotations']
        self.target, self.foot_target, self.tip_target = [], [], []
        for q in qpos:
            self.data.qpos[:] = q
            mujoco.mj_forward(model, self.data)
            for field, bound in ((self.target, self.surface), (self.foot_target, self.feet), (self.tip_target, self.tips)):
                field.append(bound.evaluate(self.data, False)[0])
        self.target, self.foot_target, self.tip_target = map(np.asarray, (self.target, self.foot_target, self.tip_target))
        self.calls, self.latest = 0, {}

    def __call__(self, coefficients):
        import mujoco
        from mink.limits.collision_avoidance_limit import compute_contact_normal_jacobian
        coordinates = self.basis @ coefficients.reshape(self.basis.shape[1], 37)
        if np.any(np.linalg.norm(coordinates[:, 3:6], axis=1) >= 1.75*np.pi):
            raise ValueError("Optimizer left the validated root chart; no output emitted")
        qpos = self.chart.qpos(coordinates)
        prior, gradient = temporal_prior(coordinates, self.chart.original)
        terms = {"temporal_and_fidelity": prior, "surface": 0., "feet_fidelity": 0.,
                 "wrist_position": 0., "wrist_orientation": 0., "collision": 0., "floor": 0.}
        velocity_gradient = np.zeros_like(coordinates)
        fromto, normal = np.empty(6), np.empty(3)
        jac1, jac2 = np.empty((3, self.model.nv)), np.empty((3, self.model.nv))
        maximum_penetration = 0.
        for frame, q in enumerate(qpos):
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            for name, bound, target, weight in (("surface", self.surface, self.target[frame], 100.),
                 ("feet_fidelity", self.feet, self.foot_target[frame], 200.),
                 ("wrist_position", self.tips, self.tip_target[frame], 10.)):
                points, jac = bound.evaluate(self.data)
                error = points - target
                w = weight/(len(qpos)*len(points))
                terms[name] += w*np.sum(error**2)
                velocity_gradient[frame] += 2*w*np.einsum('ni,nij->j', error, jac)
                if name == 'feet_fidelity':
                    floor_error = np.minimum(points[:, 2]-.002, 0.)
                    w = 10000./(len(qpos)*len(points))
                    terms['floor'] += w*(floor_error@floor_error)
                    velocity_gradient[frame] += 2*w*floor_error@jac[:, 2, :]
            for i, body in enumerate(self.wrist_bodies):
                phi = Rotation.from_matrix(self.data.xmat[body].reshape(3, 3) @ self.wrist_rotations[frame, i].T).as_rotvec()
                jr = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, None, jr, body)
                w = .1/(2*len(qpos))
                terms['wrist_orientation'] += w*(phi@phi)
                velocity_gradient[frame] += 2*w*phi@jr
            for a, b in self.pairs:
                distance = self.distance(self.data, a, b, .01, fromto)
                margin = self.clearance if (a, b) in self.visual_pairs else 0.
                error = distance-margin
                if error >= 0:
                    continue
                derivative = compute_contact_normal_jacobian(self.model, self.data, a, b, fromto, normal, jac1, jac2)
                if distance < 0:
                    derivative = -derivative
                w = 10000./len(qpos)
                terms['collision'] += w*error**2
                velocity_gradient[frame] += 2*w*error*derivative
                maximum_penetration = max(maximum_penetration, -distance)
        gradient += self.chart.pullback(coordinates, velocity_gradient)
        self.calls += 1
        self.latest = {k:float(v) for k,v in terms.items()}
        self.latest.update(calls=self.calls, max_penetration_m=float(maximum_penetration))
        self.latest['certified_mesh_distance_recoveries'] = self.distance.recoveries
        return float(sum(terms.values())), (self.basis.T@gradient).ravel()
