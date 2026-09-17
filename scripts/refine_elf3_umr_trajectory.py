#!/usr/bin/env python3
"""Offline model-based refinement of an audited UMR trajectory, not a policy.

Optimize all 31 joint trajectories together with the exact original mesh model.
The UMR root and clock stay unchanged. A soft signed-distance loss competes with
learned-correspondence surface fidelity and temporal correction priors; no frame
is frozen/rejected, no time stretching or output clipping is performed. Results
must pass independent motion audits and are never automatically promoted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from elf3_umr_asset import digest, write_json
from elf3_source_reuse import load_replay_source, require_same_canonical
from run_elf3_umr_trial import UMR, elf3_mesh_sole_points, motion_audit
from umr_smplx_source import model_geometry_fingerprint, verify_umr_checkout


def correction_prior(delta, velocity_weight=10., acceleration_weight=50.):
    """Value and exact gradient of temporal *correction*, not motion smoothing."""
    if delta.ndim != 2 or len(delta) < 3 or not np.isfinite(delta).all():
        raise ValueError("Finite trajectory with at least three frames required")
    value, grad = float(np.sum(delta * delta)), 2. * delta
    velocity = np.diff(delta, axis=0)
    value += velocity_weight * float(np.sum(velocity * velocity))
    grad[:-1] -= 2. * velocity_weight * velocity
    grad[1:] += 2. * velocity_weight * velocity
    acceleration = np.diff(delta, n=2, axis=0)
    value += acceleration_weight * float(np.sum(acceleration * acceleration))
    grad[:-2] += 2. * acceleration_weight * acceleration
    grad[1:-1] -= 4. * acceleration_weight * acceleration
    grad[2:] += 2. * acceleration_weight * acceleration
    return value / len(delta), grad / len(delta)


class BoundSurface:
    """World-space FK and Jacobians for a fixed local learned surface binding."""
    def __init__(self, model, body_ids, local_positions):
        self.model = model
        self.body_ids = np.asarray(body_ids, dtype=int)
        self.local = np.asarray(local_positions, dtype=float)
        self.unique, self.inverse = np.unique(self.body_ids, return_inverse=True)

    def evaluate(self, data, jacobian=True):
        import mujoco
        offset = np.einsum("nij,nj->ni", data.xmat[self.body_ids].reshape(-1, 3, 3), self.local)
        points = data.xpos[self.body_ids] + offset
        if not jacobian:
            return points, None
        jp = np.zeros((len(self.unique), 3, self.model.nv))
        jr = np.zeros_like(jp)
        for i, body in enumerate(self.unique):
            mujoco.mj_jacBody(self.model, data, jp[i], jr[i], int(body))
        # (angular Jacobian column) cross (rotated local displacement).
        jac = jp[self.inverse] + np.cross(jr[self.inverse].transpose(0, 2, 1),
                                         offset[:, None, :]).transpose(0, 2, 1)
        return points, jac


class TrajectoryObjective:
    def __init__(self, model, qpos, surface, pairs, *, collision_weight=10000.):
        import mujoco
        self.model, self.base, self.surface = model, qpos.copy(), surface
        self.data = mujoco.MjData(model)
        self.pairs = list(pairs)
        self.weight = collision_weight
        self.address = np.array([model.joint(j).qposadr[0] for j in range(1, model.njnt)])
        self.dof = np.array([model.joint(j).dofadr[0] for j in range(1, model.njnt)])
        feet, points, _ = elf3_mesh_sole_points(model)
        self.feet = BoundSurface(model, feet, points)
        self.target = []
        for q in qpos:
            self.data.qpos[:] = q
            mujoco.mj_forward(model, self.data)
            self.target.append(surface.evaluate(self.data, False)[0])
        self.target = np.asarray(self.target)
        self.calls = 0
        self.latest = {}

    def joint_prior(self, theta):
        return correction_prior(theta - self.base[:, self.address])

    def __call__(self, flattened):
        import mujoco
        from mink.limits.collision_avoidance_limit import compute_contact_normal_jacobian
        theta = np.asarray(flattened).reshape(len(self.base), len(self.address))
        loss, grad = self.joint_prior(theta)
        terms = {"correction": loss, "surface": 0., "collision": 0., "floor": 0.}
        fromto, normal = np.empty(6), np.empty(3)
        jac1, jac2 = np.empty((3, self.model.nv)), np.empty((3, self.model.nv))
        maximum_penetration = 0.
        for frame, angles in enumerate(theta):
            self.data.qpos[:] = self.base[frame]
            self.data.qpos[self.address] = angles
            mujoco.mj_forward(self.model, self.data)
            points, jac = self.surface.evaluate(self.data)
            error = points - self.target[frame]
            weight = 100. / (len(theta) * len(points))
            terms["surface"] += weight * float(np.sum(error * error))
            grad[frame] += 2. * weight * np.einsum("ni,nij->j", error, jac[:, :, self.dof])
            # Same enabled collider pairs as UMR; no invented geometry or gaps.
            for a, b in self.pairs:
                distance = mujoco.mj_geomDistance(self.model, self.data, a, b, .01, fromto)
                if distance >= 0.:
                    continue
                maximum_penetration = max(maximum_penetration, -distance)
                derivative = -compute_contact_normal_jacobian(
                    self.model, self.data, a, b, fromto, normal, jac1, jac2)
                weight = self.weight / len(theta)
                terms["collision"] += weight * distance * distance
                grad[frame] += 2. * weight * distance * derivative[self.dof]
            points, jac = self.feet.evaluate(self.data)
            error = np.minimum(points[:, 2] - .002, 0.)
            weight = self.weight / (len(theta) * len(points))
            terms["floor"] += weight * float(error @ error)
            grad[frame] += 2. * weight * error @ jac[:, 2, self.dof]
        self.calls += 1
        self.latest = {**terms, "max_penetration_m": float(maximum_penetration), "calls": self.calls}
        return sum(terms.values()), grad.ravel()


def run(args):
    import mujoco
    import mink
    from scipy.optimize import minimize
    from umr.retarget.binding import LinkBinding
    from umr.retarget.pipeline import select_correspondence_points
    from umr.bodies.surface import SurfacePointCloud
    from umr_smplx_source import SEGMENTS, SmplxSurfaceHuman

    verify_umr_checkout(UMR)
    source_run, stage1, output = (p.resolve() for p in (args.source_run, args.stage1, args.output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inputs = json.loads((source_run / "inputs.json").read_text())
    parent = json.loads((source_run / "receipt.json").read_text())
    stage1_inputs = json.loads((stage1 / "inputs.json").read_text())
    if (digest(source_run / "motion.npz") != parent["motion_sha256"]
            or digest(inputs["robot_xml"]) != inputs["robot_xml_sha256"]
            or digest(inputs["source"]) != inputs["source_sha256"]
            or digest(stage1_inputs["source"]) != stage1_inputs["source_sha256"]
            or inputs["robot_geometry"] != stage1_inputs["robot_geometry"]):
        raise ValueError("Source motion, source mesh or Stage I does not match")
    protected = {str(p): digest(p) for p in (source_run / "motion.npz", source_run / "inputs.json",
                 source_run / "receipt.json", stage1 / "setup/correspondence.npz",
                 stage1 / "setup/bodies.npz", Path(__file__), args.urdf.resolve(),
                 Path(inputs["source"]), Path(stage1_inputs["source"]),
                 Path(__file__).with_name("elf3_source_reuse.py"),
                 Path(__file__).with_name("run_elf3_umr_trial.py"))}
    model = mujoco.MjModel.from_xml_path(inputs["robot_xml"])
    if model_geometry_fingerprint(model) != inputs["robot_geometry"]:
        raise ValueError("Loaded robot geometry changed")
    with np.load(source_run / "motion.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    qpos, samples, fps = arrays["qpos"], arrays["source_surface_indices"], float(arrays["fps"])
    if model.nq != 38 or not 3 <= len(qpos) <= 1501 or fps != 50.:
        raise ValueError("Bounded full 31-DoF ELF3 trajectory required")
    before = motion_audit(model, qpos, fps, args.urdf)
    with np.load(stage1 / "setup/bodies.npz", allow_pickle=True) as z:
        cloud = SurfacePointCloud.from_dict(z, "human_")
    with np.load(stage1 / "setup/correspondence.npz", allow_pickle=True) as z:
        binding, segment = LinkBinding.from_dict(z), z["inherited_segment"].copy()
    # Exactly the same selected correspondences as the UMR solver, not cherry-picked hands.
    prepared = load_replay_source(Path(inputs["source"]))
    canonical_identity = require_same_canonical(
        load_replay_source(Path(stage1_inputs["source"])), prepared)
    if (not np.array_equal(arrays["times"], prepared["times"])
            or len(samples) != len(cloud.points) or samples.dtype.kind not in "iu"
            or np.any(samples >= len(prepared["canonical_points"])) or np.any(samples < 0)
            or len(np.unique(samples)) != len(samples)):
        raise ValueError("Source clock or sampled correspondence identity changed")
    from umr.bodies.robot import RobotBody, RobotSpec
    from umr.bodies.surface import transport_points
    human = SmplxSurfaceHuman(prepared, RobotBody(inputs["robot_xml"], RobotSpec.from_config(inputs["config"]["robot"])).height())
    canonical_points, _ = transport_points(human.data, cloud.body_ids, cloud.local_pos, cloud.local_normal)
    # Repeat Stage II's exact canonical transport and FPS, including its arithmetic.
    selected = select_correspondence_points(segment, SEGMENTS, inputs["config"]["retarget"]["n_selected"],
                                            points=canonical_points, method=inputs["config"]["retarget"]["point_selection"])
    surface = BoundSurface(model, binding.body_ids[selected], binding.local_pos[selected])
    geoms = mink.get_subtree_geom_ids(model, 1)
    pairs = mink.CollisionAvoidanceLimit(model, [(geoms, geoms)]).geom_id_pairs
    objective = TrajectoryObjective(model, qpos, surface, pairs, collision_weight=args.collision_weight)
    x0 = qpos[:, objective.address].ravel()
    initial, _ = objective(x0)
    history = [{"iteration": 0, "loss": initial, **objective.latest}]
    output.mkdir(parents=True)
    start = time.monotonic()

    def callback(x):
        del x
        history.append({"iteration": len(history), **objective.latest})
        if len(history) % 10 == 0:
            print(json.dumps(history[-1]), flush=True)

    ranges = model.jnt_range[1:]
    result = minimize(objective, x0, jac=True, method="L-BFGS-B",
                      bounds=list(zip(np.tile(ranges[:, 0], len(qpos)), np.tile(ranges[:, 1], len(qpos)))),
                      callback=callback, options={"maxiter": args.max_iterations, "ftol": 1e-11,
                                                  "gtol": 1e-7, "maxls": 30, "maxcor": 12})
    refined = qpos.copy()
    refined[:, objective.address] = result.x.reshape(len(qpos), -1)
    after = motion_audit(model, refined, fps, args.urdf)
    # Independent same-raw-human metric: compensated target errors are not comparable.
    raw_errors, original_raw_errors, displacements = [], [], []
    for frame, q in enumerate(refined):
        objective.data.qpos[:] = q
        mujoco.mj_forward(model, objective.data)
        points, _ = surface.evaluate(objective.data, False)
        raw, _ = human.targets(frame)
        raw_errors.append(np.linalg.norm(points - raw[samples][selected], axis=1).mean())
        original_raw_errors.append(np.linalg.norm(objective.target[frame] - raw[samples][selected], axis=1).mean())
        displacements.extend(np.linalg.norm(points - objective.target[frame], axis=1).tolist())
    arrays["qpos"] = refined
    np.savez_compressed(output / "motion.npz", **arrays)
    for p, h in protected.items():
        if digest(p) != h:
            raise ValueError("Protected input changed during refinement")
    if model_geometry_fingerprint(mujoco.MjModel.from_xml_path(inputs["robot_xml"])) != inputs["robot_geometry"]:
        raise ValueError("On-disk robot geometry changed during refinement")
    write_json(output / "inputs.json", inputs)
    write_json(output / "optimizer_history.json", history)
    input_failures = parent.get("solve_failures", parent.get("stage2_solver_failures"))
    input_warmup_failures = parent.get("warmup_failures", parent.get("warmup_solver_failures"))
    checks = {"solver_converged": bool(result.success),
              "input_umr_no_solver_failures": input_failures == 0 and input_warmup_failures == 0,
              "raw_human_error_not_worse_by_5mm": bool(np.mean(raw_errors) <= np.mean(original_raw_errors) + .005),
              "root_and_clock_unchanged": bool(np.array_equal(refined[:, :7], qpos[:, :7])),
              "no_original_collision_over_5mm": after["original_collision_self_penetration_over_5mm_frames"] == 0,
              "no_foot_below_minus_5mm": after["foot_below_minus_5mm_frames"] == 0,
              "no_angle_violation": after["joint_limit_violation_frames"] == 0,
              "no_urdf_velocity_violation": after["joint_speed_over_urdf_limit_intervals"] == 0,
              "surface_change_p95_under_20mm": bool(np.percentile(displacements, 95) <= .020)}
    receipt = {"schema": "bfm.elf3_trajectory_refinement/1", "source_run": str(source_run),
               "protected_inputs": protected, "method": "L-BFGS-B_exact_FK_and_mesh_signed_distance",
               "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
               "iterations": int(result.nit), "evaluations": int(result.nfev),
               "initial_loss": initial, "final_loss": float(result.fun), "seconds": time.monotonic() - start,
               "collision_weight": args.collision_weight, "before": before, "audit": after,
               "human_canonical_identity": canonical_identity,
               "root_and_clock_policy": "unchanged_UMR_root_and_original_50Hz_timestamps",
               "raw_human_surface_error_mean_m": float(np.mean(raw_errors)),
               "input_raw_human_surface_error_mean_m": float(np.mean(original_raw_errors)),
               "surface_change_mean_m": float(np.mean(displacements)),
               "surface_change_p95_m": float(np.percentile(displacements, 95)),
               "surface_change_max_m": float(np.max(displacements)),
               "joint_correction_max_rad": float(np.abs(refined[:, 7:] - qpos[:, 7:]).max()),
               "kinematic_checks": checks, "kinematic_candidate_pass": all(checks.values()),
               "motion_sha256": digest(output / "motion.npz"),
               "new_neural_training": False, "policy_inference": False, "promoted_to_training": False}
    # This is explicitly a geometry-based trajectory refinement, not posthoc
    # filtering or a claim that a new behavior policy has been trained.
    write_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--stage1", type=Path, required=True)
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-iterations", type=int, default=250)
    p.add_argument("--collision-weight", type=float, default=10000.)
    args = p.parse_args()
    if not 10 <= args.max_iterations <= 1000 or not 100 <= args.collision_weight <= 1000000:
        raise ValueError("Explicit bounded optimizer budget required")
    os.environ.setdefault("MUJOCO_GL", "glfw")
    sys.path.insert(0, str(UMR))
    run(args)
