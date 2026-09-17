#!/usr/bin/env python3
"""Model-fit cubic arm trajectories, retaining 31-DoF semantics and source clock.

Only 14 arm trajectories use a continuous cubic B-spline representation. Their
coefficients are optimized against the learned UMR surface motion and original
robot geometry. This is NOT output filtering, clipping or a learned policy.
Original URDF angle bounds remain unchanged; no hand-crafted pose path is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from diagnose_elf3_arm_motion import motion_metrics, traces, visual_arm_core_pairs
from elf3_umr_asset import digest, write_json
from elf3_source_reuse import load_replay_source, require_same_canonical
from refine_elf3_umr_trajectory import BoundSurface, TrajectoryObjective
from run_elf3_umr_trial import UMR, motion_audit
from umr_smplx_source import SEGMENTS, SmplxSurfaceHuman, model_geometry_fingerprint, verify_umr_checkout


def spline_basis(times, spacing):
    from scipy.interpolate import BSpline
    times = np.asarray(times, dtype=float)
    if len(times) < 4 or not np.allclose(np.diff(times), .02) or times[0] != 0. or not .04 <= spacing <= .3:
        raise ValueError("Actual complete 50Hz clock and explicit 40..300ms knot spacing required")
    knots = np.r_[np.repeat(times[0], 4), np.arange(spacing, times[-1] - 1e-9, spacing), np.repeat(times[-1], 4)]
    return BSpline.design_matrix(times, knots, 3).toarray(), knots


def arm_joint_prior(theta, base, arms):
    delta = theta - base
    count = len(theta)
    value, grad = .05 * np.sum(delta * delta) / count, .1 * delta / count
    acceleration = np.diff(theta[:, arms], n=2, axis=0)
    value += 5. * np.sum(acceleration * acceleration) / count
    grad[:-2, arms] += 10. * acceleration / count
    grad[1:-1, arms] -= 20. * acceleration / count
    grad[2:, arms] += 10. * acceleration / count
    return float(value), grad


class ArmObjective(TrajectoryObjective):
    def __init__(self, *args, basis, wrist_pose_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        names = [self.model.joint(j).name for j in range(1, self.model.njnt)]
        self.arms = np.array([i for i, name in enumerate(names) if any(part in name for part in ("shoulder", "elbow", "wrist"))])
        if len(self.arms) != 14:
            raise ValueError("Expected both complete 7-DoF ELF3 arms")
        self.basis = basis
        self.wrist_pose_loss = wrist_pose_loss
        if wrist_pose_loss:
            saved, _ = traces(self.model, self.base)
            self.wrist_target = saved
            self.wrist_bodies = np.array([self.model.body(side + "_wrist_z_link").id for side in ("l", "r")])
            self.tip_surface = BoundSurface(self.model, self.wrist_bodies, saved["tool_local_points"])

    def wrist_pose_terms(self, theta):
        """Balanced wrist task-space fidelity; SO(3) error avoids Euler wrapping.

        d ||log(R R_ref^T)||² / dq = 2 log(R R_ref^T)^T J_world.
        Fixed world reference per source frame; no hand-designed targets.
        """
        import mujoco
        from scipy.spatial.transform import Rotation
        loss, grad = 0., np.zeros_like(theta)
        if not self.wrist_pose_loss:
            return loss, grad
        position_weight, rotation_weight = 10. / (2 * len(theta)), .1 / (2 * len(theta))
        for frame, angles in enumerate(theta):
            self.data.qpos[:] = self.base[frame]
            self.data.qpos[self.address] = angles
            mujoco.mj_forward(self.model, self.data)
            positions, jac = self.tip_surface.evaluate(self.data)
            error = positions - self.wrist_target["tool_positions"][frame]
            loss += position_weight * np.sum(error * error)
            grad[frame] += 2 * position_weight * np.einsum("ni,nij->j", error, jac[:, :, self.dof])
            for i, body in enumerate(self.wrist_bodies):
                relative = self.data.xmat[body].reshape(3, 3) @ self.wrist_target["wrist_rotations"][frame, i].T
                phi = Rotation.from_matrix(relative).as_rotvec()
                jr = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, None, jr, int(body))
                loss += rotation_weight * (phi @ phi)
                grad[frame] += 2 * rotation_weight * phi @ jr[:, self.dof]
        return float(loss), grad

    def joint_prior(self, theta):
        return arm_joint_prior(theta, self.base[:, self.address], self.arms)

    def expand(self, coefficients):
        theta = self.base[:, self.address].copy()
        theta[:, self.arms] = self.basis @ coefficients.reshape(self.basis.shape[1], len(self.arms))
        return theta

    def coefficients_objective(self, coefficients):
        theta = self.expand(coefficients)
        value, grad = super().__call__(theta.ravel())
        grad = grad.reshape(theta.shape)
        pose_loss, pose_grad = self.wrist_pose_terms(theta)
        value, grad = value + pose_loss, grad + pose_grad
        self.latest["wrist_pose"] = pose_loss
        return value, (self.basis.T @ grad[:, self.arms]).ravel()


def run(args):
    import mujoco
    import mink
    from scipy.optimize import minimize
    from umr.bodies.robot import RobotBody, RobotSpec
    from umr.bodies.surface import SurfacePointCloud, transport_points
    from umr.retarget.binding import LinkBinding
    from umr.retarget.pipeline import select_correspondence_points

    verify_umr_checkout(UMR)
    folder, stage1, output = (p.resolve() for p in (args.source_run, args.stage1, args.output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inputs = json.loads((folder / "inputs.json").read_text())
    parent = json.loads((folder / "receipt.json").read_text())
    seed_inputs = json.loads((stage1 / "inputs.json").read_text())
    if (digest(folder / "motion.npz") != parent["motion_sha256"]
            or digest(inputs["robot_xml"]) != inputs["robot_xml_sha256"]
            or digest(inputs["source"]) != inputs["source_sha256"]
            or inputs["robot_geometry"] != seed_inputs["robot_geometry"]):
        raise ValueError("Motion/source/model identity changed")
    protected = {str(p): digest(p) for p in (folder / "motion.npz", folder / "inputs.json", folder / "receipt.json",
        stage1 / "setup/bodies.npz", stage1 / "setup/correspondence.npz", Path(__file__),
        Path(__file__).with_name("refine_elf3_umr_trajectory.py"), args.urdf.resolve())}
    robot = RobotBody(inputs["robot_xml"], RobotSpec.from_config(inputs["config"]["robot"]))
    model = robot.model
    if model_geometry_fingerprint(model) != inputs["robot_geometry"]:
        raise ValueError("Robot mesh changed")
    prepared = load_replay_source(inputs["source"], require_full=True)
    require_same_canonical(load_replay_source(seed_inputs["source"], require_full=True), prepared)
    human = SmplxSurfaceHuman(prepared, robot.height())
    with np.load(folder / "motion.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    qpos = arrays["qpos"]
    if not 4 <= len(qpos) <= 1501 or not np.array_equal(arrays["times"], prepared["times"]):
        raise ValueError("Complete bounded source clock required")
    with np.load(stage1 / "setup/bodies.npz", allow_pickle=True) as z:
        cloud = SurfacePointCloud.from_dict(z, "human_")
    with np.load(stage1 / "setup/correspondence.npz", allow_pickle=True) as z:
        binding, segment = LinkBinding.from_dict(z), z["inherited_segment"].copy()
    canonical, _ = transport_points(human.data, cloud.body_ids, cloud.local_pos, cloud.local_normal)
    selected = select_correspondence_points(segment, SEGMENTS, inputs["config"]["retarget"]["n_selected"],
                                            points=canonical, method=inputs["config"]["retarget"]["point_selection"])
    surface = BoundSurface(model, binding.body_ids[selected], binding.local_pos[selected])
    geoms = mink.get_subtree_geom_ids(model, 1)
    pairs = list(mink.CollisionAvoidanceLimit(model, [(geoms, geoms)]).geom_id_pairs)
    if args.visual_screen_loss:
        pairs += visual_arm_core_pairs(model)
    basis, knots = spline_basis(arrays["times"], args.knot_seconds)
    objective = ArmObjective(model, qpos, surface, pairs, basis=basis, collision_weight=10000.,
                             wrist_pose_loss=args.wrist_pose_loss)
    limits = model.jnt_range[1:][objective.arms]
    seed = np.linalg.lstsq(basis, qpos[:, objective.address[objective.arms]], rcond=None)[0]
    # Only initialize feasible control coefficients; final output is always B @ C.
    seed = np.maximum(np.minimum(seed, limits[:, 1]), limits[:, 0])
    if args.initial_spline is not None:
        initial_folder = args.initial_spline.resolve()
        previous = json.loads((initial_folder / "receipt.json").read_text())
        if (Path(previous["source_run"]) != folder or previous["knot_seconds"] != args.knot_seconds
                or previous["protected_inputs"].get(str(folder / "motion.npz")) != digest(folder / "motion.npz")
                or digest(initial_folder / "motion.npz") != previous["motion_sha256"]):
            raise ValueError("Warm start must fit the exact same original trajectory and spline clock")
        with np.load(initial_folder / "spline.npz", allow_pickle=False) as z:
            if not np.array_equal(z["knots"], knots) or not np.array_equal(z["arm_joint_indices"], objective.arms):
                raise ValueError("Warm-start spline basis/layout differs")
            seed = z["coefficients"].copy()
        if (seed.shape != (basis.shape[1], len(objective.arms)) or not np.isfinite(seed).all()
                or np.any(seed < limits[:, 0]) or np.any(seed > limits[:, 1])):
            raise ValueError("Warm-start coefficients violate original bounds")
        with np.load(initial_folder / "motion.npz", allow_pickle=False) as z:
            if not np.array_equal(objective.expand(seed.ravel()), z["qpos"][:, objective.address]):
                raise ValueError("Warm-start coefficients do not reconstruct hash-verified motion")
        protected.update({str(initial_folder / name): digest(initial_folder / name)
                          for name in ("receipt.json", "motion.npz", "spline.npz")})
    output.mkdir(parents=True)
    history = []
    start = time.monotonic()

    def callback(x):
        del x
        history.append({"iteration": len(history)+1, **objective.latest})
        if len(history) % 20 == 0:
            print(json.dumps(history[-1]), flush=True)

    initial = objective.coefficients_objective(seed.ravel())[0]
    result = minimize(objective.coefficients_objective, seed.ravel(), jac=True, method="L-BFGS-B",
        bounds=list(zip(np.tile(limits[:, 0], len(seed)), np.tile(limits[:, 1], len(seed)))),
        callback=callback, options={"maxiter": args.max_iterations, "ftol": 1e-10, "gtol": 1e-7, "maxls": 30, "maxcor": 12})
    output_q = qpos.copy()
    output_q[:, objective.address] = objective.expand(result.x)
    audit = motion_audit(model, output_q, 50., args.urdf)
    before_trace, before_mesh = traces(model, qpos)
    after_trace, after_mesh = traces(model, output_q)
    metrics = {}
    for i, side in enumerate(("left", "right")):
        metrics[side] = {"before": motion_metrics(before_trace["tool_positions"][:, i], before_trace["wrist_rotations"][:, i]),
                         "after": motion_metrics(after_trace["tool_positions"][:, i], after_trace["wrist_rotations"][:, i])}
    raw_error, displacement, tips_displacement = [], [], []
    samples = arrays["source_surface_indices"]
    for frame, q in enumerate(output_q):
        objective.data.qpos[:] = q
        mujoco.mj_forward(model, objective.data)
        points, _ = surface.evaluate(objective.data, False)
        raw, _ = human.targets(frame)
        raw_error.append(np.linalg.norm(points - raw[samples][selected], axis=1).mean())
        displacement.extend(np.linalg.norm(points - objective.target[frame], axis=1).tolist())
    tips_displacement = np.linalg.norm(after_trace["tool_positions"] - before_trace["tool_positions"], axis=2)
    # Remove stale per-frame Stage-II diagnostics rather than relabeling them as optimized measurements.
    for key in ("point_error", "solve_failures"):
        arrays.pop(key, None)
    arrays["qpos"] = output_q
    np.savez_compressed(output / "motion.npz", **arrays)
    np.savez_compressed(output / "spline.npz", knots=knots, coefficients=result.x.reshape(seed.shape),
                        arm_joint_indices=objective.arms, degree=3)
    for p, h in protected.items():
        if digest(p) != h:
            raise ValueError("Protected source/code changed")
    write_json(output / "inputs.json", inputs)
    write_json(output / "optimizer_history.json", history)
    receipt = {"schema": "bfm.elf3_arm_spline_refinement/1", "source_run": str(folder),
               "protected_inputs": protected, "knot_seconds": args.knot_seconds,
               "visual_convex_hull_soft_loss": args.visual_screen_loss,
               "wrist_pose_loss": args.wrist_pose_loss,
               "initial_spline": str(args.initial_spline.resolve()) if args.initial_spline else None,
               "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
               "iterations": int(result.nit), "seconds": time.monotonic()-start,
               "initial_loss": initial, "final_loss": float(result.fun),
               "arm_metrics": metrics, "before_visual_screen": before_mesh, "after_visual_screen": after_mesh,
               "raw_human_surface_error_mean_m": float(np.mean(raw_error)), "audit": audit,
               "surface_change_p95_m": float(np.percentile(displacement, 95)),
               "tip_displacement_p95_m": float(np.percentile(tips_displacement, 95)),
               "tip_displacement_max_m": float(tips_displacement.max()),
               "root_and_non_arm_unchanged": bool(np.array_equal(
                   output_q[:, np.setdiff1d(np.arange(model.nq), objective.address[objective.arms])],
                   qpos[:, np.setdiff1d(np.arange(model.nq), objective.address[objective.arms])])),
               "motion_sha256": digest(output / "motion.npz"), "new_neural_training": False,
               "policy_inference": False, "promoted_to_training": False, "arm_review_complete": False}
    write_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--stage1", type=Path, required=True)
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--knot-seconds", type=float, default=.12)
    p.add_argument("--max-iterations", type=int, default=500)
    p.add_argument("--visual-screen-loss", action="store_true")
    p.add_argument("--wrist-pose-loss", action="store_true")
    p.add_argument("--initial-spline", type=Path)
    args = p.parse_args()
    if not 10 <= args.max_iterations <= 1000:
        raise ValueError("Bounded optimizer iterations required")
    sys.path.insert(0, str(UMR))
    run(args)
