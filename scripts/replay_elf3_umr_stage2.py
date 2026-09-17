#!/usr/bin/env python3
"""Controlled Stage-II comparison with a fixed learned ELF3 correspondence.

No neural re-training, no URDF changes, no posthoc qpos correction. The optional
contact repair formulates the existing signed-distance limit in Delta-q units,
activates it before the desired margin, and requests recovery inside the margin.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from elf3_umr_asset import digest, write_json
from elf3_morphology_targets import MorphologyTargets
from elf3_source_reuse import load_replay_source, require_same_canonical
from run_elf3_umr_trial import UMR, elf3_mesh_sole_points, motion_audit
from umr_smplx_source import SEGMENTS, SmplxSurfaceHuman, load_prepared_source, verify_umr_checkout, model_geometry_fingerprint


def foot_feasible_initial_pose(model, qpos, floor_height, margin):
    """Place ONLY the initial T-pose using actual foot mesh geometry.

    Human torso-height initialization can put longer robot legs below ground,
    making the first bounded IK step infeasible. This sets the initial condition
    before IK; it never translates saved frames or changes human targets/limits.
    """
    import mujoco
    from refine_elf3_umr_trajectory import BoundSurface
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    bodies, points, _ = elf3_mesh_sole_points(model)
    feet = BoundSurface(model, bodies, points)
    minimum = float(feet.evaluate(data, False)[0][:,2].min())
    shift = max(0., float(floor_height+margin-minimum))
    result = qpos.copy()
    result[2] += shift
    return result, {'initial_foot_min_z_m':minimum, 'initial_root_z_correction_m':shift,
                    'target_foot_min_z_m':float(floor_height+margin), 'saved_frame_translation':False}


def recovering_contact_class(base):
    class RecoveringContactLimit(base):
        def compute_qp_inequalities(self, configuration, dt):
            del dt  # Mink 1.1.1 build_ik optimizes Delta-q, not velocity.
            import mujoco
            from mink.limits import Constraint
            from mink.limits.collision_avoidance_limit import compute_contact_normal_jacobian
            G, h = [], []
            for a, b in self.geom_id_pairs:
                distance = mujoco.mj_geomDistance(self.model, configuration.data, a, b,
                                                  self.collision_detection_distance, self._fromto)
                if distance >= self.collision_detection_distance - 1e-12:
                    continue
                derivative = compute_contact_normal_jacobian(
                    self.model, configuration.data, a, b, self._fromto, self._normal, self._jac1, self._jac2)
                # signed-distance gradient reverses inside penetration.
                G.append((-1 if distance >= 0 else 1) * derivative.copy())
                h.append(self.gain * (distance - self.minimum_distance_from_collisions))
            return Constraint(G=np.asarray(G), h=np.asarray(h)) if G else Constraint()
    return RecoveringContactLimit


def run(args):
    import mujoco
    import mink
    from scipy.spatial.transform import Rotation
    from umr.bodies.robot import RobotBody, RobotSpec
    from umr.bodies.surface import SurfacePointCloud
    from umr.retarget.binding import LinkBinding
    from umr.retarget.pipeline import UMRRetargeter

    verify_umr_checkout(UMR)
    trial, output = args.trial.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = json.loads((trial / "inputs.json").read_text())
    original = json.loads((trial / "receipt.json").read_text())
    xml, source = Path(inputs["robot_xml"]), Path(inputs["source"])
    if digest(xml) != inputs["robot_xml_sha256"] or digest(source) != inputs["source_sha256"]:
        raise ValueError("Trial source or robot changed")
    if digest(trial / "motion.npz") != original["motion_sha256"]:
        raise ValueError("Original motion changed")
    with np.load(trial / "motion.npz", allow_pickle=False) as z:
        samples = z["source_surface_indices"]
        baseline_q = z["qpos"]
    protected = {str(trial / "setup" / n): digest(trial / "setup" / n)
                 for n in ("bodies.npz", "correspondence.npz")}
    protected[str(Path(__file__))] = digest(__file__)
    transport_script = Path(__file__).with_name("elf3_morphology_targets.py")
    protected[str(transport_script)] = digest(transport_script)
    reuse_script = Path(__file__).with_name("elf3_source_reuse.py")
    protected[str(reuse_script)] = digest(reuse_script)
    if args.initialization == 'foot-feasible':
        helper = Path(__file__).with_name('refine_elf3_umr_trajectory.py')
        protected[str(helper)] = digest(helper)
    with np.load(trial / "setup/bodies.npz", allow_pickle=True) as z:
        cloud = SurfacePointCloud.from_dict(z, "human_")
    with np.load(trial / "setup/correspondence.npz", allow_pickle=True) as z:
        binding = LinkBinding.from_dict(z)
        segment = z["inherited_segment"].copy()
    cfg = inputs["config"]
    robot = RobotBody(xml, RobotSpec.from_config(cfg["robot"]))
    if model_geometry_fingerprint(robot.model) != inputs["robot_geometry"]:
        raise ValueError("Loaded geometry changed")
    prepared = load_replay_source(source)
    reused_identity = None
    is_original_source = args.source is None or args.source.resolve() == source.resolve()
    if not is_original_source:
        candidate = load_replay_source(args.source.resolve(), require_full=True)
        reused_identity = require_same_canonical(prepared, candidate)
        from umr_smplx_source import environment_packages
        if environment_packages(("numpy", "scipy", "mujoco", "torch", "mink", "trimesh")) != original["environment"]:
            raise ValueError("Stage I runtime changed; cache reuse needs an explicit new validation")
        source = args.source.resolve()
        prepared = candidate
    protected[str(source)] = digest(source)
    inputs = {**inputs, "source": str(source), "source_sha256": digest(source),
              "source_metadata": prepared["metadata"]}
    human = SmplxSurfaceHuman(prepared, robot.height())
    from umr.bodies.surface import transport_points
    robot.set_tpose()
    robot_canonical, _ = transport_points(robot.data, binding.body_ids,
                                          binding.local_pos, binding.local_normal)
    transport = MorphologyTargets(prepared["canonical_points"][samples] * human.scale,
                                  robot_canonical,
                                  prepared["canonical_joint_rotations"][cloud.body_ids])

    class Replay(UMRRetargeter):
        rejected_steps = 0
        backtracks = 0
        initialization_report = None

        def human_targets(self, frame):
            p, n = self.human.targets(frame)
            if args.targets == "morphology":
                rotation = self.human.data.xmat[cloud.body_ids].reshape(-1, 3, 3)
                return transport.positions(p[samples], rotation), n[samples]
            return p[samples], n[samples]

        def initialize_root(self, frame):
            self.robot.set_tpose()
            self.human.set_tpose()
            offset = self.robot.data.body("torso_link").xpos - self.human.data.xpos[9]
            self.human.set_frame(frame)
            forward = self.human.data.xmat[0].reshape(3, 3)[:, 2]
            r = Rotation.from_euler("z", np.arctan2(forward[1], forward[0]))
            q = self.robot.model.key_qpos[0].copy()
            q[:3] = self.human.data.xpos[9] + r.apply(offset)
            q[3:7] = r.as_quat(scalar_first=True)
            if args.initialization == 'foot-feasible':
                q, self.initialization_report = foot_feasible_initial_pose(
                    self.robot.model, q, cfg['retarget']['floor_height'], cfg['retarget']['floor_margin'])
            self.robot.set_qpos(q)

        def solve_frame(self, frame, iterations=None):
            if not args.nonlinear_check:
                return super().solve_frame(frame, iterations)
            # A linearized contact inequality alone is not a guarantee after
            # integrating a finite rotation. Check the actual model geometry
            # during SQP line search, before accepting a proposed optimizer step.
            hp, hn = self.human_targets(frame)
            target, normal = hp[self.selected], hn[self.selected]
            self.pos_task.set_target(target)
            self.nrm_task.set_target(normal)
            count = self.contact_task.update_contacts(target)
            configuration = self.robot.configuration
            contact_limits = [v for v in self.limits if isinstance(v, mink.CollisionAvoidanceLimit)]
            pairs = contact_limits[0].geom_id_pairs

            def minimum_distance():
                return min(mujoco.mj_geomDistance(self.robot.model, configuration.data, a, b, .1, None)
                           for a, b in pairs)

            failures = 0
            for _ in range(iterations or self.iterations):
                old_q = configuration.q.copy()
                old_distance = minimum_distance()
                try:
                    velocity = mink.solve_ik(configuration, self.tasks, self.dt, self.solver,
                                             damping=self.damping, limits=self.limits)
                except mink.NoSolutionFound:
                    failures += 1
                    break
                accepted = False
                for backtrack in range(13):
                    q = old_q.copy()
                    mujoco.mj_integratePos(self.robot.model, q, velocity, self.dt * (0.5 ** backtrack))
                    configuration.update(q=q)
                    if minimum_distance() >= min(0., old_distance) - 1e-8:
                        self.backtracks += backtrack
                        accepted = True
                        break
                if not accepted:
                    configuration.update(q=old_q)
                    self.rejected_steps += 1
            self.cache.refresh(configuration, force=True)
            self.floor_cache.refresh(configuration, force=True)
            error = np.linalg.norm(self.cache.pos[self.task_idx] - target, axis=1)
            cosine = np.clip(np.einsum("ij,ij->i", self.cache.nrm[self.task_idx], normal), -1., 1.)
            return {"qpos": configuration.q.copy(), "point_error": float(error.mean()),
                    "normal_error": float(np.arccos(cosine).mean()), "contact_count": count,
                    "floor_rows": self.floor_limit.n_active, "failures": failures}

    keys = ("n_selected", "point_selection", "tpose_offset", "iterations", "dt", "damping", "solver",
            "trust_region", "trust_region_radius", "floor_height", "floor_band", "floor_margin",
            "contact_threshold", "contact_weight", "posture_cost", "self_collision")
    with patch("umr.retarget.pipeline.sole_sample_points", elf3_mesh_sole_points):
        solver = Replay(robot, human, human_body_ids=cloud.body_ids, human_local_pos=cloud.local_pos,
                        human_local_normal=cloud.local_normal, robot_body_ids=binding.body_ids,
                        robot_local_pos=binding.local_pos, robot_local_normal=binding.local_normal,
                        segment=segment, segment_names=SEGMENTS, **{k: cfg["retarget"][k] for k in keys})
    contacts = [limit for limit in solver.limits if isinstance(limit, mink.CollisionAvoidanceLimit)]
    if len(contacts) != 1:
        raise ValueError("Expected exactly the original UMR contact limiter")
    if args.contact == "recover":
        contact = contacts[0]
        fixed = recovering_contact_class(mink.CollisionAvoidanceLimit)(
            robot.model, [(list(pair[:1]), list(pair[1:])) for pair in contact.geom_id_pairs],
            gain=contact.gain, minimum_distance_from_collisions=args.margin_m,
            collision_detection_distance=.10)
        solver.limits[solver.limits.index(contact)] = fixed
    solver.initialize_root(0)
    warm = solver.solve_frame(0, iterations=30)
    rows = []
    raw_errors = []
    for frame in range(len(prepared["times"])):
        row = solver.solve_frame(frame)
        row["qpos"] = row["qpos"].copy()
        rows.append(row)
        raw, _ = human.targets(frame)
        raw_errors.append(float(np.linalg.norm(
            solver.cache.pos[solver.task_idx] - raw[samples][solver.selected], axis=1).mean()))
        if frame % 50 == 0:
            print(frame, row["point_error"], row["failures"], flush=True)
    qpos = np.stack([r["qpos"] for r in rows])
    fps = float(prepared["fps"])
    audit = motion_audit(robot.model, qpos, fps, args.urdf)
    max_baseline_difference = float(np.max(np.abs(qpos - baseline_q))) if is_original_source else None
    if (is_original_source and args.contact == "upstream" and args.targets == "raw"
            and args.initialization == 'human-root' and max_baseline_difference > 1e-8):
        raise ValueError(f"Control replay is not identical to the source trial: {max_baseline_difference}")
    np.savez_compressed(output / "motion.npz", qpos=qpos, fps=fps, times=prepared["times"],
                        dof_names=np.array([robot.model.joint(j).name for j in range(1, robot.model.njnt)]),
                        root_body="torso_link", quaternion_order="wxyz", source_surface_indices=samples)
    for p, h in protected.items():
        if digest(p) != h:
            raise ValueError("Protected learned correspondence or replay changed")
    write_json(output / "inputs.json", inputs)
    receipt = {"schema": "bfm.elf3_contact_replay/1", "contact_mode": args.contact,
               "original_trial": str(trial), "original_receipt_sha256": digest(trial / "receipt.json"),
               "reused_fixed_stage1": protected, "new_neural_training": False,
               "reused_human_canonical_identity": reused_identity,
               "same_motion_as_stage1_trial": is_original_source,
               "target_mode": args.targets,
               "target_normal_mode": "unchanged_actual_SMPLX_normals",
               "raw_human_surface_error_mean_m": float(np.mean(raw_errors)),
               "canonical_offset_mean_m": float(np.linalg.norm(transport.local_offset, axis=1).mean()),
               "collision_detection_distance_m": .1 if args.contact == "recover" else .01,
               "collision_margin_m": args.margin_m if args.contact == "recover" else .02,
               "urdf_sha256": digest(args.urdf),
               "point_error_mean_m": float(np.mean([r["point_error"] for r in rows])),
               "warmup_failures": warm["failures"], "solve_failures": sum(r["failures"] for r in rows),
               "initialization": args.initialization, "initialization_report": solver.initialization_report,
               "nonlinear_contact_line_search": args.nonlinear_check,
               "line_search_backtracks": solver.backtracks, "rejected_optimizer_steps": solver.rejected_steps,
               "max_original_qpos_difference": max_baseline_difference, "audit": audit,
               "motion_sha256": digest(output / "motion.npz"),
               "promoted_to_training": False, "policy_inference": False}
    write_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--contact", choices=("upstream", "recover"), required=True)
    p.add_argument("--margin-m", type=float, default=.02)
    p.add_argument("--nonlinear-check", action="store_true")
    p.add_argument("--targets", choices=("raw", "morphology"), default="raw")
    p.add_argument("--source", type=Path, help="Optional complete motion with exactly identical canonical shape/samples")
    p.add_argument('--initialization', choices=('human-root','foot-feasible'), default='human-root',
                   help='Explicit pre-IK foot-geometry initial pose; never shifts saved output frames')
    args = p.parse_args()
    if not 0. <= args.margin_m <= .02:
        raise ValueError("Explicit nonnegative contact margin at most the original 20mm")
    if args.nonlinear_check and args.contact != "recover":
        raise ValueError("Nonlinear line search is a separate corrected-contact candidate")
    os.environ.setdefault("MUJOCO_GL", "glfw")
    sys.path.insert(0, str(UMR))
    run(args)
