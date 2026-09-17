#!/usr/bin/env python3
"""Saved-clock wrist/hand stability and visual-mesh arm/core screening.

Visual meshes are queried directly without enabling/changing URDF colliders.
MuJoCo mesh distances use convex hulls: negative distance is a conservative
screen, not proof of exact nonconvex triangle penetration. Loop discontinuities
are reported separately from within-clip motion. Nothing is promoted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from elf3_umr_asset import digest, write_json
from elf3_source_reuse import load_replay_source
from run_elf3_umr_trial import UMR
from umr_smplx_source import SmplxSurfaceHuman, model_geometry_fingerprint


def motion_metrics(position, rotation, fps=50.):
    from scipy.spatial.transform import Rotation
    if position.shape != (len(rotation), 3) or rotation.shape[1:] != (3, 3) or len(position) < 4:
        raise ValueError("Need full position and rotation sequences with >=4 frames")
    velocity = np.diff(position, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    omega = Rotation.from_matrix(rotation[1:] @ rotation[:-1].transpose(0, 2, 1)).as_rotvec() * fps
    alpha = np.diff(omega, axis=0) * fps

    def stats(array):
        norm = np.linalg.norm(array, axis=1)
        return {"rms": float(np.sqrt(np.mean(norm * norm))), "p95": float(np.percentile(norm, 95)),
                "max": float(norm.max()), "worst_interval_start": int(norm.argmax())}

    return {"speed_m_s": stats(velocity), "acceleration_m_s2": stats(acceleration),
            "angular_speed_rad_s": stats(omega), "angular_acceleration_rad_s2": stats(alpha),
            "loop_position_jump_m": float(np.linalg.norm(position[-1] - position[0])),
            "loop_rotation_jump_rad": float(Rotation.from_matrix(rotation[0] @ rotation[-1].T).magnitude())}


def visual_arm_core_pairs(model):
    arms = {side + name for side in ("l_", "r_") for name in
            ("elbow_y_link", "wrist_x_link", "wrist_y_link", "wrist_z_link")}
    core = {"torso_link", "waist_y_link", "waist_x_link", "waist_z_link"} | {
        side + "hip_" + axis + "_link" for side in ("l_", "r_") for axis in ("x", "y", "z")}
    left = [g for g in range(model.ngeom) if model.geom_group[g] == 1 and model.body(model.geom_bodyid[g]).name in arms]
    right = [g for g in range(model.ngeom) if model.geom_group[g] == 1 and model.body(model.geom_bodyid[g]).name in core]
    return [(a, b) for a in left for b in right]


def traces(model, qpos):
    import mujoco
    from umr.bodies.surface import geom_mesh_body_local
    names = ["l_wrist_z_link", "r_wrist_z_link"]
    tips = []
    for name in names:
        body = model.body(name).id
        geoms = [g for g in range(model.ngeom) if model.geom_group[g] == 1 and model.geom_bodyid[g] == body]
        vertices = np.concatenate([np.asarray(geom_mesh_body_local(model, g).vertices) for g in geoms])
        tips.append(vertices[np.linalg.norm(vertices, axis=1).argmax()])
    data = mujoco.MjData(model)
    positions, rotations, tool_positions = [], [], []
    pairs = visual_arm_core_pairs(model)
    penetrations, per_frame_pair = [], []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        p = np.array([data.body(name).xpos.copy() for name in names])
        r = np.array([data.body(name).xmat.reshape(3, 3).copy() for name in names])
        positions.append(p)
        rotations.append(r)
        tool_positions.append(p + np.einsum("nij,nj->ni", r, tips))
        distances = [mujoco.mj_geomDistance(model, data, a, b, .01, None) for a, b in pairs]
        i = int(np.argmin(distances))
        penetrations.append(max(0., -distances[i]))
        per_frame_pair.append(pairs[i])
    p, r, tip = np.asarray(positions), np.asarray(rotations), np.asarray(tool_positions)
    deepest = int(np.argmax(penetrations))
    pair = per_frame_pair[deepest]
    return {"wrist_positions": p, "wrist_rotations": r, "tool_positions": tip,
            "tool_local_points": np.asarray(tips), "visual_hull_penetration_m": np.asarray(penetrations)}, {
        "visual_hull_max_penetration_m": float(max(penetrations)),
        "visual_hull_over_5mm_frames": int(np.sum(np.asarray(penetrations) > .005)),
        "visual_hull_worst_frame": deepest,
        "visual_hull_worst_pair": [model.geom(g).name for g in pair],
        "visual_hull_pairs": len(pairs),
        "scope": "forearm/wrist visual meshes versus torso/waist/hips; convex-hull conservative screen"}


def audit_candidate_arms(folder):
    """Single-run screening for queues; deliberately never grants acceptance."""
    import mujoco
    folder = Path(folder)
    inputs = json.loads((folder / "inputs.json").read_text())
    receipt = json.loads((folder / "receipt.json").read_text())
    if (digest(folder / "motion.npz") != receipt["motion_sha256"]
            or digest(inputs["robot_xml"]) != inputs["robot_xml_sha256"]):
        raise ValueError("Candidate motion/model changed before arm review")
    model = mujoco.MjModel.from_xml_path(inputs["robot_xml"])
    if model_geometry_fingerprint(model) != inputs["robot_geometry"]:
        raise ValueError("Candidate robot geometry changed")
    with np.load(folder / "motion.npz", allow_pickle=False) as z:
        qpos = z["qpos"].copy()
        if (qpos.shape != (len(qpos), model.nq) or not np.isfinite(qpos).all()
                or len(qpos) < 4 or not np.array_equal(z["times"], np.arange(len(qpos)) / 50.)
                or z["dof_names"].tolist() != [model.joint(j).name for j in range(1, model.njnt)]):
            raise ValueError("Candidate layout/clock invalid")
    saved, mesh = traces(model, qpos)
    return {"schema": "bfm.elf3_candidate_arm_review/1", "run": str(folder.resolve()),
            "motion_sha256": digest(folder / "motion.npz"), "frames": len(qpos), "mesh_screen": mesh,
            "metrics": {side: motion_metrics(saved["tool_positions"][:, i], saved["wrist_rotations"][:, i])
                        for i, side in enumerate(("left", "right"))},
            "arm_review_complete": False, "training_approved": False,
            "note": "Screen only. Requires source fidelity, exact-mesh and physical tracking review; no universal wrist speed pass threshold."}


def run(args):
    import mujoco
    from umr.bodies.robot import RobotBody, RobotSpec
    args.output.mkdir(parents=True, exist_ok=False)
    suite = json.loads(args.suite.read_text())
    reports = []
    for index, row in enumerate(suite["rows"]):
        folder = Path(row["run"])
        inputs = json.loads((folder / "inputs.json").read_text())
        receipt = json.loads((folder / "receipt.json").read_text())
        if digest(folder / "motion.npz") != row["motion_sha256"]:
            raise ValueError("Motion changed after suite audit")
        model = mujoco.MjModel.from_xml_path(inputs["robot_xml"])
        if model_geometry_fingerprint(model) != inputs["robot_geometry"]:
            raise ValueError("Robot geometry changed")
        with np.load(folder / "motion.npz", allow_pickle=False) as z:
            qpos = z["qpos"].copy()
        original_path = Path(receipt["source_run"]) / "motion.npz"
        if digest(original_path) != receipt["protected_inputs"][str(original_path)]:
            raise ValueError("Original Stage II trajectory changed")
        with np.load(original_path, allow_pickle=False) as z:
            original = z["qpos"].copy()
        saved, mesh = traces(model, qpos)
        before, _ = traces(model, original)
        source = load_replay_source(inputs["source"], require_full=True)
        human = SmplxSurfaceHuman(source, RobotBody(inputs["robot_xml"], RobotSpec.from_config(inputs["config"]["robot"])).height())
        hp = source["joint_positions"][:, [20, 21]].astype(float) * human.scale
        hr = source["joint_rotations"][:, [20, 21]].astype(float)
        hp[:, :, 2] += human.ground_offset
        details = {}
        for side, side_index in (("left", 0), ("right", 1)):
            details[side] = {"refined_wrist": motion_metrics(saved["wrist_positions"][:, side_index], saved["wrist_rotations"][:, side_index]),
                             "refined_mesh_tip": motion_metrics(saved["tool_positions"][:, side_index], saved["wrist_rotations"][:, side_index]),
                             "original_wrist": motion_metrics(before["wrist_positions"][:, side_index], before["wrist_rotations"][:, side_index]),
                             "human_wrist": motion_metrics(hp[:, side_index], hr[:, side_index])}
        np.savez_compressed(args.output / f"motion_{index+1:02d}_traces.npz", **saved,
                            human_wrist_positions=hp, human_wrist_rotations=hr,
                            original_wrist_positions=before["wrist_positions"],
                            original_wrist_rotations=before["wrist_rotations"])
        report = {"index": index+1, "origin_id": row["origin_id"], "run": row["run"],
                  "motion_sha256": row["motion_sha256"], "metrics": details, "mesh_screen": mesh}
        reports.append(report)
        print(json.dumps({"motion": index+1, "mesh": mesh, "angular_speed_max_rad_s": {
            side: {kind: round(details[side][kind]["angular_speed_rad_s"]["max"], 3)
                   for kind in ("original_wrist", "refined_wrist", "human_wrist")} for side in details}}), flush=True)
    write_json(args.output / "report.json", {"schema": "bfm.elf3_arm_diagnostic/1", "rows": reports,
               "suite_sha256": digest(args.suite), "script_sha256": digest(__file__),
               "policy_inference": False, "training_approved": False,
               "note": "Robot/human arm lengths and joint frames differ. Angular speeds are world-frame geodesic increments, not absolute orientation errors."})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    sys.path.insert(0, str(UMR))
    run(p.parse_args())
