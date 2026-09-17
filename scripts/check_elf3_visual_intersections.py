#!/usr/bin/env python3
"""Spot-check convex-hull alarms against actual triangle-edge intersections.

Positive segment/triangle intersections confirm a surface crossing. A negative
check is not a general no-collision certificate (containment/coplanarity remain).
Never edits meshes or changes original URDF collision masks.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from elf3_umr_asset import digest, write_json
from run_elf3_umr_trial import UMR


def segment_surface_crossings(mesh, starts, ends):
    bounds = mesh.bounds
    possible = np.all(np.maximum(starts, ends) >= bounds[0] - 1e-8, axis=1) & np.all(
        np.minimum(starts, ends) <= bounds[1] + 1e-8, axis=1)
    starts, ends = starts[possible], ends[possible]
    directions = ends - starts
    lengths = np.linalg.norm(directions, axis=1)
    valid = lengths > 1e-8
    starts, directions, lengths = starts[valid], directions[valid], lengths[valid]
    directions = directions / lengths[:, None]
    # Avoid constructing a whole-mesh ray/triangle candidate matrix. Stop once a
    # real crossing is proven; returned counts are lower bounds, not an area/depth.
    for first in range(0, len(starts), 32):
        positions, direction = starts[first:first+32], directions[first:first+32]
        locations, rays, triangles = mesh.ray.intersects_location(positions, direction, multiple_hits=True)
        if not len(rays):
            continue
        distances = np.einsum("ij,ij->i", locations - positions[rays], direction[rays])
        inside = (distances > 1e-6) & (distances < lengths[first:first+32][rays] - 1e-6)
        if inside.any():
            return {"crossing_hits": int(inside.sum()), "crossing_edges": int(len(np.unique(rays[inside]))),
                    "crossed_triangles": int(len(np.unique(triangles[inside]))),
                    "stopped_after_confirmation": True, "candidate_edges": len(starts)}
    return {"crossing_hits": 0, "crossing_edges": 0, "crossed_triangles": 0,
            "stopped_after_confirmation": False, "candidate_edges": len(starts)}


def run(args):
    import mujoco
    from umr.bodies.surface import geom_mesh_body_local
    report = json.loads(args.diagnostic.read_text())
    results = []
    for row in report["rows"]:
        folder = Path(row["run"])
        if digest(folder / "motion.npz") != row["motion_sha256"]:
            raise ValueError("Diagnostic motion changed")
        inputs = json.loads((folder / "inputs.json").read_text())
        model = mujoco.MjModel.from_xml_path(inputs["robot_xml"])
        data = mujoco.MjData(model)
        frame = row["mesh_screen"]["visual_hull_worst_frame"]
        with np.load(folder / "motion.npz", allow_pickle=False) as z:
            data.qpos[:] = z["qpos"][frame]
        mujoco.mj_forward(model, data)
        names = row["mesh_screen"]["visual_hull_worst_pair"]
        meshes = []
        for name in names:
            geom = model.geom(name).id
            body = model.geom_bodyid[geom]
            mesh = geom_mesh_body_local(model, geom).copy()
            transform = np.eye(4)
            transform[:3, :3] = data.xmat[body].reshape(3, 3)
            transform[:3, 3] = data.xpos[body]
            mesh.apply_transform(transform)
            meshes.append(mesh)
        directional = []
        for a, b in ((0, 1), (1, 0)):
            edges = meshes[a].edges_unique
            directional.append(segment_surface_crossings(meshes[b], meshes[a].vertices[edges[:, 0]],
                                                           meshes[a].vertices[edges[:, 1]]))
        result = {"index": row["index"], "frame": frame, "pair": names, "directions": directional,
                  "surface_crossing_confirmed": any(v["crossing_hits"] > 0 for v in directional),
                  "mesh_watertight": [bool(m.is_watertight) for m in meshes]}
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", {"schema": "bfm.elf3_actual_mesh_spot_check/1", "rows": results,
               "diagnostic_sha256": digest(args.diagnostic), "scope": "only worst hull-alarm pair/frame per motion; positive triangle crossings, not penetration depth"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostic", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    sys.path.insert(0, str(UMR))
    run(p.parse_args())
