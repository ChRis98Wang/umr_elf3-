#!/usr/bin/env python3
"""Independent saved-motion audit and synchronized before/after MuJoCo video.

No physics, policy, interpolated frames, speed changes, or camera differences.
The source side is the previous UMR + whole-body refinement, not raw AMASS.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from diagnose_elf3_arm_motion import motion_metrics, traces
from elf3_umr_asset import digest, write_json
from run_elf3_umr_trial import UMR
from umr_smplx_source import model_geometry_fingerprint


def load_pair(candidate):
    import mujoco
    candidate = Path(candidate).resolve()
    receipt = json.loads((candidate / "receipt.json").read_text())
    if receipt["schema"] not in ("bfm.elf3_arm_spline_refinement/1", "bfm.elf3_whole_body_spline/1"):
        raise ValueError("Expected explicit arm/full-body spline candidate, not a policy video")
    whole_body = receipt["schema"] == "bfm.elf3_whole_body_spline/1"
    if whole_body and receipt.get("root_and_all_31_joints_optimized") is not True:
        raise ValueError("Missing whole-body optimization declaration")
    source = Path(receipt["source_run"])
    paths = [source / "motion.npz", candidate / "motion.npz"]
    if (digest(paths[0]) != receipt["protected_inputs"][str(paths[0])]
            or digest(paths[1]) != receipt["motion_sha256"]):
        raise ValueError("Source/candidate motion hash changed")
    inputs = json.loads((candidate / "inputs.json").read_text())
    if digest(inputs["robot_xml"]) != inputs["robot_xml_sha256"]:
        raise ValueError("Candidate XML changed")
    model = mujoco.MjModel.from_xml_path(inputs["robot_xml"])
    if model_geometry_fingerprint(model) != inputs["robot_geometry"]:
        raise ValueError("Original robot geometry changed")
    clips = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            q = z["qpos"].copy()
            if (q.shape != (len(q), 38) or not 4 <= len(q) <= 1501 or not np.isfinite(q).all()
                    or float(z["fps"]) != 50. or not np.array_equal(z["times"], np.arange(len(q))/50.)
                    or z["dof_names"].tolist() != [model.joint(j).name for j in range(1, model.njnt)]
                    or str(z["quaternion_order"].item()) != "wxyz" or str(z["root_body"].item()) != "torso_link"
                    or not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1., atol=1e-5, rtol=0)):
                raise ValueError("Comparison requires original full clock and 31-DoF convention")
            clips.append(q)
    arms = [model.joint(j).qposadr[0] for j in range(1, model.njnt)
            if any(s in model.joint(j).name for s in ("shoulder", "elbow", "wrist"))]
    unchanged = np.setdiff1d(np.arange(model.nq), arms)
    if clips[0].shape != clips[1].shape:
        raise ValueError("Source clock changed")
    if not whole_body and not np.array_equal(clips[0][:, unchanged], clips[1][:, unchanged]):
        raise ValueError("Source clock or non-arm trajectory changed")
    return model, clips, source, receipt


def fidelity_stats(before, after):
    from scipy.spatial.transform import Rotation
    output = {}
    for i, side in enumerate(("left", "right")):
        position = np.linalg.norm(after["tool_positions"][:, i] - before["tool_positions"][:, i], axis=1)
        wrist = np.linalg.norm(after["wrist_positions"][:, i] - before["wrist_positions"][:, i], axis=1)
        rotation = Rotation.from_matrix(after["wrist_rotations"][:, i] @ before["wrist_rotations"][:, i].transpose(0, 2, 1)).magnitude()
        output[side] = {}
        for label, values in (("tip_displacement_m", position), ("wrist_displacement_m", wrist), ("rotation_error_rad", rotation)):
            output[side][label] = {"mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
                                   "max": float(values.max()), "worst_frame": int(values.argmax())}
    return output


def run(args):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio

    model, clips, source, receipt = load_pair(args.candidate)
    whole_body = receipt['schema'] == 'bfm.elf3_whole_body_spline/1'
    args.output.mkdir(parents=True, exist_ok=False)
    saved, screens = zip(*(traces(model, clip) for clip in clips))
    fidelity = fidelity_stats(*saved)
    metrics = {side: {stage: motion_metrics(s["tool_positions"][:, i], s["wrist_rotations"][:, i])
                     for stage, s in zip(("before", "after"), saved)}
               for i, side in enumerate(("left", "right"))}
    # Recompute instead of trusting optimization logs. All metrics are fresh FK.
    for side in metrics:
        for stage in metrics[side]:
            for key in metrics[side][stage]:
                actual, expected = metrics[side][stage][key], receipt["arm_metrics"][side][stage][key]
                if isinstance(actual, dict):
                    actual, expected = list(actual.values()), list(expected.values())
                if not np.allclose(actual, expected, rtol=1e-8, atol=1e-9):
                    raise ValueError("Independent wrist audit disagrees with optimizer receipt")
    comparison = {"schema": "bfm.elf3_whole_body_comparison/1" if whole_body else "bfm.elf3_arm_comparison/1", "source_run": str(source),
                  "candidate": str(args.candidate.resolve()), "frames": len(clips[0]),
                  "motion_sha256": [digest(folder / "motion.npz") for folder in (source, args.candidate)],
                  "metrics": metrics, "fidelity": fidelity, "mesh_screens": dict(zip(("before", "after"), screens)),
                  "root_and_non_arm_bit_identical": not whole_body, "training_approved": False,
                  "physical_tracking_validated": False, "policy_inference": False}
    if whole_body:
        from refine_elf3_whole_body import joint_temporal_metrics
        comparison['joint_metrics'] = {stage:joint_temporal_metrics(model,q) for stage,q in zip(('before','after'),clips)}
        for stage in ('before','after'):
            for joint, values in comparison['joint_metrics'][stage].items():
                for key, value in values.items():
                    if not np.allclose(value,receipt['joint_metrics'][stage][joint][key],atol=1e-9,rtol=1e-8):
                        raise ValueError('Independent joint audit disagrees with receipt')
    write_json(args.output / "comparison.json", comparison)
    # Cases for the exact triangle spot checker: old worst pair on both versions,
    # and the new worst pair. A negative spot check is not global certification.
    cases = [(source, screens[0], "old_worst_before"),
             (args.candidate, screens[0], "old_worst_after"),
             (args.candidate, screens[1], "new_worst_after")]
    write_json(args.output / "mesh_cases.json", {"rows": [
        {"index": i+1, "case": label, "run": str(folder.resolve()),
         "motion_sha256": digest(folder / "motion.npz"), "mesh_screen": screen}
        for i, (folder, screen, label) in enumerate(cases)]})
    if args.audit_only:
        return
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.5, 135, -12
    camera.lookat[:] = np.r_[clips[0][:, :2].mean(axis=0), .8]
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    data = mujoco.MjData(model)
    video = args.output / "elf3_arm_before_after.mp4"
    snapshots = {0, len(clips[0])//2, len(clips[0])-1,
                 fidelity["left"]["tip_displacement_m"]["worst_frame"],
                 fidelity["right"]["tip_displacement_m"]["worst_frame"],
                 screens[0]["visual_hull_worst_frame"]}
    with mujoco.Renderer(model, height=720, width=640) as renderer, imageio.get_writer(
        str(video), fps=50, codec="libx264", quality=8, ffmpeg_params=["-threads", "2", "-movflags", "+faststart"]) as writer:
        for frame in range(len(clips[0])):
            # One shared source-root camera: never independently recenter the
            # candidate and hide changes to its root trajectory.
            camera.lookat[:] = np.r_[clips[0][frame, :2], .8]
            pictures = []
            for index, qpos in enumerate(clips):
                data.qpos[:] = qpos[frame]
                data.qvel[:] = 0.
                data.time = frame / 50.
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                im = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, 640, 105), fill=(19, 24, 35))
                after_label = "AFTER: continuous whole-body model" if whole_body else "AFTER: continuous arms + wrist pose"
                draw.text((14, 10), ("BEFORE: previous UMR refinement" if index == 0 else after_label), font=font, fill="white")
                draw.text((14, 39), f"ELF3 | frame {frame+1}/{len(qpos)} | {frame/50.:.2f}s | 1x speed", font=small, fill="white")
                draw.text((14, 66), "KINEMATIC REFERENCE - NOT a trained policy", font=small, fill=(255, 205, 110))
                draw.rectangle((0, 656, 640, 720), fill=(19, 24, 35))
                stage = "before" if index == 0 else "after"
                peaks = [metrics[side][stage]["angular_speed_rad_s"]["max"] for side in ("left", "right")]
                draw.text((14, 665), f"Whole-clip wrist peaks L/R: {peaks[0]:.2f} / {peaks[1]:.2f} rad/s", font=small, fill="white")
                draw.text((14, 691), "Under review: no physics, object contact or payload", font=small, fill=(255, 205, 110))
                pictures.append(np.asarray(im))
            combined = np.concatenate(pictures, axis=1)
            writer.append_data(combined)
            if frame in snapshots:
                Image.fromarray(combined).save(args.output / f"frame_{frame:04d}.png")
    write_json(args.output / "video_receipt.json", {"kind": "synchronized_kinematic_comparison", "fps": 50,
               "frames": len(clips[0]), "video_sha256": digest(video), "comparison_sha256": digest(args.output / "comparison.json"),
               "identical_camera": True, "camera": "shared_source_root_xy_follow", "interpolation": False, "time_scaling": False,
               "physics_step": False, "policy_inference": False})
    print(video, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit-only", action="store_true")
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(UMR))
    run(p.parse_args())
