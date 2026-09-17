#!/usr/bin/env python3
"""Author three synthetic human-pose smoke tests; never read AMASS motions.

These are designed parametric input trajectories, not captured behavior or robot
joint scripts. Robot motion must subsequently be solved by official UMR.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from elf3_umr_asset import digest, write_json

DEMO_NAMES = ("arm_raise", "arm_wave", "shallow_squat")


def author_poses(name, frames=151):
    if name not in DEMO_NAMES or frames < 3:
        raise ValueError("Unknown authored example or invalid frame count")
    phase = np.linspace(0, 2 * np.pi, frames)
    lift = .5 - .5 * np.cos(phase)
    pose = np.zeros((frames, 55, 3), dtype=np.float64)
    pose[:, 0] = Rotation.from_matrix([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]).as_rotvec()
    pose[:, 16, 2], pose[:, 17, 2] = -1.15, 1.15
    pose[:, 18, 1], pose[:, 19, 1] = -.15, .15
    if name == "arm_raise":
        pose[:, 16, 2] += 1.05 * lift
        pose[:, 17, 2] -= 1.05 * lift
    elif name == "arm_wave":
        pose[:, 16, 2] += .95 * lift
        pose[:, 18, 1] -= .85 * lift
        pose[:, 18, 2] = .45 * lift + .25 * lift * np.sin(3 * phase)
        pose[:, 20, 2] = .25 * lift * np.sin(3 * phase)
    else:
        bend = .30 * lift
        pose[:, [1, 2], 0] = -bend[:, None]
        pose[:, [4, 5], 0] = 2 * bend[:, None]
        pose[:, [7, 8], 0] = -bend[:, None]
        pose[:, 16, 2] += .25 * lift
        pose[:, 17, 2] -= .25 * lift
        pose[:, 18, 1] -= .55 * lift
        pose[:, 19, 1] += .55 * lift
    return pose.reshape(frames, 165)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--body-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    import torch
    import smplx
    torch.set_num_threads(2)
    model = smplx.SMPLX(str(args.body_model.resolve()), gender="neutral", use_pca=False,
                        flat_hand_mean=True, num_betas=10, ext="npz", batch_size=1)
    feet = np.isin(model.lbs_weights.detach().numpy().argmax(axis=1), [7, 8, 10, 11])
    if not feet.any():
        raise ValueError("Cannot identify human foot vertices")
    args.output.mkdir(parents=True)
    rows = []
    for name in DEMO_NAMES:
        poses = author_poses(name)
        trans = np.zeros((len(poses), 3), dtype=np.float64)
        # Construct the human reference's floor placement, not a repair of the
        # solved robot trajectory. Only robot qpos comes from UMR downstream.
        for i in range(len(poses)):
            with torch.inference_mode():
                body = model(global_orient=torch.tensor(poses[i:i+1, :3], dtype=torch.float32),
                             body_pose=torch.tensor(poses[i:i+1, 3:66], dtype=torch.float32),
                             betas=torch.zeros((1, 10)), transl=torch.zeros((1, 3)),
                             left_hand_pose=torch.zeros((1, 45)), right_hand_pose=torch.zeros((1, 45)))
            trans[i, 2] = .008 - float(body.vertices[0, feet, 2].min())
        path = args.output / f"{name}.npz"
        np.savez_compressed(path, poses=poses, trans=trans, betas=np.zeros(10), gender="neutral",
                            surface_model_type="smplx", mocap_frame_rate=50.,
                            source_kind="authored_parametric_smplx")
        rows.append({"name": name, "source_sha256": digest(path), "frames": len(poses), "fps": 50.,
                     "source_kind": "authored_parametric_smplx", "amass_used": False})
        print(f"authored {name}: {len(poses)} frames", flush=True)
    write_json(args.output / "sources.json", {
        "schema": "umr_elf3.authored_demo_sources/1", "generator_sha256": digest(__file__),
        "body_model_sha256": digest(args.body_model), "rows": rows,
        "scope": "synthetic pipeline demonstrations, not motion-capture quality benchmarks",
        "human_floor_placement": "authored source foot surface 8mm; no robot output correction",
        "publish_scope": "robot-only rendered videos, not SMPL-X models/meshes or AMASS data"})


if __name__ == "__main__":
    main()
