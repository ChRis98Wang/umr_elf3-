#!/usr/bin/env python3
"""Render a verified official ELF3 result; kinematic playback only, no mj_step."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from elf3_umr_asset import digest, write_json
from run_official_elf3 import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder, output = args.run.resolve(), args.output.absolute()
    inputs = verify_inputs(folder)
    receipt = json.loads((folder / "receipt.json").read_text())
    if (receipt.get("status") != "kinematic_result_requires_review"
            or digest(folder / "motion.npz") != receipt["motion_sha256"]
            or digest(folder / "inputs.json") != receipt["inputs_sha256"]):
        raise ValueError("Preview requires an unchanged, completed official run")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    with np.load(folder / "motion.npz", allow_pickle=False) as motion:
        qpos = motion["qpos"].copy()
        fps = float(motion["fps"])
    model = mujoco.MjModel.from_xml_path(inputs["robot_xml"])
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 35, -15, 2.8
    camera.lookat[:] = [*qpos[0, :2], .75]
    option = mujoco.MjvOption()
    option.geomgroup[2:] = 0
    output.mkdir(parents=True)
    gifs = []
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = small = ImageFont.load_default()
    authored = inputs["clock"].get("source_kind") == "authored_parametric_smplx"
    title = Path(inputs["source"]).stem.replace("_", " ").title()
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        with imageio.get_writer(output / "official_elf3.mp4", fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p") as writer:
            for i, pose in enumerate(qpos):
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 76), fill=(20, 24, 30))
                draw.text((16, 10), f"OFFICIAL UMR -> ELF3 | {title}", fill="white", font=font)
                draw.text((16, 36), "KINEMATIC ONLY / NOT A TRAINED POLICY / QUALITY REVIEW REQUIRED",
                          fill=(255, 215, 110), font=small)
                source_label = "Self-authored human test" if authored else "Licensed source: local-only preview"
                draw.text((16, 57), f"{source_label} | Frame {i+1}/{len(qpos)} | 50 Hz", fill="white", font=small)
                writer.append_data(np.asarray(frame))
                if i % 4 == 0:
                    gifs.append(frame.resize((640, 480)))
                if i == len(qpos) // 2:
                    frame.save(output / "middle.png")
    gifs[0].save(output / "official_elf3.gif", save_all=True, append_images=gifs[1:],
                 duration=80, loop=0)
    write_json(output / "preview.json", {"motion_sha256": receipt["motion_sha256"],
               "frames_mp4": len(qpos), "frames_gif": len(gifs), "mj_step_called": False,
               "kinematic_only": True, "training_approved": False,
               "redistribution_permission": "UNCONFIRMED_DO_NOT_UPLOAD",
               "files": {p.name: digest(p) for p in output.iterdir() if p.suffix in (".gif", ".mp4", ".png")}})
    print(output)


if __name__ == "__main__":
    main()
