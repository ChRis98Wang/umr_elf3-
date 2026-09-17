#!/usr/bin/env python3
"""Local-only pinned ELF3 asset import for the UMR kinematic pilot.

No ROS, actuators, installation, or modification of the upstream UMR checkout.
The narrow converter fails on unsupported URDF features instead of approximating
them. Original meshes/URDF stay private until redistribution terms are clarified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ELF3_COMMIT = "1c9954040d114b3ff5e133b3611c5b327d19d029"
BASE_URL = ("https://raw.githubusercontent.com/bxirobotics/bxi_controller_ros2/"
            + ELF3_COMMIT + "/resources/elf3_dof31/urdf/")
TPOSE = {"l_shoulder_x_joint": np.pi / 2, "r_shoulder_x_joint": -np.pi / 2,
         "l_elbow_y_joint": np.pi / 2, "r_elbow_y_joint": np.pi / 2}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def safe_mesh_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or len(p.parts) != 2 or p.parts[0] != "meshes":
        raise ValueError(f"Unsupported mesh path: {name}")
    if p.suffix.lower() != ".stl":
        raise ValueError(f"Unsupported mesh format: {name}")
    return p.as_posix()


def fetch_assets(output, resume=False):
    output = Path(output).resolve()
    if output.is_symlink() or (output / "provenance.json").exists():
        raise FileExistsError("Refusing to overwrite a completed/redirected asset directory")
    output.mkdir(parents=True, exist_ok=resume)
    (output / "meshes").mkdir(exist_ok=resume)

    def fetch(relative):
        for attempt in range(3):
            try:
                with urllib.request.urlopen(BASE_URL + relative, timeout=30) as response:
                    content = response.read(80 * 1024 * 1024 + 1)
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                print(f"retry {relative}: attempt {attempt + 2}", flush=True)
        if not content or len(content) > 80 * 1024 * 1024:
            raise ValueError(f"Empty/oversized asset: {relative}")
        path = output / relative
        if path.exists() or path.is_symlink():
            if not resume or path.is_symlink() or digest(path) != hashlib.sha256(content).hexdigest():
                raise ValueError(f"Existing asset does not match freshly fetched pinned source: {relative}")
        else:
            with path.open("xb") as f:
                f.write(content)
        print(f"downloaded {relative}: {len(content)} bytes", flush=True)
        return relative, {"sha256": digest(path), "bytes": len(content)}

    first = fetch("elf3.urdf")
    root = ET.parse(output / "elf3.urdf").getroot()
    paths = sorted({safe_mesh_path(m.attrib["filename"]) for m in root.findall(".//mesh")})
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict([first, *pool.map(fetch, paths)])
    write_json(output / "provenance.json", {
        "schema": "bfm.elf3_assets/1", "upstream_commit": ELF3_COMMIT,
        "upstream_url": BASE_URL, "files": files,
        "redistribution": "not_authorized_by_this_trial; license_review_required",
    })
    return output


def numbers(text):
    return np.asarray([float(x) for x in text.split()])


def fmt(value):
    return " ".join(f"{float(v):.16g}" for v in value)


def transform(origin):
    t = np.eye(4)
    if origin is not None:
        t[:3, 3] = numbers(origin.get("xyz", "0 0 0"))
        t[:3, :3] = Rotation.from_euler("xyz", numbers(origin.get("rpy", "0 0 0"))).as_matrix()
    return t


def pose_attributes(origin):
    t = transform(origin)
    return {"pos": fmt(t[:3, 3]),
            "quat": fmt(Rotation.from_matrix(t[:3, :3]).as_quat(scalar_first=True))}


def verify_assets(assets):
    manifest = json.loads((assets / "provenance.json").read_text())
    if manifest["upstream_commit"] != ELF3_COMMIT:
        raise ValueError("Unexpected ELF3 source revision")
    for relative, expected in manifest["files"].items():
        if relative != "elf3.urdf":
            safe_mesh_path(relative)
        path = assets / relative
        if path.is_symlink() or digest(path) != expected["sha256"]:
            raise ValueError(f"Changed pinned asset: {relative}")
    return manifest


def convert(assets, output):
    """Preserve link frames, URDF joint axes/limits and inertias, including sensors."""
    import mujoco

    assets, output = Path(assets).resolve(), Path(output).resolve()
    manifest = verify_assets(assets)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    urdf = ET.parse(assets / "elf3.urdf").getroot()
    links = {l.attrib["name"]: l for l in urdf.findall("link")}
    joints = urdf.findall("joint")
    movable = [j for j in joints if j.get("type") != "fixed"]
    roots = set(links) - {j.find("child").get("link") for j in joints}
    if roots != {"torso_link"} or len(movable) != 31:
        raise ValueError("Expected the pinned 31-DoF torso-root ELF3")
    mj = ET.Element("mujoco", model="elf3_umr_kinematic")
    ET.SubElement(mj, "compiler", angle="radian", autolimits="true", fusestatic="false",
                  inertiafromgeom="false", balanceinertia="false")
    ET.SubElement(mj, "option", timestep="0.002", gravity="0 0 -9.81")
    asset = ET.SubElement(mj, "asset")
    visual = ET.SubElement(mj, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    world = ET.SubElement(mj, "worldbody")
    ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="floor", type="plane", size="20 20 0.1",
                  rgba="0.2 0.24 0.29 1", contype="1", conaffinity="1", group="0")
    meshes = {}

    def build(name, parent, joint=None):
        link = links[name]
        body = ET.SubElement(parent, "body", name=name,
                             **pose_attributes(joint.find("origin") if joint is not None else None))
        if joint is None:
            ET.SubElement(body, "freejoint", name="floating_base")
        elif joint.get("type") == "revolute":
            lim = joint.find("limit")
            ET.SubElement(body, "joint", name=joint.get("name"), type="hinge",
                          axis=joint.find("axis").get("xyz"),
                          range=f"{lim.get('lower')} {lim.get('upper')}")
        elif joint.get("type") != "fixed":
            raise ValueError(f"Unsupported joint: {joint.attrib}")
        inertial = link.find("inertial")
        if inertial is not None:
            origin = transform(inertial.find("origin"))
            i = inertial.find("inertia").attrib
            tensor = np.array([[float(i["ixx"]), float(i["ixy"]), float(i["ixz"])],
                               [float(i["ixy"]), float(i["iyy"]), float(i["iyz"])],
                               [float(i["ixz"]), float(i["iyz"]), float(i["izz"])]])
            tensor = origin[:3, :3] @ tensor @ origin[:3, :3].T
            ET.SubElement(body, "inertial", pos=fmt(origin[:3, 3]),
                          mass=inertial.find("mass").get("value"),
                          fullinertia=fmt([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                                           tensor[0, 1], tensor[0, 2], tensor[1, 2]]))
        for kind in ("visual", "collision"):
            for idx, element in enumerate(link.findall(kind)):
                geometry = element.find("geometry")
                mesh = geometry.find("mesh")
                if mesh is None or len(geometry) != 1:
                    raise ValueError("This pinned importer supports only the supplied mesh geometry")
                relative = safe_mesh_path(mesh.get("filename"))
                if relative not in manifest["files"]:
                    raise ValueError("Unpinned mesh")
                scale = mesh.get("scale", "1 1 1")
                key = (relative, scale)
                if key not in meshes:
                    meshes[key] = f"mesh_{len(meshes)}"
                    ET.SubElement(asset, "mesh", name=meshes[key], file=str(assets / relative), scale=scale)
                is_visual = kind == "visual"
                color = element.find("material/color")
                rgba = color.get("rgba") if color is not None else "0.75 0.78 0.83 1"
                ET.SubElement(body, "geom", name=f"{name}_{kind}_{idx}", type="mesh",
                              mesh=meshes[key], group="1" if is_visual else "3",
                              contype="0" if is_visual else "1",
                              conaffinity="0" if is_visual else "1", rgba=rgba,
                              **pose_attributes(element.find("origin")))
        for child_joint in joints:
            if child_joint.find("parent").get("link") == name:
                build(child_joint.find("child").get("link"), body, child_joint)

    build("torso_link", world)
    # No artificial joint damping, torque limits, masses or contact exclusions.
    xml = output / "elf3.xml"
    ET.indent(mj)
    ET.ElementTree(mj).write(xml, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    audit = validate_urdf_fk(urdf, model)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    for name, value in TPOSE.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    # Canonical grounding only, never a per-frame output correction.
    lows = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        m = model.geom_dataid[g]
        verts = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        lows.append((verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min())
    data.qpos[2] -= min(lows)
    ET.SubElement(ET.SubElement(mj, "keyframe"), "key", name="T_pose", qpos=fmt(data.qpos))
    ET.indent(mj)
    ET.ElementTree(mj).write(xml, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    arm_vectors = {}
    for side, sign in (("l", 1), ("r", -1)):
        delta = data.body(f"{side}_wrist_x_link").xpos - data.body(f"{side}_shoulder_y_link").xpos
        if not np.allclose(delta, [0, sign * .512, 0], atol=1e-7):
            raise ValueError(f"Invalid ELF3 straight-arm T pose: {delta}")
        arm_vectors[side] = delta.tolist()
    write_json(output / "asset_audit.json", {
        "schema": "bfm.elf3_asset_audit/1", "upstream_commit": ELF3_COMMIT,
        "xml_sha256": digest(xml), "urdf_sha256": digest(assets / "elf3.urdf"),
        "root_body": "torso_link", "nq": model.nq, "nv": model.nv,
        "hinge_count": len(movable), "mass_kg_declared": float(model.body_mass.sum()),
        "urdf_fk": audit, "tpose_arm_vectors_m": arm_vectors,
        "tpose_joints": TPOSE, "collision_geometry": "original_URDF_only; incomplete_link_coverage; mesh_convex_hulls",
        "not_verified": ["dynamics", "motor_parameters", "physical_tracking", "real_hardware"],
    })
    return xml


def validate_urdf_fk(urdf, model, samples=12):
    """Independent matrix-chain FK comparison at seeded in-limit random poses."""
    import mujoco
    joints = urdf.findall("joint")
    moving = [j for j in joints if j.get("type") != "fixed"]
    if (model.nq, model.nv) != (38, 37):
        raise ValueError("ELF3 requires a floating base plus all 31 hinge joints")
    if {model.joint(j).name for j in range(1, model.njnt)} != {j.get("name") for j in moving}:
        raise ValueError("Robot joint names do not match URDF")
    rng = np.random.default_rng(73)
    data = mujoco.MjData(model)
    max_pos = max_rot = 0.
    for sample in range(samples):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = rng.normal(size=3)
        rot = Rotation.random(random_state=rng)
        data.qpos[3:7] = rot.as_quat(scalar_first=True)
        root = np.eye(4)
        root[:3, :3], root[:3, 3] = rot.as_matrix(), data.qpos[:3]
        transforms = {"torso_link": root}
        angles = {}
        for j in moving:
            name, lim = j.get("name"), j.find("limit")
            lo, hi = float(lim.get("lower")), float(lim.get("upper"))
            joint = model.joint(name)
            if not np.allclose(joint.range, [lo, hi], atol=1e-12, rtol=0):
                raise ValueError(f"Lost URDF limits: {name}")
            if not np.allclose(joint.axis, numbers(j.find("axis").get("xyz")), atol=1e-12):
                raise ValueError(f"Changed joint axis: {name}")
            angles[name] = 0. if sample == 0 else rng.uniform(lo, hi)
            data.qpos[joint.qposadr[0]] = angles[name]
        pending = list(joints)
        while pending:
            before = len(pending)
            for j in pending[:]:
                parent = j.find("parent").get("link")
                if parent not in transforms:
                    continue
                t = transform(j.find("origin"))
                if j.get("type") != "fixed":
                    turn = np.eye(4)
                    turn[:3, :3] = Rotation.from_rotvec(numbers(j.find("axis").get("xyz")) * angles[j.get("name")]).as_matrix()
                    t = t @ turn
                transforms[j.find("child").get("link")] = transforms[parent] @ t
                pending.remove(j)
            if len(pending) == before:
                raise ValueError("Cyclic/disconnected URDF")
        mujoco.mj_forward(model, data)
        for name, expected in transforms.items():
            b = data.body(name)
            max_pos = max(max_pos, float(np.max(np.abs(b.xpos - expected[:3, 3]))))
            max_rot = max(max_rot, float(np.max(np.abs(b.xmat.reshape(3, 3) - expected[:3, :3]))))
    if max(max_pos, max_rot) > 1e-10:
        raise ValueError(f"URDF/MJCF FK mismatch: {max_pos}, {max_rot}")
    mass = sum(float(m.get("value")) for m in urdf.findall("link/inertial/mass"))
    if not np.isclose(mass, model.body_mass.sum(), atol=1e-10, rtol=0):
        raise ValueError("Declared mass changed")
    return {"samples": samples, "links_per_sample": len(transforms),
            "max_position_component_error_m": max_pos, "max_rotation_matrix_error": max_rot,
            "joint_axes_and_limits_preserved": True, "declared_mass_preserved": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--resume", action="store_true", help="Re-fetch and hash-check every existing file")
    c = sub.add_parser("convert")
    c.add_argument("--assets", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(fetch_assets(args.output, args.resume) if args.command == "fetch" else convert(args.assets, args.output))


if __name__ == "__main__":
    main()
