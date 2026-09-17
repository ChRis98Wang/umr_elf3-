"""Validate the pinned ELF3 joint contract without importing a simulator."""

from __future__ import annotations

import hashlib
import ast
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def load_elf3_contract(path):
    """Fail closed on morphology/limit mismatch; preserve explicit joint names."""
    path = Path(path).resolve()
    contract = json.loads(path.read_text())
    if contract.get("schema") != "bfm.elf3_joint_contract/1" or contract.get("root_body") != "torso_link":
        raise ValueError("Expected an explicit torso-root ELF3 contract")
    if (contract.get("motion_anchor_body") != "torso_link"
            or contract.get("sparse_pelvis_body") != "waist_z_link"
            or contract.get("tracking_body_names", [None])[0] != "torso_link"):
        raise ValueError("ELF3 reset root must be torso; sparse pelvis is a separate target frame")
    upstream = {}
    for source_name, source_receipt in contract["sources"].items():
        filename = "upstream_joints.py" if source_name.endswith("policies/joints.py") else "upstream_demo.py"
        source = path.parent / filename
        if sha256(source) != source_receipt["sha256"]:
            raise ValueError("Pinned upstream parameter source changed")
        upstream[filename] = source.read_text()

    def value(source, name):
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return node.value
        raise ValueError(f"Missing pinned assignment: {name}")

    rows_node = value(upstream["upstream_joints.py"], "ELF3_ISAAC_PARAMETERS")
    expected_pd = {n: (pos, kp, kd, scale) for n, pos, kp, kd, scale in ast.literal_eval(rows_node.args[1])}
    head_node = value(upstream["upstream_demo.py"], "ELF3_COMMAND_DEFAULTS").args[0]
    for key, node in zip(head_node.keys, head_node.values):
        fields = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        expected_pd[ast.literal_eval(key)] = (fields["position"], fields["kp"], fields["kd"], 0.)
    urdf = Path(contract["urdf"])
    if sha256(urdf) != contract["urdf_sha256"]:
        raise ValueError("Pinned ELF3 URDF changed")
    tree = ET.parse(urdf).getroot()
    joints = {j.get("name"): j for j in tree.findall("joint") if j.get("type") != "fixed"}
    rows = contract["joints"]
    names = [r["name"] for r in rows]
    if len(names) != 31 or len(set(names)) != 31 or set(names) != set(joints):
        raise ValueError("ELF3 requires all 31 named joints")
    for row in rows:
        if tuple(row[k] for k in ("default_position", "kp", "kd", "action_scale")) != expected_pd[row["name"]]:
            raise ValueError(f"PD/default/action scale differs from pinned upstream: {row['name']}")
        j = joints[row["name"]]
        if (row["type"] != j.get("type") or row["parent"] != j.find("parent").get("link")
                or row["child"] != j.find("child").get("link")
                or row["origin"] != j.find("origin").attrib
                or row["axis"] != [float(x) for x in j.find("axis").get("xyz").split()]
                or row["limit"] != {k: float(v) for k, v in j.find("limit").attrib.items()}):
            raise ValueError(f"ELF3 URDF contract mismatch for {row['name']}")
        if any(not math.isfinite(row[k]) for k in ("default_position", "kp", "kd", "action_scale")):
            raise ValueError("Nonfinite control parameter")
        if (not row["limit"]["lower"] <= row["default_position"] <= row["limit"]["upper"]
                or row["kp"] <= 0 or row["kd"] < 0 or row["action_scale"] < 0):
            raise ValueError("Invalid default or PD gain")
    expected_heads = {"head_y_joint", "head_z_joint"}
    policy_names = contract["upstream_isaac_joint_order"]
    if len(policy_names) != 29 or len(set(policy_names)) != 29 or set(policy_names) != set(names) - expected_heads:
        raise ValueError("ELF3 body action layout does not match the upstream 29-body/2-head split")
    for row in rows:
        if row["name"] in expected_heads and (row["control_role"] != "separate_head_default" or row["action_scale"] != 0.):
            raise ValueError("Head actuator must not silently become a body-policy action")
    link_names = {l.get("name") for l in tree.findall("link")}
    if not set(contract["tracking_body_names"]) <= link_names or contract["motion_anchor_body"] not in link_names:
        raise ValueError("Unknown tracking frame")
    return contract
