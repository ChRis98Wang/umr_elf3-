#!/usr/bin/env python3
"""Native MuJoCo + Chinese motion selector for audited ELF3 kinematic references.

Never calls mj_step, a policy, or a training script. All frames and joint names
are verified against the saved suite. Closing either window stops this viewer;
independent, explicitly started retargeting services are not owned by the GUI.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import queue
import time

import numpy as np

from elf3_umr_asset import digest, write_json
from umr_smplx_source import model_geometry_fingerprint


@dataclass
class Playback:
    lengths: list[int]
    index: int = 0
    position: float = 0.
    paused: bool = False
    speed: float = 1.
    follow: bool = True
    stopped: bool = False
    loop: bool = False
    pairs: dict[int, int] = field(default_factory=dict)
    notice: str = ""

    @property
    def frame(self):
        return min(int(self.position), self.lengths[self.index] - 1)

    def select(self, index):
        self.index = int(index) % len(self.lengths)
        self.position = 0.
        self.notice = ""

    def key(self, key):
        if ord("1") <= key < ord("1") + len(self.lengths):
            self.select(key - ord("1"))
        elif key in (ord("["), 263):
            self.select(self.index - 1)
        elif key in (ord("]"), 262):
            self.select(self.index + 1)
        elif key == 32:
            self.paused = not self.paused
        elif key in (ord("R"), ord("r")):
            self.position = 0.
        elif key in (ord("F"), ord("f")):
            self.follow = not self.follow
        elif key in (ord("V"), ord("v")):
            if self.index in self.pairs:
                # Same source clock, so compare variants at the exact current frame.
                position = self.position
                self.select(self.pairs[self.index])
                self.position = min(position, self.lengths[self.index] - 1)
            else:
                self.notice = "当前动作没有已载入的优化版；V 不会切换。"
        elif key in (ord("Q"), ord("q")):
            self.stopped = True

    def advance(self, seconds):
        if not self.paused:
            position = self.position + max(0., seconds) * 50. * self.speed
            self.position = (position % self.lengths[self.index] if self.loop
                             else min(position, self.lengths[self.index] - 1))


def install_shortcuts(panel, events):
    """Intercept before widget/class bindings, including focused TButton Space.

    A toplevel binding runs AFTER the TButton class and cannot undo its invoke.
    This private first bindtag handles only documented shortcuts, not mouse clicks.
    """
    tag = f"Elf3PlaybackShortcut_{id(panel)}"

    def keycode(event):
        if event.keysym == "space":
            return 32
        char = event.char.upper() if event.char else ""
        if char in tuple("123456789RFVQ[]"):
            return ord(char)
        return None

    def press(event):
        code = keycode(event)
        if code is not None:
            events.put(("key", code))
            return "break"

    def release(event):
        return "break" if keycode(event) is not None else None

    panel.bind_class(tag, "<KeyPress>", press)
    panel.bind_class(tag, "<KeyRelease>", release)

    def register(widget):
        widget.bindtags((tag, *[t for t in widget.bindtags() if t != tag]))
        for child in widget.winfo_children():
            register(child)
    register(panel)
    return tag


def display_label(clip):
    prefix = {'arm_candidate':'[优化候选] ', 'whole_body_candidate':'[全身优化候选] '}
    return prefix.get(clip.get('variant'), '[旧参考] ') + clip["label"]


def load_suite(path, candidates=()):
    import mujoco
    suite = json.loads(path.read_text())
    if suite["schema"] != "bfm.elf3_pilot_suite/1" or not 1 <= len(suite["rows"]) <= 9:
        raise ValueError("Expected a bounded audited ELF3 suite")
    clips, model, geometry = [], None, None
    for row in suite["rows"]:
        folder = Path(row["run"])
        if (not row["kinematic_candidate_pass"] or digest(folder / "motion.npz") != row["motion_sha256"]
                or digest(folder / "receipt.json") != row["receipt_sha256"]):
            raise ValueError("Suite motion changed or was not accepted as a kinematic candidate")
        inputs = json.loads((folder / "inputs.json").read_text())
        if digest(inputs["robot_xml"]) != inputs["robot_xml_sha256"]:
            raise ValueError("Robot XML changed")
        if model is None:
            model = mujoco.MjModel.from_xml_path(inputs["robot_xml"])
            geometry = model_geometry_fingerprint(model)
        if inputs["robot_geometry"] != geometry:
            raise ValueError("All clips must reference the same verified ELF3 geometry")
        with np.load(folder / "motion.npz", allow_pickle=False) as z:
            qpos = z["qpos"].copy()
            if (qpos.shape != (row["frames"], 38) or not np.isfinite(qpos).all()
                    or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1., atol=1e-5, rtol=0)
                    or z["dof_names"].tolist() != [model.joint(j).name for j in range(1, model.njnt)]
                    or str(z["root_body"].item()) != "torso_link"
                    or str(z["quaternion_order"].item()) != "wxyz" or float(z["fps"]) != 50.
                    or not np.array_equal(z["times"], np.arange(len(qpos)) / 50.)):
                raise ValueError("Invalid ELF3 frame layout or clock")
        name = row["origin_id"].split("/")[-1].removesuffix("_stageii")
        if "sitting" in name:
            label = "坐下参考 " + name.split("sitting")[-1]
        elif "lifting_light" in name:
            label = "轻物抬举参考 " + name.split("lifting_light")[-1]
        else:
            label = "重物抬举参考 " + name.split("lifting_heavy")[-1]
        clips.append({"qpos": qpos, "name": name, "label": label, "row": row, "variant": "legacy"})
    if candidates:
        from compare_elf3_arm_candidate import load_pair
        if len(clips) + len(candidates) > 9:
            raise ValueError("Viewer supports at most 9 explicitly selected variants")
        seen = set()
        for candidate in candidates:
            candidate_model, pair, source, receipt = load_pair(Path(candidate))
            matches = [i for i, c in enumerate(clips) if Path(c["row"]["run"]).resolve() == source]
            if len(matches) != 1 or matches[0] in seen:
                raise ValueError("Each candidate must match a distinct original suite motion")
            original = matches[0]
            if (model_geometry_fingerprint(candidate_model) != geometry
                    or not np.array_equal(pair[0], clips[original]["qpos"])):
                raise ValueError("Comparison model/original motion differs from suite")
            seen.add(original)
            clips[original]["paired_index"] = len(clips)
            variant = 'whole_body_candidate' if receipt['schema']=='bfm.elf3_whole_body_spline/1' else 'arm_candidate'
            clips.append({"qpos": pair[1], "name": clips[original]["name"] + "__" + variant,
                          "label": clips[original]["label"], "paired_index": original, "variant": variant,
                          "row": {"run": str(Path(candidate).resolve()), "motion_sha256": receipt["motion_sha256"],
                                  "arm_review_complete": False, "training_approved": False}})
    return model, clips


def run(args):
    import mujoco
    import mujoco.viewer
    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import ttk

    if not 1 <= args.max_seconds <= 7200:
        raise ValueError("Viewer lifetime must be bounded to 1..7200 seconds")
    model, clips = load_suite(args.suite.resolve(), args.candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    state = Playback([len(c["qpos"]) for c in clips],
                     pairs={i: c["paired_index"] for i, c in enumerate(clips) if "paired_index" in c})
    state.select(args.start - 1)
    events = queue.SimpleQueue()
    event_log = (args.output / "events.jsonl").open("x")
    panel = tk.Tk()
    panel.title("ELF3 动作选择 · UMR 运动学参考")
    panel.geometry(f"460x{730 + 35 * max(0, len(clips)-6)}+40+70")
    panel.configure(padx=14, pady=10)
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
        tkfont.nametofont(name).configure(family="Noto Sans CJK SC", size=11)
    ttk.Label(panel, text=f"ELF3 · {len(clips)} 条开发候选", font=("Noto Sans CJK SC", 16, "bold")).pack(anchor="w")
    ttk.Label(panel, text="末端稳定性与可见网格碰撞复查中，未验收。\n不是训练策略；不进行物理推进。\n场景不含椅子/箱子，未验证支撑或负载。", foreground="#a64b00").pack(anchor="w", pady=8)
    buttons = []
    for index, clip in enumerate(clips):
        button = ttk.Button(panel, text=f"{index + 1}   {display_label(clip)}",
                            command=lambda i=index: events.put(("select", i)))
        button.pack(fill="x", pady=3)
        buttons.append(button)
    controls = ttk.Frame(panel)
    controls.pack(fill="x", pady=8)
    ttk.Button(controls, text="暂停 / 继续", command=lambda: events.put(("key", 32))).pack(side="left")
    ttk.Button(controls, text="从头播放", command=lambda: events.put(("key", ord("R")))).pack(side="left", padx=6)
    speed = tk.StringVar(value="1.0")
    speed_box = ttk.Combobox(controls, textvariable=speed, values=("0.25", "0.5", "1.0", "2.0"), width=5, state="readonly")
    speed_box.pack(side="left")
    speed_box.bind("<<ComboboxSelected>>", lambda e: events.put(("speed", float(speed.get()))))
    compare_button = ttk.Button(panel, text="原版 / 优化版，同帧对照（V）",
                                command=lambda: events.put(("key", ord("V"))))
    compare_button.pack(fill="x", pady=5)
    follow = tk.BooleanVar(value=True)
    ttk.Checkbutton(panel, text="相机跟随机器人（F 切换）", variable=follow,
                    command=lambda: events.put(("follow", follow.get()))).pack(anchor="w")
    loop = tk.BooleanVar(value=False)
    ttk.Checkbutton(panel, text="循环播放（非周期动作首尾会跳变）", variable=loop,
                    command=lambda: events.put(("loop", loop.get()))).pack(anchor="w")
    frame_var = tk.DoubleVar(value=0.)
    seek = ttk.Scale(panel, from_=0, to=1, variable=frame_var,
                     command=lambda value: events.put(("seek", float(value))))
    seek.pack(fill="x", pady=8)
    status = ttk.Label(panel, text="正在启动 MuJoCo…", wraplength=390)
    status.pack(anchor="w")
    ttk.Label(panel, text=f"1–{len(clips)} 切换 · 空格暂停 · R 重播 · Q 退出\nV 原版/优化版同帧切换（支持的动作）\n鼠标拖动旋转视角，滚轮缩放。\n关闭任一窗口即退出查看器；批处理独立运行。", wraplength=390).pack(anchor="w", pady=12)
    panel.protocol("WM_DELETE_WINDOW", lambda: events.put(("key", ord("Q"))))
    install_shortcuts(panel, events)
    data = mujoco.MjData(model)
    data.qpos[:] = clips[state.index]["qpos"][0]
    mujoco.mj_forward(model, data)
    started = time.monotonic()
    seen, changed = set(), True
    viewer = None
    frames_drawn = 0
    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=lambda key: events.put(("key", key)),
                                         show_left_ui=False, show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.8, 135, -15
                viewer.opt.geomgroup[3] = 0
            print(f"[ready] Native MuJoCo viewer + Chinese selector. 1..{len(clips)} / V / Space / R / Q", flush=True)
            write_json(args.output / "ready.json", {"suite": str(args.suite.resolve()), "suite_sha256": digest(args.suite),
                       "motions": len(clips), "physics_step": False, "policy_inference": False,
                       "candidates": [str(p.resolve()) for p in args.candidate], "comparison_pairs": state.pairs,
                       "script_sha256": digest(__file__), "max_seconds": args.max_seconds})
            last = time.monotonic()
            while viewer.is_running() and not state.stopped and time.monotonic() - started < args.max_seconds:
                panel.update()
                now = time.monotonic()
                state.advance(now - last)
                last = now
                while not events.empty():
                    action, value = events.get()
                    if action == "key":
                        state.key(value)
                    elif action == "select":
                        state.select(value)
                    elif action == "speed":
                        state.speed = value
                    elif action == "follow":
                        state.follow = value
                    elif action == "loop":
                        state.loop = bool(value)
                    elif action == "seek":
                        state.position = np.clip(value, 0., 1.) * (state.lengths[state.index] - 1)
                    event_log.write(json.dumps({"elapsed": now - started, "action": action, "value": value,
                                               "index": state.index, "frame": state.frame, "paused": state.paused}) + "\n")
                    event_log.flush()
                    changed = True
                clip = clips[state.index]
                seen.add(state.index)
                with viewer.lock():
                    data.qpos[:] = clip["qpos"][state.frame]
                    data.qvel[:] = 0.
                    data.time = state.frame / 50.
                    mujoco.mj_forward(model, data)
                    if state.follow:
                        viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], .75]
                if changed:
                    panel.title(f"ELF3 · {state.index + 1} {display_label(clip)}")
                    compare_button.configure(state="normal" if state.index in state.pairs else "disabled")
                    follow.set(state.follow)
                    changed = False
                frame_var.set(state.frame / max(1, len(clip["qpos"]) - 1))
                mode = ('已到末帧，按 R 重播' if not state.loop and state.frame == len(clip['qpos']) - 1
                        else '暂停' if state.paused else '播放')
                variant_note = (f"V 同帧对照第 {state.pairs[state.index]+1} 项" if state.index in state.pairs
                                else "此项没有已载入的优化版")
                status.configure(text=f"当前：{state.index + 1} {display_label(clip)}\n{state.frame + 1}/{len(clip['qpos'])} 帧  |  {data.time:.2f} 秒  |  {state.speed:g} 倍速\n{mode}\n{state.notice or variant_note}")
                viewer.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  f"ELF3 31 DoF | {state.index + 1}/{len(clips)} {clip['name']}\nKINEMATIC ONLY - NOT a trained policy\n1..{len(clips)} select | V compare | SPACE pause | R restart | F follow | Q exit", ""),
                                 (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                  "UNDER ARM / MESH REVIEW - NOT ACCEPTED\nNo physics stepping. No chair / object support.\n" + "\n".join(f"{i+1}: {c['name']}" for i, c in enumerate(clips)), "")])
                viewer.sync()
                frames_drawn += 1
                time.sleep(.01)
    finally:
        if viewer is not None:
            viewer.close()
        panel.destroy()
        event_log.close()
        write_json(args.output / "closed.json", {"elapsed_seconds": time.monotonic() - started,
                   "selected_motion_indices": sorted(seen), "frames_drawn": frames_drawn,
                   "viewer_closed": True, "policy_inference": False, "physics_step": False})
        print("[closed] MuJoCo and selector closed; no viewer worker retained", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True, type=Path)
    p.add_argument("--candidate", action="append", type=Path, default=[], help="Append an explicit same-source arm/full-body candidate; V toggles at the same frame")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--max-seconds", type=int, default=3600)
    run(p.parse_args())
