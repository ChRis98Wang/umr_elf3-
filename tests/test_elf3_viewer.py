import json
import os
import queue
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from view_elf3_umr_suite import Playback, load_suite, install_shortcuts, display_label


class PlaybackTests(unittest.TestCase):
    def test_selection_and_wrap(self):
        state = Playback([100] * 6)
        state.key(ord("6"))
        self.assertEqual(state.index, 5)
        state.key(262)
        self.assertEqual(state.index, 0)

    def test_pause_speed_and_seek_clock(self):
        state = Playback([100])
        state.advance(.2)
        self.assertEqual(state.frame, 10)
        state.key(32)
        state.advance(1.)
        self.assertEqual(state.frame, 10)
        state.key(32)
        state.speed = .5
        state.advance(.4)
        self.assertEqual(state.frame, 20)
        state.key(ord("R"))
        self.assertEqual(state.frame, 0)

    def test_exit_and_follow(self):
        state = Playback([100])
        state.key(ord("F"))
        self.assertFalse(state.follow)
        state.key(ord("Q"))
        self.assertTrue(state.stopped)

    def test_lifetime_loops_without_interpolation(self):
        state = Playback([20], loop=True)
        state.advance(.5)
        self.assertEqual(state.frame, 5)

    def test_noncyclic_motion_holds_last_frame_by_default(self):
        state = Playback([20])
        state.advance(.5)
        self.assertEqual(state.frame, 19)
        state.advance(1.)
        self.assertEqual(state.frame, 19)
        state.key(ord("R"))
        self.assertEqual(state.frame, 0)
        state.advance(.02)
        self.assertEqual(state.frame, 1)

    def test_compare_keeps_frame_speed_and_pause(self):
        state = Playback([100, 100], pairs={0: 1, 1: 0}, paused=True, speed=.5)
        state.position = 42.5
        state.key(ord("V"))
        self.assertEqual(state.index, 1)
        self.assertEqual(state.position, 42.5)
        self.assertTrue(state.paused)
        self.assertEqual(state.speed, .5)
        state.key(ord("V"))
        self.assertEqual(state.index, 0)

    def test_unpaired_compare_does_not_reset_and_explains(self):
        state = Playback([100], position=42.)
        state.key(ord("V"))
        self.assertEqual(state.position, 42.)
        self.assertIn("没有", state.notice)

    def test_explicit_old_and_candidate_labels(self):
        self.assertIn("旧参考", display_label({"label": "test", "variant": "legacy"}))
        self.assertIn("优化候选", display_label({"label": "test", "variant": "arm_candidate"}))


@unittest.skipUnless(os.environ.get("ELF3_TK_TESTS") == "1", "Opt-in native Tk interaction test")
class TkShortcutTests(unittest.TestCase):
    def test_focused_button_space_only_emits_pause(self):
        import tkinter as tk
        from tkinter import ttk
        root = tk.Tk()
        events, invoked = queue.SimpleQueue(), []
        try:
            button = ttk.Button(root, text="motion selector test", command=lambda: invoked.append(True))
            button.pack()
            tag = install_shortcuts(root, events)
            self.assertEqual(button.bindtags()[0], tag)
            root.update()
            button.focus_force()
            root.update()
            button.event_generate("<KeyPress-space>")
            button.event_generate("<KeyRelease-space>")
            root.update()
            self.assertEqual(events.get_nowait(), ("key", 32))
            self.assertTrue(events.empty())
            self.assertEqual(invoked, [])
            button.invoke()
            self.assertEqual(invoked, [True])
        finally:
            root.destroy()


SUITE = ROOT / "local/elf3_pilot_suite_20260915a/suite.json"


@unittest.skipUnless(SUITE.exists(), "Optional audited local motion suite")
class RealSuiteTests(unittest.TestCase):
    def test_six_motion_layouts_load_without_gui_or_physics(self):
        model, clips = load_suite(SUITE)
        self.assertEqual(model.nq, 38)
        self.assertEqual(len(clips), 6)
        self.assertEqual(sum(len(c["qpos"]) for c in clips), 2291)

    @unittest.skipUnless((ROOT / "local/elf3_arm_spline_20260915c/receipt.json").exists(), "Optional candidate")
    def test_candidate_append_and_pair_identity(self):
        _, clips = load_suite(SUITE, [ROOT / "local/elf3_arm_spline_20260915c"])
        self.assertEqual(len(clips), 7)
        self.assertEqual(clips[3]["paired_index"], 6)
        self.assertEqual(clips[6]["paired_index"], 3)
        self.assertFalse(clips[6]["row"]["arm_review_complete"])


if __name__ == "__main__":
    unittest.main()
