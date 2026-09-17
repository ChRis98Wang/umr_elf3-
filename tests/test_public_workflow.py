"""Portable release entry points; synthetic data only, no production work."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from launch_elf3_library import build_command, parser
from prepare_elf3_migration import inventory, parse_policy_parameters
from plan_elf3_batch import plan
from scripts import umr_full_source_v5 as full
from scripts import umr_heading_source_v6 as heading
from scripts.umr_root_initialization_v6 import root_reference


class PortableLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='elf3-public-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.assets = self.root / 'assets'
        self.assets.mkdir()
        for name in ('provenance.json', 'elf3.urdf'):
            (self.assets / name).write_text('fixture')
        for name in ('plan.json', 'robot.xml', 'body.npz'):
            (self.root / name).write_text('fixture')
        self.python = self.root / 'python-alias'
        self.python.symlink_to(sys.executable)
        self.argv = ['--batch-plan', str(self.root / 'plan.json'), '--assets', str(self.assets),
                     '--robot-xml', str(self.root / 'robot.xml'), '--body-model', str(self.root / 'body.npz'),
                     '--prepare-python', str(self.python), '--output', str(self.root / 'output')]

    def test_dry_run_by_default_and_bounded_cleanup(self):
        args = parser().parse_args(self.argv)
        self.assertFalse(args.start)
        unit, command = build_command(args)
        self.assertEqual(unit, 'bfm-umr-refresh-elf3-library.service')
        for item in ('--property=KillMode=control-group', '--property=Restart=no',
                     '--property=RuntimeMaxSec=86400', '--property=MemoryMax=32G'):
            self.assertIn(item, command)
        self.assertEqual(command[command.index('--prepare-python') + 1], str(self.python))

    def test_cli_dry_run_never_calls_systemd(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/launch_elf3_library.py'), *self.argv],
                                capture_output=True, text=True, timeout=10,
                                env={**os.environ, 'PATH': str(self.root)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('DRY RUN: nothing started', result.stdout)

    def test_shell_like_service_name_rejected(self):
        with self.assertRaises(ValueError):
            build_command(parser().parse_args([*self.argv, '--name', 'bad/name;command']))

    def test_excess_workers_rejected(self):
        with self.assertRaises(ValueError):
            build_command(parser().parse_args([*self.argv, '--workers', '5']))

    def test_existing_output_requires_explicit_resume(self):
        (self.root / 'output').mkdir()
        with self.assertRaises(FileExistsError):
            build_command(parser().parse_args(self.argv))
        _, cmd = build_command(parser().parse_args([*self.argv, '--resume']))
        self.assertIn('--resume', cmd)

    def test_symlink_output_rejected(self):
        (self.root / 'output').symlink_to(self.assets, target_is_directory=True)
        with self.assertRaises(ValueError):
            build_command(parser().parse_args([*self.argv, '--resume']))

    def test_missing_input_rejected(self):
        (self.root / 'plan.json').unlink()
        with self.assertRaises(FileNotFoundError):
            build_command(parser().parse_args(self.argv))


class PortableInventoryTests(unittest.TestCase):
    def test_new_collection_without_private_previous_inventory(self):
        with tempfile.TemporaryDirectory(prefix='elf3-inventory-test-') as tmp:
            root = Path(tmp)
            source = root / 'amass' / 'synthetic'
            source.mkdir(parents=True)
            np.savez(source / 'motion.npz', root_orient=np.zeros((4, 3)),
                     pose_body=np.zeros((4, 63)), trans=np.zeros((4, 3)),
                     betas=np.zeros(16), gender='neutral', surface_model_type='smplx', mocap_frame_rate=100.)
            args = SimpleNamespace(output=root / 'inventory', source_root=source.parent,
                                   previous_inventory=None, body_model=root / 'missing-model.npz')
            with contextlib.redirect_stdout(io.StringIO()):
                inventory(args)
            saved = json.loads((args.output / 'inventory.json').read_text())
            self.assertEqual(saved['valid_smplx_motion_files'], 1)
            self.assertEqual(saved['new_sources_need_split_assignment'], 1)
            self.assertIsNone(saved['previous_inventory_sha256'])
            self.assertEqual(saved['missing_body_model_files'], 1)
            batch = plan(saved, {'schema': 'bfm.elf3_joint_contract/1', 'urdf_joint_order': []})
            self.assertEqual(batch['unique_jobs'], 1)
            self.assertFalse(batch['training_auto_start'])

    def test_inventory_output_cannot_pollute_source_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                inventory(SimpleNamespace(output=root / 'output', source_root=root))

    def test_upstream_parameter_source_is_parsed_not_executed(self):
        names = [f'body_{i}' for i in range(29)]
        rows = [(n, 0., 10., 1., .2) for n in names]
        code = f'raise RuntimeError("never execute")\nELF3_POLICY_JOINTS = Names({names!r})\n'
        code += f'ELF3_ISAAC_JOINTS = Names({names!r})\nELF3_ISAAC_PARAMETERS = Params(None, {rows!r})\n'
        demo = 'ELF3_COMMAND_DEFAULTS = Defaults({"head_y_joint": P(position=0.,kp=10.,kd=1.),"head_z_joint": P(position=0.,kp=10.,kd=1.)})'
        _, _, parsed = parse_policy_parameters(code, demo)
        self.assertEqual(len(parsed), 31)
        self.assertEqual(parsed['head_z_joint']['action_scale'], 0.)


class FrozenSourceTests(unittest.TestCase):
    def test_frozen_core_and_v5_are_byte_identical(self):
        self.assertEqual(full.core.sha256(full.CORE_PATH), full.CORE_SHA256)
        self.assertEqual(full.core.sha256(Path(full.__file__)), heading.V5_SHA)

    def test_long_clock_is_not_cropped_or_stretched(self):
        times, _, _, _ = full.full_sampling_grid(18031, 120.)
        self.assertEqual(times[0], 0.)
        self.assertGreater(times[-1], 150.)
        self.assertLessEqual(times[-1], (18031 - 1) / 120.)
        self.assertLess((18031 - 1) / 120. - times[-1], .02)

    def test_degenerate_forward_heading_has_t0_only_proof(self):
        h, proof = heading.heading_reference(np.zeros(3))
        self.assertTrue(proof['fallback_used'])
        self.assertEqual(proof['future_frames_inspected'], 0)
        trace = root_reference(h, {'heading_reference': proof, 'posed_heading_rotation': h.tolist()})
        self.assertTrue(trace['fallback_used'])
        self.assertEqual(trace['frame_index'], 0)

    def test_global_yaw_equivariance(self):
        r = Rotation.from_rotvec(np.array([.1, .02, .1]))
        z = Rotation.from_euler('z', .8)
        a, _ = heading.heading_reference(r.as_rotvec())
        b, _ = heading.heading_reference((z * r).as_rotvec())
        np.testing.assert_allclose(a @ r.as_matrix(), b @ (z * r).as_matrix(), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
