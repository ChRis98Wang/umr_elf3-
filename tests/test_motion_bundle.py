"""Export tests use fabricated robot arrays, never licensed motion data."""
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from elf3_umr_asset import digest
from package_elf3_motions import package, robot_arrays, FIELDS


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='elf3-bundle-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'run'
        self.attempt = self.run / 'job_00000/attempt_000'
        self.raw = self.attempt / 'raw'
        self.raw.mkdir(parents=True)
        self.frames = 4
        q = np.zeros((self.frames, 38)); q[:, 3] = 1.
        self.arrays = dict(qpos=q, times=np.arange(self.frames)/50., fps=50.,
                           dof_names=np.array([f'joint_{i}' for i in range(31)]),
                           root_body='torso_link', quaternion_order='wxyz',
                           source_surface_indices=np.arange(10), point_error=np.ones(4))
        np.savez_compressed(self.raw / 'motion.npz', **self.arrays)
        self.receipt = {'schema': 'bfm.elf3_full_stage2/1', 'training_approved': False,
                        'motion_sha256': digest(self.raw / 'motion.npz'), 'full_duration_verified': True,
                        'frames': self.frames, 'warmup_failures': 0, 'solve_failures': 0, 'audit': {}}
        self.result = {'source_sha256': 'a'*64, 'origin_id': 'Synthetic/test', 'split': 'train',
                       'frames': self.frames, 'stage2_complete': True, 'refinement_complete': False,
                       'training_approved': False, 'status': 'raw_complete_refinement_pending'}
        self.plan = {'schema': 'bfm.elf3_library_queue/1', 'jobs': [
            {'source_sha256': 'a'*64, 'origin_id': 'Synthetic/test', 'split': 'train',
             'target_50hz_frames': self.frames}]}
        self.save()

    def save(self):
        (self.run / 'plan.json').write_text(json.dumps(self.plan))
        (self.raw / 'receipt.json').write_text(json.dumps(self.receipt))
        artifacts = {f'raw/{n}': digest(self.raw / n) for n in ('motion.npz', 'receipt.json')}
        self.result.setdefault('artifacts', {}).update(artifacts)
        (self.attempt / 'result.json').write_text(json.dumps(self.result))
        pointer = {'result': str(self.attempt / 'result.json'), 'sha256': digest(self.attempt / 'result.json')}
        (self.attempt.parent / 'completed.json').write_text(json.dumps(pointer))

    def export(self, selection='all'):
        return package(self.run, self.root / 'bundle', selection)

    def test_raw_arrays_preserved_surface_metadata_excluded(self):
        result = self.export()
        self.assertEqual(result['unique_motions'], 1)
        self.assertFalse(result['training_approved'])
        self.assertEqual(result['redistribution_permission'], 'UNCONFIRMED_DO_NOT_UPLOAD')
        with zipfile.ZipFile(self.root / 'bundle' / result['archive_name']) as z:
            manifest = json.loads(z.read('manifest.json'))
            self.assertNotIn(str(self.root), json.dumps(manifest))
            with np.load(BytesIO(z.read('raw/job_00000.npz')), allow_pickle=False) as motion:
                self.assertEqual(set(motion.files), set(FIELDS))
                for key in FIELDS:
                    np.testing.assert_array_equal(motion[key], self.arrays[key])

    def test_corrupt_result_pointer_rejected(self):
        (self.attempt / 'result.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.export()

    def test_changed_motion_rejected_without_success_summary(self):
        with (self.raw / 'motion.npz').open('ab') as f:
            f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.export()
        self.assertTrue((self.root / 'bundle/INCOMPLETE.json').exists())
        self.assertFalse((self.root / 'bundle/summary.json').exists())

    def test_output_refuses_overwrite(self):
        self.export()
        with self.assertRaises(FileExistsError):
            self.export()

    def test_output_cannot_modify_frozen_run(self):
        with self.assertRaises(ValueError):
            package(self.run, self.run / 'export')

    def test_outside_run_result_rejected(self):
        pointer = {'result': str(self.root / 'elsewhere.json'), 'sha256': 'a'*64}
        (self.root / 'elsewhere.json').write_text('{}')
        (self.attempt.parent / 'completed.json').write_text(json.dumps(pointer))
        with self.assertRaises(ValueError):
            self.export()

    def test_review_only_excludes_pending(self):
        with self.assertRaisesRegex(ValueError, 'No finalized'):
            self.export('review-only')
        self.assertFalse((self.root / 'bundle').exists())

    def test_optimized_variants_keep_quarantine_status(self):
        opt = self.attempt / 'whole_body'; opt.mkdir()
        np.savez_compressed(opt / 'motion.npz', **self.arrays)
        receipt = {'schema': 'bfm.elf3_whole_body_spline/1', 'training_approved': False,
                   'motion_sha256': digest(opt/'motion.npz'), 'audit': {},
                   'optimizer_success': False, 'iterations': 450,
                   'tip_displacement_p95_m': .03, 'tip_displacement_max_m': .06,
                   'raw_human_surface_error_mean_m': .01, 'input_raw_human_surface_error_mean_m': .01}
        (opt/'receipt.json').write_text(json.dumps(receipt))
        self.result.update(refinement_complete=True, status='quarantined_quality')
        self.result['artifacts'].update({f'whole_body/{n}': digest(opt/n) for n in ('motion.npz','receipt.json')})
        self.save()
        result = self.export()
        self.assertEqual(result['optimized_variants'], 1)
        with zipfile.ZipFile(self.root/'bundle'/result['archive_name']) as z:
            self.assertIn('optimized/quarantined_quality/job_00000.npz', z.namelist())
            manifest = json.loads(z.read('manifest.json'))
            self.assertFalse(manifest['motions'][0]['variants']['whole_body']['metrics']['optimizer_success'])

    def test_clock_mismatch_rejected(self):
        self.arrays['times'] += 1.
        np.savez_compressed(self.raw / 'motion.npz', **self.arrays)
        with self.assertRaises(ValueError):
            robot_arrays(self.raw / 'motion.npz', self.frames)


if __name__ == '__main__':
    unittest.main()
