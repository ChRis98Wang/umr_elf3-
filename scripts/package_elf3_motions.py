#!/usr/bin/env python3
"""Create a LOCAL robot-only motion bundle from finalized library receipts.

This does not upload, assign a data license, or approve training. Raw motions
and optimized candidates remain separate, with the original quality status.
Human surfaces, body models, robot meshes and private host paths are excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from elf3_umr_asset import digest, write_json

FIELDS = ('qpos', 'times', 'fps', 'dof_names', 'root_body', 'quaternion_order')
STATUSES = {'quarantined_quality', 'kinematic_review_required',
            'raw_complete_refinement_pending', 'quarantined_ik_failure'}
AUDIT_FIELDS = ('joint_limit_violation_frames', 'joint_limit_max_excess_rad',
                'joint_speed_over_urdf_limit_intervals', 'joint_speed_max_rad_s',
                'foot_visual_mesh_min_z_m', 'original_collision_self_penetration_max_m')


def contained_file(root, path):
    path = Path(path)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file() or path.is_symlink():
        raise ValueError('Input must be a regular file within the selected run')
    return resolved


def checked_json(path, expected):
    if not re.fullmatch(r'[0-9a-f]{64}', expected) or digest(path) != expected:
        raise ValueError('Receipt hash mismatch: ' + Path(path).name)
    return json.loads(Path(path).read_text())


def robot_arrays(path, frames):
    if not 2 <= frames <= 90001:
        raise ValueError('Unexpected motion length')
    with zipfile.ZipFile(path) as archive:
        if sum(row.file_size for row in archive.infolist()) > 256 * 1024 * 1024:
            raise ValueError('Oversized motion archive')
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in FIELDS}
    q = arrays['qpos']
    names = arrays['dof_names']
    if (q.shape != (frames, 38) or not np.isfinite(q).all()
            or not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1., atol=1e-5, rtol=0)
            or names.shape != (31,) or names.dtype.kind not in 'US'
            or len(set(names.tolist())) != 31
            or any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', n) for n in names.tolist())
            or float(arrays['fps']) != 50.
            or not np.array_equal(arrays['times'], np.arange(frames) / 50.)
            or str(arrays['root_body'].item()) != 'torso_link'
            or str(arrays['quaternion_order'].item()) != 'wxyz'):
        raise ValueError('Expected full-clock finite 31-DoF ELF3 robot arrays')
    return arrays


def selected_metrics(receipt, kind):
    out = {key: receipt['audit'][key] for key in AUDIT_FIELDS if key in receipt.get('audit', {})}
    if kind == 'raw':
        out.update({key: receipt[key] for key in ('warmup_failures', 'solve_failures')})
    else:
        out.update({key: receipt[key] for key in ('optimizer_success', 'iterations',
                    'tip_displacement_p95_m', 'tip_displacement_max_m',
                    'raw_human_surface_error_mean_m', 'input_raw_human_surface_error_mean_m')})
    return out


def package(run, output, selection='all'):
    run, output = Path(run).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Refusing to overwrite a bundle directory')
    if output.resolve() == run or output.resolve().is_relative_to(run):
        raise ValueError('Keep export output outside the frozen source run')
    plan = json.loads((run / 'plan.json').read_text())
    if plan.get('schema') != 'bfm.elf3_library_queue/1':
        raise ValueError('Expected a finalized library queue')
    candidates = []
    for pointer_path in sorted(run.glob('job_*/completed.json')):
        folder = pointer_path.parent
        if not re.fullmatch(r'job_[0-9]{5}', folder.name):
            raise ValueError('Unexpected job folder name')
        index = int(folder.name[4:])
        pointer = json.loads(contained_file(run, pointer_path).read_text())
        result_path = contained_file(folder, pointer['result'])
        result = checked_json(result_path, pointer['sha256'])
        if not result.get('stage2_complete'):
            continue
        row = plan['jobs'][index]
        if (result['status'] not in STATUSES or result.get('training_approved') is not False
                or result['source_sha256'] != row['source_sha256']
                or result['origin_id'] != row['origin_id'] or result['split'] != row['split']
                or result['frames'] != row['target_50hz_frames']):
            raise ValueError('Job identity, status or clock differs from its frozen plan')
        origin = Path(result['origin_id'])
        if (origin.is_absolute() or '..' in origin.parts
                or not re.fullmatch(r'[0-9a-f]{64}', result['source_sha256'])):
            raise ValueError('Expected relative source identifier and SHA-256, not host paths')
        if selection == 'review-only' and result['status'] != 'kinematic_review_required':
            continue
        candidates.append((folder.name, result_path.parent, result))
    if not candidates:
        raise ValueError('No finalized full motions match the requested selection')
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    names = None
    archive_path = output / 'elf3_robot_motions_NOT_TRAINING_APPROVED.zip'
    try:
        # Member NPZs are already compressed; do not recompress the ZIP wrapper.
        with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for job, attempt, result in candidates:
                entry = {'job_id': job, 'origin_id': result['origin_id'], 'split': result['split'],
                         'source_sha256': result['source_sha256'], 'frames': result['frames'],
                         'quality_status': result['status'], 'training_approved': False, 'variants': {}}
                for kind in ('raw', 'whole_body'):
                    if kind == 'whole_body' and not result['refinement_complete']:
                        continue
                    receipt_path = contained_file(attempt, attempt / kind / 'receipt.json')
                    receipt = checked_json(receipt_path, result['artifacts'][kind + '/receipt.json'])
                    expected_schema = ('bfm.elf3_full_stage2/1' if kind == 'raw'
                                       else 'bfm.elf3_whole_body_spline/1')
                    if receipt.get('schema') != expected_schema or receipt.get('training_approved') is not False:
                        raise ValueError('Unexpected receipt schema or training claim')
                    motion = contained_file(attempt, attempt / kind / 'motion.npz')
                    expected = result['artifacts'][kind + '/motion.npz']
                    if digest(motion) != expected or receipt['motion_sha256'] != expected:
                        raise ValueError('Motion checksum disagrees with archived evidence')
                    if kind == 'raw' and (receipt.get('full_duration_verified') is not True
                                          or receipt['frames'] != result['frames']):
                        raise ValueError('Missing full-duration raw evidence')
                    arrays = robot_arrays(motion, result['frames'])
                    if names is None:
                        names = arrays['dof_names'].tolist()
                    if arrays['dof_names'].tolist() != names:
                        raise ValueError('Inconsistent named-joint ordering in bundle')
                    if digest(motion) != expected:
                        raise ValueError('Motion changed during export')
                    target = (f'raw/{job}.npz' if kind == 'raw' else
                              f'optimized/{result["status"]}/{job}.npz')
                    buffer = BytesIO()
                    np.savez_compressed(buffer, **arrays)
                    data = buffer.getvalue()
                    archive.writestr(target, data)
                    entry['variants'][kind] = {'path': target, 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest(),
                        'original_motion_sha256': expected,
                        'metrics': selected_metrics(receipt, kind)}
                entries.append(entry)
            summary = {'schema': 'umr_elf3.robot_bundle/1', 'selection': selection,
                'unique_motions': len(entries), 'raw_variants': len(entries),
                'optimized_variants': sum('whole_body' in x['variants'] for x in entries),
                'frames_at_50hz': sum(x['frames'] for x in entries),
                'by_quality_status': dict(Counter(x['quality_status'] for x in entries)),
                'by_dataset': dict(Counter(x['origin_id'].split('/')[0] for x in entries)),
                'root_body': 'torso_link', 'quaternion_order': 'wxyz', 'fps': 50.,
                'dof_names': names, 'training_approved': False, 'physical_tracking_validated': False,
                'redistribution_permission': 'UNCONFIRMED_DO_NOT_UPLOAD',
                'plan_sha256': digest(run / 'plan.json'), 'export_script_sha256': digest(__file__)}
            manifest = {**summary, 'motions': entries}
            archive.writestr('manifest.json', json.dumps(manifest, indent=2, allow_nan=False) + '\n')
            archive.writestr('DATA_NOTICE.txt',
                'LOCAL RESEARCH BUNDLE — NOT TRAINING APPROVED\n'
                'Derived from licensed AMASS/SMPL-X inputs. Code MIT license does not cover data.\n'
                'Redistribution permission is unconfirmed: do not upload this bundle without permission.\n'
                'Raw and optimized motions are separate; quarantined motions retain their failure status.\n'
                'Only robot state arrays are included. qpos/times/joint names are unchanged.\n')
        write_json(output / 'manifest.json', manifest)
        summary.update(archive_name=archive_path.name, archive_bytes=archive_path.stat().st_size,
                       archive_sha256=digest(archive_path))
        write_json(output / 'summary.json', summary)
        return summary
    except BaseException:
        # Preserve partial files for inspection; absence of summary means incomplete.
        write_json(output / 'INCOMPLETE.json', {'complete': False, 'do_not_upload': True})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--selection', choices=('all', 'review-only'), default='all')
    args = p.parse_args()
    print(json.dumps(package(args.run, args.output, args.selection), indent=2))


if __name__ == '__main__':
    main()
