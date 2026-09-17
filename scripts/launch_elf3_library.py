#!/usr/bin/env python3
"""Print a bounded systemd launch command; execute only with --start.

No dependency installation, source replacement, automatic restart or training.
This launcher does not alter the worker's model, quality gates or frozen recipe.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', default='library', help='Unique suffix for the owned service')
    p.add_argument('--batch-plan', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--robot-xml', type=Path, required=True)
    p.add_argument('--body-model', type=Path, required=True)
    p.add_argument('--prepare-python', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-seconds', type=int, default=82800)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--start', action='store_true', help='Actually start; default only prints the command')
    return p


def build_command(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.name):
        raise ValueError('Service name must be a short alphanumeric suffix')
    if not 1 <= args.workers <= 4 or not 60 <= args.max_seconds <= 82800:
        raise ValueError('Require 1..4 workers and 60..82800 seconds')
    if args.output.is_symlink():
        raise ValueError('Output cannot be a symbolic link')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Existing output requires explicit --resume with identical inputs')
    # Keep the venv executable path: resolving its symlink can lose the venv.
    prepare_python = args.prepare_python.absolute()
    if not prepare_python.is_file() or not os.access(prepare_python, os.X_OK):
        raise ValueError('Provide an executable Python from the SMPL-X environment')
    for path in (args.batch_plan, args.robot_xml, args.body_model,
                 args.assets / 'provenance.json', args.assets / 'elf3.urdf'):
        if not path.is_file():
            raise FileNotFoundError(path)
    unit = f'bfm-umr-refresh-elf3-{args.name}.service'
    cmd = ['systemd-run', '--user', '--unit=' + unit,
           '--property=Type=exec', '--property=Restart=no', '--property=KillMode=control-group',
           '--property=RuntimeMaxSec=86400', '--property=TimeoutStopSec=30',
           '--property=MemoryMax=32G', '--property=TasksMax=512', '--property=CPUQuota=800%',
           '--property=WorkingDirectory=' + str(ROOT), '--setenv=CUDA_VISIBLE_DEVICES=0',
           '--setenv=PYTHONNOUSERSITE=1', '--setenv=PYTHONUNBUFFERED=1',
           '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1',
           '--setenv=MKL_NUM_THREADS=1', '--setenv=NUMEXPR_NUM_THREADS=1',
           sys.executable, str(ROOT / 'scripts/run_elf3_library.py')]
    for flag, path in (('--batch-plan', args.batch_plan), ('--assets', args.assets),
                       ('--robot-xml', args.robot_xml), ('--body-model', args.body_model),
                       ('--output', args.output)):
        cmd += [flag, str(path.resolve())]
    cmd += ['--prepare-python', str(prepare_python), '--workers', str(args.workers),
            '--max-seconds', str(args.max_seconds)]
    if args.resume:
        cmd.append('--resume')
    return unit, cmd


def main():
    args = parser().parse_args()
    unit, cmd = build_command(args)
    print(shlex.join(cmd), flush=True)
    print('Stop: ' + shlex.join(['systemctl', '--user', 'stop', unit]), flush=True)
    if args.start:
        from elf3_umr_asset import verify_assets
        from run_elf3_umr_trial import UMR
        from umr_smplx_source import verify_umr_checkout
        verify_assets(args.assets.resolve())
        verify_umr_checkout(UMR)
        subprocess.run(cmd, check=True, timeout=30)
    else:
        print('DRY RUN: nothing started. Add --start after reviewing inputs and resource limits.')


if __name__ == '__main__':
    main()
