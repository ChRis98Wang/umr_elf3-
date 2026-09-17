#!/usr/bin/env python3
"""Audit staged/tracked release content without reading untracked private data."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = {'.npz', '.npy', '.pkl', '.pickle', '.pt', '.pth', '.ckpt',
                    '.onnx', '.stl', '.obj', '.urdf', '.usd', '.usda', '.usdc',
                    '.mp4', '.gif', '.zip', '.bz2', '.gz', '.pem', '.key', '.log'}
BLOCKED_DIRS = {'local', 'data', 'assets', 'models', 'outputs', 'runs', 'logs', '.ssh'}
PATTERNS = {
    'private key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    'GitHub token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'host-specific home path': re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/'),
}


def main():
    entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=ROOT).split(b'\0')
    errors = []
    checked = 0
    for entry in filter(None, entries):
        meta, name = entry.decode().split('\t', 1)
        mode, sha, stage = meta.split()
        path = Path(name)
        if stage != '0':
            errors.append(f'{name}: unresolved index entry')
        if mode == '160000':
            if name != 'external/umr_trial_20260908' or sha != '0aa1855fe4f65a73681ffbd1d9f95ab1c2bad9ca':
                errors.append(f'{name}: unexpected dependency commit')
            continue
        checked += 1
        if mode == '120000' or set(path.parts) & BLOCKED_DIRS or path.suffix.lower() in BLOCKED_SUFFIXES:
            errors.append(f'{name}: generated, licensed, sensitive or redirected content')
        if path.name.startswith(('.env', 'id_rsa', 'id_ed25519')):
            errors.append(f'{name}: private configuration or credential name')
        size = int(subprocess.check_output(['git', 'cat-file', '-s', sha], cwd=ROOT))
        if size > 1_000_000:
            errors.append(f'{name}: oversized release file ({size} bytes)')
            continue
        raw = subprocess.check_output(['git', 'cat-file', 'blob', sha], cwd=ROOT)
        try:
            content = raw.decode('utf-8')
        except UnicodeDecodeError:
            errors.append(f'{name}: unexpected binary file')
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                errors.append(f'{name}: {label}')
        if path.suffix == '.py':
            try:
                compile(content, name, 'exec')
            except SyntaxError as exc:
                errors.append(f'{name}: {exc}')
    if checked == 0:
        errors.append('No tracked release files; stage the explicit release scope first')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'Release audit passed: {checked} text files; pinned upstream submodule checked.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
