#!/usr/bin/env python3
"""Audit staged/tracked release content without reading untracked private data."""
from pathlib import Path
import argparse
import hashlib
import json
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
DEMO_MEDIA = {f'docs/media/official_{name}.{ext}' for name in
              ('arm_raise', 'arm_wave', 'shallow_squat') for ext in ('gif', 'mp4')}


def media_registry(raw):
    manifest = json.loads(raw)
    if (manifest.get('schema') != 'umr_elf3.authored_demo_media/1'
            or manifest.get('upstream_commit') != 'e24fc070030dc0bb0b2c024ecb9f795a3995d725'):
        raise ValueError('Unrecognized demo media manifest')
    entries = {}
    for row in manifest['rows']:
        if (row.get('amass_used') is not False or row.get('source_kind') != 'authored_parametric_smplx'
                or row.get('policy_inference') is not False or row.get('training_approved') is not False):
            raise ValueError('Media must be self-authored, non-policy research examples')
        for filename, value in row['files'].items():
            name = 'docs/media/' + filename
            if name not in DEMO_MEDIA or name in entries:
                raise ValueError('Unapproved/duplicate demo media path')
            if (not re.fullmatch(r'[a-f0-9]{64}', value['sha256'])
                    or type(value['bytes']) is not int or not 0 < value['bytes'] < 8 * (1 << 20)):
                raise ValueError('Invalid media hash/size')
            entries[name] = value
    if set(entries) != DEMO_MEDIA:
        raise ValueError('Missing expected demo GIF/MP4 pair')
    return entries


def verify_media(name, raw, registry):
    expected = registry.get(name)
    if (expected is None or len(raw) != expected['bytes']
            or hashlib.sha256(raw).hexdigest() != expected['sha256']):
        raise ValueError('Media bytes do not match the reviewed manifest')
    if name.endswith('.gif') and raw[:6] not in (b'GIF87a', b'GIF89a'):
        raise ValueError('Invalid GIF header')
    if name.endswith('.mp4') and raw[4:8] != b'ftyp':
        raise ValueError('Invalid MP4 header')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worktree', action='store_true', help='Also check current untracked, non-ignored files')
    args = parser.parse_args()
    entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=ROOT).split(b'\0')
    errors = []
    checked = 0
    media_checked = set()
    registry = {}
    names = {e.decode().split('\t', 1)[1] for e in entries if e}
    manifest_name = 'docs/media/manifest.json'
    if args.worktree:
        names.update(filter(None, subprocess.check_output(
            ['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')))
        indexed = {e.decode().split('\t', 1)[1]: e for e in entries if e}
        entries = [indexed.get(n, f'100644 unknown 0\t{n}'.encode()) for n in sorted(names)]
    if manifest_name in names:
        try:
            raw = ((ROOT / manifest_name).read_bytes() if args.worktree else
                   subprocess.check_output(['git', 'show', ':' + manifest_name], cwd=ROOT))
            registry = media_registry(raw)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f'{manifest_name}: {exc}')
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
        if args.worktree and (ROOT / path).is_symlink():
            mode = '120000'
        if (mode == '120000' or set(path.parts) & BLOCKED_DIRS
                or (path.suffix.lower() in BLOCKED_SUFFIXES and name not in DEMO_MEDIA)):
            errors.append(f'{name}: generated, licensed, sensitive or redirected content')
            continue
        if path.name.startswith(('.env', 'id_rsa', 'id_ed25519')):
            errors.append(f'{name}: private configuration or credential name')
        size = ((ROOT / path).stat().st_size if args.worktree else
                int(subprocess.check_output(['git', 'cat-file', '-s', sha], cwd=ROOT)))
        cap = 8 * (1 << 20) if name in DEMO_MEDIA else 1_000_000
        if size > cap:
            errors.append(f'{name}: oversized release file ({size} bytes)')
            continue
        raw = ((ROOT / path).read_bytes() if args.worktree else
               subprocess.check_output(['git', 'cat-file', 'blob', sha], cwd=ROOT))
        if name in DEMO_MEDIA:
            try:
                verify_media(name, raw, registry)
                media_checked.add(name)
            except ValueError as exc:
                errors.append(f'{name}: {exc}')
            continue
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
    if registry and media_checked != set(registry):
        errors.append('Media manifest references missing or invalid release files')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'Release audit passed: {checked - len(media_checked)} text files, '
          f'{len(media_checked)} verified demo videos/images; pinned legacy submodule checked.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
