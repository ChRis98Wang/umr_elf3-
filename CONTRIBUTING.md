# Contributing

Keep changes small, reproducible and explicit about their validation scope.
This is experimental retargeting code, not a certified robot controller.

1. Initialize the pinned UMR submodule; leave upstream code clean.
2. Use an existing compatible environment or a separate development venv.
3. Run `python -m unittest discover -s tests -p 'test_*.py' -v`.
4. Stage only intended code/docs and run `python scripts/check_release.py`.
5. Describe the test results, skips and any same-source before/after metrics.

Do not upload AMASS clips, SMPL-X models, human surface archives, robot assets,
weights, generated motion artifacts, recordings, credentials, full local logs
or private machine paths. `.gitignore` is a convenience, not license clearance.
The release audit checks tracked/staged text and the allowed Git submodule;
it is a useful guard, not a substitute for human review or a universal secret scanner.

Never equate a valid receipt, zero IK exceptions or a smooth GIF with a passed
motion-quality gate. Report convergence, geometry, endpoint fidelity and actual
source duration separately. No automatic promotion of quarantined data.

The frozen core/v5 source files have embedded SHA dependencies. Do not reformat
them. Algorithm changes should use a new explicit version, tests and new output
directories rather than editing hashes to accept old results.

Public CI needs no licensed data and skips optional local-asset tests. Those
skips must be reported. New numerical tests should use synthetic geometry and
should not start long-running learning, a GUI, or an unbounded subprocess.
