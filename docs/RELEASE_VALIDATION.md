# Standalone packaging validation

Recorded on 2026-09-17. This validates the packaged **code**, not an accepted
ELF3 dataset or a trained controller. The production queue stayed stopped.

## Local checks

- `python -m unittest discover -s tests -p 'test_*.py'`: **176 discovered,
  149 passed, 27 explicitly skipped, zero failures**.
- The skips require private ELF3/AMASS artifacts or opt-in native Tk interaction.
  Synthetic geometry, IK/contact logic, spline/rotation math, identity checks,
  queue counts, timeout cleanup, inventory portability and launcher dry-run
  behavior were exercised without restarting the batch.
- Thirteen public CLI `--help` checks passed, including asset import, inventory,
  planning, launching, queue/worker, refinement, preview, GUI and diagnostics.
- Relative Markdown links resolved inside the standalone checkout.
- `git diff --cached --check` passed; the release-content audit passed before
  commit. Only text code/docs and the pinned backend Git submodule are tracked.
- UMR backend checkout was clean at its pinned commit; no upstream source edits.

The local checks reused existing installed environments. They did not reinstall
IsaacLab, rerun learning/retargeting, perform a throughput benchmark or open a GUI.
A GitHub CPU workflow is provided; consult its live status badge for the remote
result rather than interpreting this local report as a CI-success claim.

## Extraction scope

Twenty-four original adapter scripts were copied byte-for-byte. Three received
portable-entry changes: `prepare_elf3_migration.py` (explicit source/model paths,
optional prior inventory), `run_elf3_library.py` (explicit preparation Python),
and `view_elf3_umr_suite.py` (first-item/on-screen defaults).

New helpers provide a bounded dry-run launcher, release-content checks, a pure
joint-contract validator extracted from the prior robot module, and a scripts
package marker. Original workspace files and saved run receipts were untouched.

The frozen preparation modules retain their original hashes:

| Module | SHA-256 |
|---|---|
| `umr_smplx_source.py` | `d2b41fe5cffaed03637d3e37cbdf9fef1f4c30996d636b6146483bf5e9b66b94` |
| `umr_full_source_v5.py` | `5a8a788ba9d42efb4db257e821554c9ba11a1363fdc4972d1792057e1aff7f23` |

No AMASS/SMPL-X/ELF3 data assets or generated trajectories were published.
The aggregate batch report excludes per-source records and private host paths.
