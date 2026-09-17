# Tools and file contracts

## Main path

| Entry point | Purpose |
|---|---|
| `elf3_umr_asset.py fetch/convert` | Pinned upstream asset retrieval and 31-DoF MuJoCo conversion |
| `prepare_elf3_migration.py contract/inventory` | Joint parameter contract and local source inventory |
| `plan_elf3_batch.py` | Deduplicated full-source plan, preserving previous split membership if supplied |
| `launch_elf3_library.py` | Safe dry-run/default launcher; `--start` explicitly creates a finite owned service |
| `run_elf3_library.py` | Queue ownership, hashes, attempts, resume and cleanup |
| `elf3_library_worker.py` | Full source preparation → learned correspondence → IK → optional refinement |
| `refine_elf3_whole_body.py` | Explicit raw/legacy input refinement; does not approve training |
| `diagnose_elf3_arm_motion.py` | Endpoint/joint motion and arm/core convex-hull metrics |
| `check_elf3_visual_intersections.py` | Selected triangle-level intersection checks |
| `run_elf3_umr_trial.py preview` | Labeled kinematic MP4/GIF export from a run directory |
| `package_elf3_motions.py` | Hash-checked local robot-state bundle; preserves quality classes, never uploads |

Every command exposes `--help`. Run paths are intentionally explicit and
outputs normally refuse to overwrite an existing directory.

## Batch outputs

```text
local/library/
  plan.json                  # absolute local inputs and frozen code hashes
  status.json                # live while running; stop_requested after graceful stop
  stage1_cache/<identity>/   # learned correspondence for an exact shape/robot/recipe
  job_00000/
    completed.json           # SHA-bound pointer, only for an archived attempt
    attempt_000.log
    attempt_000/
      prepared.npz           # licensed derived human surface data; keep private
      raw/                   # motion.npz, inputs.json, receipt.json
      whole_body/            # optional optimized candidate + receipts/checkpoints
      whole_body.log
      result.json
```

Raw trajectory archives contain `qpos[N,38]`, `times[N]`, `fps=50`, the 31
`dof_names`, `root_body='torso_link'`, `quaternion_order='wxyz'`, surface sample
indices and IK diagnostics. Consumers must map joint names, not assume G1 order.
Root XY/Z is in metres; hinge positions are radians.

`stage2_complete` means a complete raw trajectory was archived, not accepted.
`refinement_complete` means refinement wrote its result, not converged.
`kinematic_review_required` is not a training approval.
Full result/code/asset hashes must match on resume. Finished failed attempts are
preserved; `--resume` skips archived attempts rather than silently rerunning them.
Interrupted attempts can be retried in a new attempt directory. Coefficient
checkpoints do not serialize L-BFGS optimizer history.

## Existing GUI and comparisons

`view_elf3_umr_suite.py` retains the earlier **audited legacy pilot-suite** format
(`bfm.elf3_pilot_suite/1`) and up to 9 explicitly selected variants. This is not
yet a browser for arbitrary new library `result.json` files. Do not fabricate
`kinematic_candidate_pass` to load a quarantined motion into the old suite.
For new library outputs, the `preview` command above is the supported raw
kinematic visualization route.

When you have a locally generated, correctly audited legacy suite:

```bash
"$UMR_PY" scripts/view_elf3_umr_suite.py \
  --suite /absolute/path/to/suite.json --output local/viewer_session \
  --start 1 --max-seconds 600
```

Controls: number keys / arrows select motions; Space pauses; R rewinds; F
toggles follow; V compares a supplied same-source variant at the same frame;
Q exits. Supply `--candidate /absolute/path/to/candidate` for paired candidates.
Non-cyclic clips hold their last frame instead of hiding the loop seam.
Closing the viewer releases its resources. No physics stepping or policy is run.

`compare_elf3_arm_candidate.py` also supports explicit whole-body candidates
and labels rendered comparisons. Raw inputs, receipts and source identities
must match. A recording is visualization evidence, not a success certificate.

## Historical modules

The `run_elf3_prepared_batch.py`, `expand_elf3_umr_prepared.py`,
`run_elf3_arm_batch.py`, `replay_elf3_umr_stage2.py`,
`refine_elf3_umr_trajectory.py`, `refine_elf3_arm_spline.py` and
`audit_elf3_pilot_suite.py` entry points preserve earlier experiments and schemas.
They are included for provenance/review; use the library path for new full
source batches. `umr_backend.py` retains G1 conversion utilities only because
the byte-frozen historical core exposes them; **do not use its G1 exporter for ELF3**.

The repository intentionally omits the original ScaleTrack/IsaacLab training
integration. It cannot train an ELF3 ScaleBFM policy by itself.
