# Official UMR → ELF3: actual validation, 2026-09-17

Upstream `hanyang9/UMR@e24fc070030dc0bb0b2c024ecb9f795a3995d725`, unmodified.
Four completed runs: one full licensed AMASS sample (local-only) and three
self-authored test motions used in the README. **644 output frames in total.**
This is not four newly accepted training motions, and not the whole AMASS library.

## Recipe and environment

Independent Python 3.12.13; Torch 2.12.0+cu130 on RTX 5080; NumPy 1.26.4;
SciPy 1.17.1; MuJoCo 3.3.7; Clarabel 0.11.1; Trimesh 5.1.0; the official
SMPL-X dependency fork at `a5b8e4ac`. GPU operation was tested, not just imported.
The older Torch/CUDA wheel suggested by upstream was not used on this GPU.
The existing unofficial and IsaacLab environments were not changed.

All runs build fresh correspondence data and train 4,096-point correspondence
for 500 epochs. Official bidirectional initialization and default LQR filtering
are enabled. No legacy trajectory optimization or robot-output repair is applied.

The original floating-root ELF3 MJCF is used directly. Twelve deterministic
URDF-versus-MJCF FK samples across 36 links gave maximum position-component
error `6.66e-16 m` and rotation-matrix error `9.99e-16`. All 31 axes/ranges and
total declared mass agree with the pinned URDF.

## Measured results

| Complete source | Frames | Queue elapsed | Angle / speed violations | Lowest visual foot Z | Original self-penetration |
|---|---:|---:|---:|---:|---:|
| AMASS motorcycle sample, local-only | 191 | 32.66 s | 0 / 0 | −0.135 mm | 0 mm |
| Authored arm raise | 151 | 31.70 s | 0 / 0 | +15.436 mm | 0 mm |
| Authored single-arm movement | 151 | 31.45 s | 0 / 0 | +15.398 mm | 0 mm |
| Authored shallow squat | 151 | 31.52 s | 0 / 0 | +15.363 mm | 0 mm |

Elapsed time is the runner's successful end-to-end compute/audit time, excluding
environment downloads, source preparation, earlier failed integration attempts
and later rendering. Do not treat this tiny selected sample as a throughput or
motion-quality benchmark for 9,483 sources.

The AMASS clip has a small negative visual-foot height, so it does not meet a
strict nonnegative-floor gate. The three authored examples have positive foot
height, not established physical contact. Collider checks only cover originally
enabled URDF geometry and miss some visual links. All four retain
`training_approved=false` and `physical_tracking_validated=false`.

The licensed AMASS reference is also the source of the previous unofficial
191-frame smoke run. A systematic same-source fidelity, shoulder-continuity and
full-body collision comparison has **not** been completed; we do not claim the
official method has already improved the entire library.

## Integration failures retained, not hidden

1. The first trial completed correspondence learning but stopped at a missing
   SMPL-X segmentation file. The sparse checkout initially omitted it, and then
   exposed only its Git LFS pointer. We fetched the actual 1,323,168-byte file
   and verified SHA-256 `bb69c10801205c9cfb5353fdeb1b9cc5ade53d14c265c3339421cdde8b9c91e7`.
   Preflight now rejects absent files, unresolved pointers and changed hashes.
2. The next pipeline finished all 191 frames but our strict output check rejected
   the generated XML filename. The upstream automatically rewrites a floating
   XML even for this already-floating robot. We explicitly set its supported
   `robot.xml_policy.add_freejoint_root=false` and reran in a fresh directory.
   We did not loosen the identity check or edit upstream code.

Earlier artifacts remain local. The final four receipts report upstream exit 0
and successful read-only audits. Result containers preserve official final qpos
verbatim, including dtype, and export pickle-free named-joint arrays.

## Remaining work

- Broader same-source quality comparison, including known shoulder/collision and
  long-motion failures, before replacing any old library entries.
- Safe multi-source cache reuse/batch integration and complete quality review.
- Dynamic tracking validation and ELF3 control-policy training.
- Downstream locomotion/manipulation tasks and physical acceptance.

The old full-library service remains stopped. The bounded official trial and
three-example services naturally exited with `MainPID=0`; no viewer or training
job is intended to remain running after the export.
