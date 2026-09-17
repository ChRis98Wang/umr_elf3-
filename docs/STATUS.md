# Status and known limitations

Snapshot: **2026-09-17, Asia/Shanghai, after the owner stopped the batch**.
This is a historical report, not a live GitHub dashboard.

The local inventory contained 9,483 unique motion files. At stop, 487 full
Stage II outputs (186,735 frames at 50 Hz) had finalized job receipts.
465 had refinement outputs: 461 quality-quarantined and 4 awaiting kinematic
review. Another 22 retained their full raw output but no completed refinement.
No trajectory was approved for training. No ELF3 policy training or physical
tracking validation had started. The owned service and its descendants exited.

Only finalized job receipts are counted. Interrupted attempts retain local
partial outputs/checkpoints and are not counted as completed. The underlying
licensed data, private machine paths, per-source manifests and recordings are
not distributed here. [Machine-readable aggregate](batch_summary_20260917.json).

## Quality failures

Counts below overlap; a motion can fail multiple checks.

| Check | Motions failing |
|---|---:|
| Optimizer not converged | 398 |
| Foot visual mesh below z = 0 | 358 |
| Original collision geometry self-penetration > 1e-6 m | 314 |
| Hand/tool displacement P95 > 20 mm | 195 |
| Hand/tool displacement maximum > 50 mm | 150 |
| Arm/core visual convex-hull penetration > 1e-6 m | 14 |
| Joint speed over original URDF limit | 3 |

The raw IK receipts recorded zero solver failures for these 487 motions.
That is **not** proof of natural motion, absence of penetration or controller
trackability. Optimizer termination and quality acceptance are separate fields.

Of the 22 pending refinements, 12 require multiple root-rotation charts,
8 exceed the current whole-body optimizer's 1,501-frame support, and 2 failed
mesh-distance bound convergence. Raw outputs are preserved, not cropped or
silently accepted.

## Additional limitations

- The 31-DoF model is kinematic. Actuation, balance, payload handling and contact
  materials have not been physically validated or calibrated on ELF3.
- Original ELF3 collision shapes omit 15 visually represented links. Additional
  arm/core convex-hull checks improve screening but are not full-body certification.
- GUI previews set `qpos` and call forward kinematics, not `mj_step`; there is no
  trained policy, supporting chair, carried box or scene-object interaction.
- Body-model preparation is CPU-only; the main batch's Stage I requires CUDA.
- The current library worker handles raw full-length motion, but its whole-body
  refinement supports only 4–1,501 frames and a bounded root rotation chart.
- A byte-level duplicate split check is not a subject-disjoint evaluation guarantee.
  Review dataset/subject splits before any learning experiment.
- There is no automatic training promotion, no completed ELF3 ScaleBFM integration,
  and no claim that all local AMASS data has been retargeted.

## What changed for the standalone repository

This packaging does not change IK costs, whole-body optimization, thresholds,
the frozen source core/v5, or the stopped original batch. It makes environment
and inventory paths explicit; makes the previous inventory optional; moves the
pure joint-contract validator out of the ScaleTrack dependency; adds a dry-run
launcher, release audit, public tests, CI and documentation; and gives the GUI
a first-item/default-on-screen starting position.

Keep the original workspace to resume old jobs: their manifests bind absolute
paths and code hashes. Portability edits are a new code snapshot, not permission
to alter those receipts.
