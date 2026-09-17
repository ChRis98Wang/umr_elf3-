# Official UMR → ELF3 integration

The new opt-in path calls the authors' **unmodified**
[hanyang9/UMR](https://github.com/hanyang9/UMR), pinned to
`e24fc070030dc0bb0b2c024ecb9f795a3995d725`.
The historical `external/umr_trial_20260908` submodule and its 487 results remain
unchanged and **unofficial**. They are not silently relabeled or overwritten.

This is an offline retargeting integration, not an ELF3 policy, a successful
physical manipulation demonstration, or a completed ScaleBFM reproduction.

## Scope and design

- Reuse locally held ELF3 URDF/MJCF/meshes. Check all 31 joint names, axes,
  original limits, link transforms and total declared mass against the URDF.
- Keep the floating root at `torso_link`. The surface point-cloud center is
  `waist_z_link`; it is a correspondence reference, **not** a changed robot root.
  Disable upstream's optional automatic root/XML rewrite because this checked
  MJCF already has the required floating root.
- Specify ELF3 T-pose shoulder/elbow values, plus all original URDF joint limits.
  No G1 knee/wrist/shoulder limit overrides are inherited.
- Convert complete AMASS **neutral SMPL-X** poses to 50 Hz: SO(3) interpolation
  for rotations, linear interpolation for translation. No cropping, time warping,
  heading alignment, frame freezing or foot-height repair. The unsampled tail is
  less than one 50 Hz interval. Other SMPL variants/genders are explicitly rejected
  by this initial adapter, not substituted with the wrong body model.
- Use official correspondence defaults: 4,096 points, 500 epochs, shape-specific
  human template, and the official batch recipe's bidirectional initialization.
  The official default **LQR trajectory filter remains enabled**. No legacy
  B-spline refinement is applied. “Unchanged qpos export” means unchanged from
  the official **final filtered result**, not an unfiltered solver trajectory.
- The upstream SMPL-X motion path uses ten shape coefficients and neutral
  hands/face. Input interpolation preserves all 55 rotations, but this does not
  mean the official body-only solve reproduces facial/finger articulation.
- Independent read-only checks report angle/speed excess, actual visual foot
  height, original collider penetration and wrist motion. These do not auto-approve
  training and do not certify collision-free full-body motion.

## Installation without changing legacy/IsaacLab environments

The official checkout is local-only. At the pinned revision, the root tree has
no project-level LICENSE (a vendored Three.js license is present). Public access
does not itself establish permission to redistribute/relicense the entire project.
We therefore reference its upstream and do not copy it into our MIT adapter.
Review author terms before distributing official code or assets.

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/hanyang9/UMR.git local/UMR_official_source
git -C local/UMR_official_source fetch --depth 1 origin e24fc070030dc0bb0b2c024ecb9f795a3995d725
git -C local/UMR_official_source switch --detach e24fc070030dc0bb0b2c024ecb9f795a3995d725
git -C local/UMR_official_source sparse-checkout set scripts robot_configs
git -C local/UMR_official_source sparse-checkout add --skip-checks assets/smplx_parts_segm.pkl
git -C local/UMR_official_source lfs install --local
git -C local/UMR_official_source lfs pull --include=assets/smplx_parts_segm.pkl --exclude=''

uv venv --python 3.12 local/venv_umr_official
uv pip install --python local/venv_umr_official/bin/python 'torch==2.12.0'
uv pip install --python local/venv_umr_official/bin/python -r requirements-official-elf3.txt
```

The local RTX 5080 wheel was checked with an actual CUDA matrix operation.
This dependency recipe differs from the older Torch wheel in the upstream README;
it is an explicitly recorded hardware compatibility choice, not a claim that the
official published environment was reproduced exactly. SOMA, object decomposition,
FBX conversion and other source adapters are outside this minimal environment.

The segmentation asset must be the actual 1,323,168-byte file, not a Git LFS
pointer. Preflight verifies its pinned SHA-256. If the LFS batch endpoint is
unavailable, the file is also served by GitHub's
[media endpoint](https://media.githubusercontent.com/media/hanyang9/UMR/e24fc070030dc0bb0b2c024ecb9f795a3995d725/assets/smplx_parts_segm.pkl).
Verify SHA-256 `bb69c10801205c9cfb5353fdeb1b9cc5ade53d14c265c3339421cdde8b9c91e7`
before using it; a different file is not a valid substitute.

## Prepare one complete source

Use the asset import/conversion commands in the legacy setup guide first if the
locally licensed ELF3 assets are not already available. Do not distribute them.

```bash
OFFICIAL_PY=local/venv_umr_official/bin/python
"$OFFICIAL_PY" scripts/run_official_elf3.py prepare \
  --upstream local/UMR_official_source \
  --source /absolute/path/to/neutral_smplx_motion.npz \
  --robot-xml /absolute/path/to/elf3.xml \
  --assets /absolute/path/to/elf3_assets \
  --body-model /absolute/path/to/SMPLX_NEUTRAL_2020.npz \
  --output local/official_trial

# Foreground; finite pipeline timeout and owned subprocess-group cleanup.
"$OFFICIAL_PY" scripts/run_official_elf3.py run \
  --prepared local/official_trial --timeout 1800

MUJOCO_GL=egl "$OFFICIAL_PY" scripts/preview_official_elf3.py \
  --run local/official_trial --output local/official_preview
```

The run refuses changed protected inputs, modified upstream tracked code, or an
already attempted output directory. Retry in a new directory; do not overwrite a
failed run to conceal its provenance. Long unattended runs should additionally use
a bounded systemd user service with `KillMode=control-group`, `Restart=no`, finite
runtime/stop time, memory and task limits. The original full-library service stays
stopped. This integration does **not** start a batch job or policy training.

## Outputs

`inputs.json` records source/model/config hashes, the robot contract and input clock.
`run.log` contains the unmodified official program's logs; `receipt.json` records
completion/failure and dependency versions. `official_result.npz` is kept intact.
`motion.npz` is a pickle-free robot-only export with identical final qpos values:
root XYZ + quaternion WXYZ + 31 named joint angles. `audit.json` reports screening
results, always with `training_approved=false`.

All inputs, body models, correspondence data, generated trajectories and recordings
remain under ignored local paths. Switching code does not change AMASS/SMPL-X data
redistribution terms. The previous 94.17 MiB data bundle remains unpublished.

## Verification boundary

The synthetic tests cover full-duration clocks, rotation wrap-around, source
metadata/shape validation, pickle avoidance, output-directory protection and input
hash checks. They do not demonstrate retarget quality. A completed real-source
trial must be inspected separately, followed by same-source multi-motion comparison
before considering large-scale replacement and downstream tracking training.
