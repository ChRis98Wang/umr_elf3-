# Installation and local prerequisites

## 1. Checkout

```bash
git clone --recurse-submodules https://github.com/ChRis98Wang/umr_elf3-.git
cd umr_elf3-
# For a clone made without --recurse-submodules:
git submodule update --init
```

The UMR submodule must remain clean at
`0aa1855fe4f65a73681ffbd1d9f95ab1c2bad9ca`. Do not use `git submodule update --remote`.
The dated directory name is retained to minimize changes to the audited pipeline.
No backend source or G1 policy is transplanted into the ELF3 adapter.

## 2. Two environments; no IsaacLab installation

Use existing environments if they meet the recorded requirements. Do not run an
installer inside your IsaacLab/ScaleBFM environment merely to run this adapter.

| Purpose | Recorded Python | Dependency file |
|---|---|---|
| Correspondence, IK, refinement, preview | 3.10 | `requirements-retarget.txt` |
| SMPL-X surface preparation on CPU | 3.11 | `requirements-prepare.txt` |
| Public synthetic tests only | 3.10 | `requirements-ci.txt` |

Optional setup for a **new machine**, not required for existing installations:

```bash
python3.10 -m venv .venv-umr
.venv-umr/bin/python -m pip install -r requirements-retarget.txt
python3.11 -m venv .venv-smplx
.venv-smplx/bin/python -m pip install -r requirements-prepare.txt
```

Torch is large; choose an appropriate official PyTorch wheel/index for your
machine. The pin records the tested version, not a guarantee of driver support
on every system. GPU/driver compatibility must be checked before a batch.
The public CPU tests do not require Torch or a GPU.

```bash
UMR_PY=/absolute/path/to/umr-venv/bin/python
SMPLX_PY=/absolute/path/to/smplx-venv/bin/python
"$UMR_PY" -c 'import mujoco,mink,numpy,scipy,torch; print(mujoco.__version__,torch.cuda.is_available())'
CUDA_VISIBLE_DEVICES='' "$SMPLX_PY" -c 'import smplx,torch,numpy; print(numpy.__version__)'
"$UMR_PY" scripts/launch_elf3_library.py --help
```

GUI tools additionally require the interpreter's `tkinter` module, a desktop
display and a working OpenGL context. Headless MP4/GIF export uses EGL and
imageio-ffmpeg. The default queue does not launch any GUI.

## 3. Licensed inputs

Obtain AMASS **SMPL-X** motion files and matching SMPL-X `.npz` body models
yourself under the applicable terms. Older AMASS SMPL-H/SMPL files are not
silently treated as SMPL-X. Metadata-only and unsupported files are recorded as
inventory rejects. Check `rejected_count` and `missing_body_model_files` before
creating a batch.

`--body-model` accepts a matching model file or a directory containing the
gender-specific model files recognized by `model_file_for_gender`. Never rename
a neutral model to pretend it is a male/female model. **The current batch runner
requires a single model file for its frozen input contract**; mixed-gender
collections need separate gender-matched plans/runs or a future runner extension.
The recorded 9,483-motion local plan used neutral sources.

The ELF3 fetcher downloads assets directly from a pinned upstream commit into
an ignored local directory. Source hashes are stored in `provenance.json`.
The converter preserves 31 joints and original URDF kinematics, limits and
inertias; it is not a motor/contact calibration or a PhysX-parity certificate.

## 4. Resources and ownership

- Linux with a working `systemd --user` manager; the queue fails outside its
  owned finite service. It is not a portable Windows/background-daemon wrapper.
- Stage I currently requires NVIDIA GPU 0, `nvidia-smi`, at least 4 GiB free
  memory at admission and a usable CUDA PyTorch install. Only one learner runs
  at a time, with a 20% process memory-fraction limit; this does not guarantee
  every GPU can fit the job. There is no CPU fallback in the library worker.
- CPU preparation, IK and refinement run with single-threaded BLAS per worker.
  Up to 4 workers, service CPU quota 800%, memory cap 32 GiB, task cap 512.
- At least 50 GiB free disk is required to continue admitting jobs. This is a
  reserve check, **not an estimate of the space needed for an entire library**.
- Submissions stop after at most 23 hours; the service has a hard 24-hour limit.
  Jobs have a 4,200-second timeout. The service never automatically restarts.
- Prepared surfaces, checkpoints and receipts are currently retained per attempt.
  Processing is bounded in concurrency, **not in total accumulated disk use**.
  Full-library uncompressed human surfaces can occupy hundreds of GiB.
- Use `systemctl --user stop bfm-umr-refresh-elf3-NAME.service`. `KillMode=control-group`
  cleans owned worker descendants; do not use broad `pkill python` commands.

If the service fails, inspect its journal and per-attempt logs. Do not delete
inputs or change frozen code to make `--resume` accept mismatching results.
Changing the algorithm or recipe requires a new output directory and plan.
