# Attribution and distribution boundaries

The root MIT license covers this repository's adapter code, tests and original
documentation. It does **not** relicense third-party software, robot assets,
human models, motion datasets, or generated motion data.

- **UMR backend:** [longchengzhuo/Unified-Motion-Retargeting](https://github.com/longchengzhuo/Unified-Motion-Retargeting),
  commit `0aa1855fe4f65a73681ffbd1d9f95ab1c2bad9ca`, MIT. This is an
  **unofficial independent implementation**, not the original paper authors'
  official release. The unmodified Git submodule carries its own LICENSE.
- **ELF3 source:** [bxirobotics/bxi_controller_ros2](https://github.com/bxirobotics/bxi_controller_ros2),
  commit `1c9954040d114b3ff5e133b3611c5b327d19d029`. The fetcher retrieves the
  URDF and meshes directly from that revision. We do not distribute those
  assets or downloaded control-parameter source files here; review upstream
  terms before using or redistributing them.
- **AMASS and SMPL-X:** obtain data and body models separately from
  [AMASS](https://amass.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/)
  under their applicable terms. Neither raw inputs nor derived surfaces,
  correspondence artifacts, trajectories, recordings or weights are bundled.
  This project's MIT license does not grant permission to redistribute them.
  See the [data card](docs/DATASET.md) for the local bundle and its pending
  redistribution-permission review; the bundle itself is not uploaded.
- **Development origin:** extracted from the local ELF3 adaptation work in
  [scalebfm-loco](https://github.com/ChRis98Wang/scalebfm-loco), developed alongside
  [ScaleBFM](https://github.com/zengweishuai/ScaleBFM). ScaleTrack, ScaleBridge,
  IsaacLab, G1 policies and their training code are not copied into this repo.

Upstream projects and robot/paper authors do not endorse this adapter.
The historical `bfm.*` receipt schemas and dated module names are retained for
provenance compatibility, not to imply an official implementation.
