# UMR → ELF3

[![CPU checks](https://github.com/ChRis98Wang/umr_elf3-/actions/workflows/ci.yml/badge.svg)](https://github.com/ChRis98Wang/umr_elf3-/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/adapter-MIT-blue.svg)](LICENSE)

将 AMASS **SMPL-X** 动作重定向到 **ELF3 31 自由度机器人**的实验性工具链。
包括人体表面准备、UMR 神经对应学习、逐帧 IK、全身轨迹优化、质量检查与可停止的批处理。

Experimental AMASS/SMPL-X → ELF3 retargeting with learned surface correspondence,
MuJoCo/Mink IK and model-based trajectory refinement. **Kinematic references,
not an ELF3 control policy or a completed ScaleBFM reproduction.**

> 依赖的是固定版本的[非官方 UMR 实现](https://github.com/longchengzhuo/Unified-Motion-Retargeting)。
> 当前穿地、自碰撞和优化收敛仍有问题。没有训练好的 ELF3 策略，不可直接部署实机。

## 当前能做什么

- 从原始 SMPL-X 网格采样人体表面，保留完整动作的 50 Hz 时钟与首帧朝向证据。
- 保留 ELF3 原始 31 个关节的名称、轴、限位与惯量；不套用 G1 的 29 自由度导出格式。
- 按实际人体形状与机器人几何校验并复用 Stage I 对应；Stage II 连续求解整条动作。
- 对根位姿和全部关节做三次 B 样条优化，检查末端漂移、关节速度、脚底高度与碰撞。
- 1–4 个有界后台任务、输入/结果 SHA-256 校验、断点续跑、失败隔离和整个进程组清理。
- 保留已有动作对比、MP4/GIF 导出与历史候选动作切换 GUI 工具。

## 真实进度：2026-09-17 停止快照

| 项目 | 数量 |
|---|---:|
| 本地去重动作计划（不是整个官方 AMASS） | 9,483 |
| 完成 UMR 重定向并归档 | 487 |
| 产生全身优化结果 | 465 |
| 通过当前运动学门槛，仍待复核 | 4 |
| 优化结果质量不合格，隔离 | 461 |
| 原始重定向完成、优化未完成 | 22 |
| 正式批准训练 / 已启动 ELF3 策略训练 | 0 / 否 |

队列已按要求停止；4 条待复核不是 4 条训练合格数据。
完整问题分类见[状态与限制](docs/STATUS.md)，可机器读取的汇总见
[batch_summary_20260917.json](docs/batch_summary_20260917.json)。

## 快速开始

Linux + systemd 用户服务。批处理当前要求 NVIDIA GPU（Stage I）和两个 Python 环境：
UMR Python 3.10、SMPL-X 准备 Python 3.11。已有环境可直接使用，**不需要 IsaacLab**。
依赖、模型准备和资源预算见[安装说明](docs/SETUP.md)。

```bash
git clone --recurse-submodules https://github.com/ChRis98Wang/umr_elf3-.git
cd umr_elf3-

# 换成自己的路径；不要将 AMASS、人体模型或生成数据提交到 Git。
UMR_PY=/absolute/path/to/umr-venv/bin/python
SMPLX_PY=/absolute/path/to/smplx-venv/bin/python
AMASS_ROOT=/absolute/path/to/licensed/amass
BODY_MODEL=/absolute/path/to/SMPLX_NEUTRAL_2020.npz

# 从固定的上游版本获取 ELF3；不执行下载的 Python 文件。
"$UMR_PY" scripts/elf3_umr_asset.py fetch --output local/elf3_assets
"$UMR_PY" scripts/elf3_umr_asset.py convert --assets local/elf3_assets --output local/elf3_model
"$UMR_PY" scripts/prepare_elf3_migration.py contract --assets local/elf3_assets --output local/contract

# 只生成清单和计划，不开始求解或训练。
"$UMR_PY" scripts/prepare_elf3_migration.py inventory \
  --source-root "$AMASS_ROOT" --body-model "$BODY_MODEL" --output local/inventory
"$UMR_PY" scripts/plan_elf3_batch.py \
  --inventory local/inventory/inventory.json --contract local/contract/joint_contract.json \
  --output local/plan

# 默认只打印命令，不启动。确认 missing_body_model_files=0 和资源后，加 --start。
"$UMR_PY" scripts/launch_elf3_library.py \
  --name library --batch-plan local/plan/batch_plan.json \
  --assets local/elf3_assets --robot-xml local/elf3_model/elf3.xml \
  --body-model "$BODY_MODEL" --prepare-python "$SMPLX_PY" \
  --output local/library --workers 4
```

实际启动时，在最后一条命令末尾加 `--start`。第一次建议只对一个小的、合法持有的
数据目录建立计划；全库优化通过率目前很低。

```bash
# 查看状态、日志；主动停止会清理该服务的子进程。
python3 -m json.tool local/library/status.json
journalctl --user -u bfm-umr-refresh-elf3-library.service -n 40 --no-pager
systemctl --user stop bfm-umr-refresh-elf3-library.service
```

续跑需原命令加 `--resume --start`，代码、输入路径、配方和校验值必须完全一致。
**不能用这个独立仓库直接续跑旧 ScaleBFM 工作区的冻结队列**；保留旧目录和运行器。

## 测试与可视化

```bash
"$UMR_PY" -m unittest discover -s tests -p 'test_*.py' -v

# 对已生成的原始轨迹做运动学录像，不启动策略、不进行物理推进。
MUJOCO_GL=egl "$UMR_PY" scripts/run_elf3_umr_trial.py preview \
  --output local/library/job_00000/attempt_000/raw
```

GUI 的候选套件格式与同帧对比说明见[工具与输出格式](docs/TOOLS.md)。
公开 CI 只测数值、合成几何和队列逻辑；需要私有 ELF3/AMASS 产物的测试会明确跳过，
不能把 CI 通过当作动作质量或物理跟踪验收。

## 技术文档与边界

- [技术路线与模型解释](docs/TECHNICAL_ROUTE.md)
- [安装、资源与依赖](docs/SETUP.md)
- [已知问题、当前数据与下一步](docs/STATUS.md)
- [独立打包验证记录](docs/RELEASE_VALIDATION.md)
- [工具、输出及复现边界](docs/TOOLS.md)
- [贡献与发布规则](CONTRIBUTING.md)

适配代码采用 MIT；第三方 UMR、ELF3 资产、AMASS、SMPL-X 分别遵循各自条款。
**本仓库不包含人体模型、动作数据、训练权重或机器人网格。** 详见 [NOTICE](NOTICE.md)。
