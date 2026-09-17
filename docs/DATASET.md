# ELF3 重定向数据卡

## 发布状态

**已本地打包并校验，尚未公开上传数据。** 本页不是下载链接，也不是数据许可。
适配代码的 MIT 许可证不覆盖 AMASS/SMPL-X 及其派生结果。

AMASS 官网条款要求对第三方提供数据事先取得书面许可，并包含不分发条款。
目前没有确认这些派生机器人轨迹具有独立的公开再分发授权，因此先保留本地包，
等待权利方许可或明确覆盖该派生输出的条款确认。这是当前的发布决定，
不是对所有重定向输出版权性质的最终法律判断。
[AMASS 许可原文](https://amass.is.tue.mpg.de/license.html)。

人体模型及相关材料另受
[SMPL-X 模型许可](https://smpl-x.is.tue.mpg.de/modellicense.html)约束，未打入本数据包。
获得公开分发权限后，仍应附上适用的数据许可证，不应标成 MIT 数据。

## 本地包内容

统计来自 2026-09-17 已停止的原批次，只包含有正式归档收据的结果；
不包括中断作业或本次重复使用的实跑样本。

| 内容 | 数量 |
|---|---:|
| 不同源动作 | 487 |
| 原始 Stage II 轨迹 | 487 |
| 全身优化候选轨迹 | 465 |
| 完整轨迹变体总数 | 952 |
| 原始动作总帧数，50 Hz | 186,735 |
| 优化结果隔离 | 461 |
| 运动学待复核 | 4 |
| 原始重定向完成、优化未完成 | 22 |
| 批准用于训练 | 0 |

数据集来源计数：BMLrub 172、BMLmovi 126、ACCAD 22、KIT 167。
“计算完成”仅表示生成完整轨迹，不表示自然协调、可物理跟踪或可以安全训练。

本地 ZIP 文件名：`elf3_robot_motions_NOT_TRAINING_APPROVED.zip`。
大小 **98,739,431 bytes（94.17 MiB）**，SHA-256：

```text
1e5cbac34e552f4ed9eaaa2b579f15bd05b3fe97bf419713483cbc39e5e85675
```

已对全部 952 个轨迹成员重读并校验 SHA-256，ZIP CRC 检查通过。
该哈希对应这一份本地包；重新生成 ZIP 的封装时间戳可能不同，包哈希也可能不同。

## 文件布局

```text
manifest.json
DATA_NOTICE.txt
raw/job_00000.npz
...
optimized/kinematic_review_required/job_00004.npz
optimized/quarantined_quality/job_00000.npz
...
```

`manifest.json` 记录源动作 ID/hash、原 train/validation 划分、质量状态、
各变体 hash 和保留的关键指标。所有条目 `training_approved=false`。
优化失败/质量不合格没有被换名为“通过”；raw 与 optimized 不应混为一批。

每个 NPZ 只保留以下机器人状态数组，使用 `allow_pickle=False` 加载：

| 字段 | 含义 |
|---|---|
| `qpos` | `[N,38]`：根位置 xyz、根四元数 wxyz、31 个关节角度 |
| `times` | `[N]`，从 0 开始，50 Hz，保持源完整采样时钟 |
| `fps` | 50 |
| `dof_names` | 31 个关节名称；按名称映射，不套 G1 顺序 |
| `root_body` | `torso_link` |
| `quaternion_order` | `wxyz` |

位置单位米、角度单位弧度。导出保持以上原有数组数值与精度，不做平滑、裁剪、
地面平移或重采样。人体表面索引、点误差、模型、几何、绝对路径及完整日志不在 NPZ 内。
IK/优化关键指标通过 manifest 的允许字段保留，而不是复制含本机路径的完整收据。

## 自己生成本地包

```bash
# 使用 README 中配置好的 UMR_PY；这里的源必须是完整 library 输出。
"$UMR_PY" scripts/package_elf3_motions.py \
  --run /absolute/path/to/library \
  --output local/exports/research_bundle --selection all

# 只取进入运动学待复核的来源；仍然不是训练合格集。
"$UMR_PY" scripts/package_elf3_motions.py \
  --run /absolute/path/to/library \
  --output local/exports/review_bundle --selection review-only
```

打包器验证结果指针、源身份、轨迹/收据 hash、完整时间轴与 31 关节布局。
它不替代质量评估、不修改原始队列、不执行上传；默认标记再分发权限未确认。
输出目录不可覆盖，失败会留下 `INCOMPLETE.json`；没有成功的 `summary.json`
就不能把部分 ZIP 当完成品。
