# 初始朝向退化的 source 恢复 v6

> 历史协议，保留当时的范围、版本与本地实验标识，不是当前执行状态或新用户安装路径。
> 独立仓库的现状以 [STATUS.md](STATUS.md) 为准；本次仅将机器专属路径改为环境相对说明。

范围：第一批固定 20 中 8 条 source 准备失败（6 个 crawl、Lie、Lie_to_crouch），原因均为首帧 pelvis local +Z 的世界 XY 投影小于 0.1。此版本仅恢复完整 source 准备，不改冻结 v5/core/queue/Stage I，不做 GPU、Stage I、Stage II、物理或训练替换。下一批新增的两个 `General_A9_-___Lie_(forward)` / `_t2` 只登记为未验证来源，不加入本轮固定 8。

## 本地轴约定与模型证据

冻结 core 的 `CANONICAL_ROTATION=[[0,0,1],[1,0,0],[0,1,0]]` 把 SMPL-X +Z anterior 映到机器人 +X forward，把 SMPL-X +X left 映到机器人 +Y left。不是将机器人 forward 轴误用到人体坐标。

安装模型实现依据：SMPL-X 环境内 `smplx/joint_names.py` 的前 3 项是 `pelvis,left_hip,right_hip`；`body_models.py` 的 SMPL-X `forward` 把 `global_orient` 放在 full_pose 的根关节，再传给 `lbs`，root mean 为零。准备时记录这两个代码文件的 SHA，实际推理当前性别/完整 betas 的零姿态模型，检查 `left_hip-right_hip` 向量与 +X 正对齐（cosine>0.8、两髋距离>5cm），并将真实数值和完整模型 SHA 写入 metadata；不只靠字符串推断左右。

## t0-only 固定朝向

令 R0 为第 0 个 50 Hz 采样的 pelvis 旋转，F=R0·(0,0,1)，L=R0·(1,0,0)，世界 up=(0,0,1)。

- 若 `norm(F_xy)>=0.1`，严格复用 v5 `Rz(-atan2(F_y,F_x))`，表达式及浮点运算顺序不变。
- 否则，用 `L×world_up=(L_y,-L_x,0)` 的方向定义参考 yaw，再做同样的单个 Rz 变换。R0 为正交矩阵，前方向近竖直时可证明 `norm(L_xy)>=sqrt(1-0.1²)`，因此侧轴水平投影稳定，不需寻找后续帧。

fallback 是**人体侧轴定义的坐标参考方向**，不是“机器人将朝此方向走”的任务目标。每条只在 t0 选一次，全部帧共用同一个全局 heading、同一个首 pelvis XY offset、同一 canonical material face/barycentric、同一完整时钟和全局 floor percentile。初始躺姿/爬姿、全部帧及帧间运动保留；不裁剪、换 clip、重置 chunk、插帧修姿或把源旋转扶正。

Rz 协变：若整条原始世界坐标先旋转任意固定 Z yaw，fallback 与正常分支的归一化结果应一致（浮点误差内）。单测覆盖该关系。阈值两侧跨不同 clip 的 yaw 约定**不保证数值连续**，这被明确披露；不会逐帧切换，不会产生本条动作内部的阈值跳变，也不通过平滑修改原动作。

## 接口与审计

新模块 `scripts/umr_heading_source_v6.py` 显式独立实现准备 orchestration，复用冻结 v5 的全长时钟/插值/batch partition/预算/严格数值检查及原子输出、core 的真实 mesh/canonical 采样。v5 SHA 固定为 `5a8a788ba9d42efb4db257e821554c9ba11a1363fdc4972d1792057e1aff7f23`。不修改原函数，不用 monkeypatch 或动态改写旧代码。

`prepare_source(source,body_model,output,*,fps=50.,points=4096,chunk_size=16,seed=0,limits=None)`；`load_prepared_source(path,*,limits=None)` 在全部 v5 校验外重算 v6 heading proof、首存储 pelvis 旋转与实际 canonical 左轴证据。扩展 proof schema 为 `bfm.smplx_t0_heading_reference/6`；原 interchange/canonical 身份字段不改变，追加 t0 轴/投影范数/分支/阈值、未来帧访问数 0、禁止逐帧切换等可审计字段。

## 固定验收与重要限制

读取 `local/umr_refresh_queue_v5_queue-prepare-20260912a/{inputs,status}.json` 和 8 个对应 prepare.log，严格固定原失败 indices `[1,2,9,10,11,16,18,19]` 与源 SHA。本轮 CPU 完整准备这 8 个源，另外使用已验正常 KIT 30.03 秒来源做 v5 数组逐值兼容对照。全部采样点从 t0 到原始结尾允许的最后 50 Hz 时刻；预算不足明确失败，不截短。

资源：有限 `bfm-umr-full-source-*` user service，CUDA 隐藏，runtime≤420秒、MemoryMax=6GiB、有限 stop/tasks、Restart=no、KillMode=control-group；原始与既有派生数据不覆盖。每条成功/失败保留收据及源/代码/协议 SHA，最终验收 MainPID=0、ControlGroup 空。

**v6 source 恢复不表示旧 Stage II 已接通。** 冻结 core 的 `initialize_root` 仍使用 pelvis anterior +Z 投影判断，单个全局 yaw 无法把竖直 anterior 变成水平，因此恢复来源依然需要未来独立版本的 root-yaw 初始化规则；本任务不接这一环节。更不把 source 可读、canonical 可复用或数据恢复数量说成重定向质量、物理跟踪或训练提升已通过。原始 AMASS/SMPL-X 和派生文件未获许可不得公开推送。
