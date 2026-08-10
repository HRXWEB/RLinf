# 单卡 A800 LIBERO 实战教程改写设计

## 目标

将 `_docs/13-LIBERO训练实战.md` 的主路径改写为一份可在全新租赁机上顺序执行的中文教程。目标机器固定为单张 A800 80GB、CUDA 12.4.1、Python 3.10.16、cuDNN 9 和 Ubuntu 22.04。读者应能从首次登录开始，完成 RLinf/OpenPI/LIBERO 安装、模型下载、W&B 配置、单卡 smoke test、正式训练、监控、恢复和评测。

本次只修改 `_docs/13-LIBERO训练实战.md`。不修改源码、示例 YAML、安装脚本或正式 Sphinx 文档。

## 内容方案

采用“单卡实战为主线、通用内容作为后半部分”的结构：

1. 标明目标硬件和核验边界。
2. 从全新机器开始检查 GPU、驱动、系统、CPU RAM、磁盘和共享内存。
3. 安装最小系统工具并 clone RLinf。
4. 通过 `requirements/install.sh` 创建原生 `.venv`，验证 Python、PyTorch、Ray、LIBERO 和 OpenPI。
5. 配置 Hugging Face 与 W&B；所有用户可变配置通过 Hydra CLI override 传入，不要求修改 YAML。
6. 下载并验证 π₀ LIBERO-Spatial SFT checkpoint。
7. 用 ASCII 和 Mermaid 解释 RLinf 与 HIL-SERL 的角色映射：HIL-SERL actor 约等于 RLinf 的 RolloutWorker 与 EnvWorker，HIL-SERL learner 约等于 RLinf ActorWorker。
8. 解释 `total_num_envs`、`rollout_epoch`、`algorithm.update_epoch` 和 `runner.max_steps` 的不同作用域。
9. 先展开 resolved config，再运行保守的单卡 1-step smoke test。
10. 按 Cluster/初始化、rollout、actor training、日志和 checkpoint 阶段验收。
11. 给出单卡正式训练起点、W&B 监控、checkpoint 恢复和独立评测命令。
12. 保留常见失败分流和 LIBERO-Pro/Plus 边界，但不让它们打断标准 LIBERO 主流程。

## 命令与配置约束

教程主路径使用原生 `.venv`：

```bash
bash requirements/install.sh embodied --model openpi --env libero
source .venv/bin/activate
```

W&B 使用 CLI override：

```text
'runner.logger.logger_backends=[wandb]'
```

smoke test 使用低峰值参数：

```text
CUDA_VISIBLE_DEVICES=0
env.train.total_num_envs=8
env.train.rollout_epoch=2
env.train.max_episode_steps=10
env.train.max_steps_per_rollout_epoch=10
actor.micro_batch_size=1
actor.global_batch_size=8
algorithm.update_epoch=1
runner.max_steps=1
runner.save_interval=-1
runner.val_check_interval=-1
```

正式训练的保守起点使用：

```text
env.train.total_num_envs=16
env.train.rollout_epoch=4
actor.micro_batch_size=1
actor.global_batch_size=256
algorithm.update_epoch=4
```

教程必须说明这些参数只是单卡起点，不是仓库已实测的 A800 性能保证。`global_batch_size` 必须能被 `micro_batch_size × actor_world_size` 整除。减少 `total_num_envs` 并增加 `rollout_epoch` 只能近似保持采集规模，不代表时间、样本组成或训练结果完全等价。

## 数据流说明

文档明确区分算法术语与 RLinf 类名：

```text
HIL-SERL actor    ≈ RLinf RolloutWorker + EnvWorker
HIL-SERL learner  ≈ RLinf ActorWorker
```

RLinf 同步具身 PPO 的主链为：

```text
ActorWorker 权重
  -> RolloutWorker 推理
  -> EnvWorker 执行动作并组装 trajectory
  -> ActorWorker 计算 advantage/loss 并更新参数
```

不得把 RLinf 的 RolloutWorker 描述成遍历 rollout buffer 并更新模型，也不得把三个 WorkerGroup 描述成同一个进程。

## 错误处理与安全边界

- 不在命令中写入 W&B 或 Hugging Face token；使用交互式登录或环境提供的 secret。
- 安装命令前说明其网络、磁盘和系统包副作用。
- 不宣称静态配置通过等于显存足够。
- 把 rollout OOM 与 actor OOM 分开排查。
- 明确 `max_steps=1` 仍可能运行较久，因为一个 global step 包含完整采样和更新。
- 保留 EGL/OSMesa、Ray、模型路径、checkpoint 层级和 W&B 离线模式的故障分流。

## 验证方案

完成修改后执行：

1. 检查教程中引用的脚本、配置、模型名和配置键均存在。
2. 用 Hydra `--cfg job --resolve` 静态验证命令结构；不在当前非目标 GPU 环境启动训练。
3. 检查所有相对 Markdown 链接存在。
4. 检查代码围栏和 Mermaid 围栏闭合。
5. 检查命令未包含真实凭据、占位 TODO 或行尾空格。
6. 核验 single-GPU batch 整除关系：`8 % (1 × 1) = 0` 与 `256 % (1 × 1) = 0`。
7. 对照 `rlinf/config.py`、`EnvWorker`、`MetricLogger`、安装脚本和 quickstart YAML 做 doc-to-code 检查。

## 完成标准

- 新用户可从一台初始 A800 租赁机开始逐条执行，不需要先阅读其他章节才能补齐关键命令。
- W&B 通过 CLI override 启用，YAML 不需要修改。
- smoke test 与正式训练命令明确分离。
- actor、rollout、env 的职责与 HIL-SERL 对照准确。
- `total_num_envs` 与 `rollout_epoch` 的并发/时间权衡有数字示例。
- 现有恢复、评测和 Pro/Plus 内容仍可使用，且与单卡主流程不冲突。
