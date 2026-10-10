# GraphGRPO 修复后短训练验证记录

## 运行信息

- 日期：2026-10-09
- 实验 ID：`graphgrpo_fresh_smoke_20261009_v2`
- Agent：`GraphGRPOAgent`
- 初始化：`--fresh`，未加载共享旧模型
- 设备：CPU
- 数据族：`brandimarte`
- 过滤实例：`mk02`、`mk03`
- 划分：1 个训练实例、1 个验证实例，`val_ratio=0.5`
- 训练轮数：1 epoch，训练 episode 1，验证 episode 1
- 训练调用：每 1 个环境步调用一次，GraphGRPO `rollout_steps=128`
- 事件：clean 场景

## 命令

```bash
uv run python scripts/auto_train_loop.py \
  --agent GraphGRPOAgent \
  --families brandimarte \
  --instances mk02 mk03 \
  --fresh --device cpu \
  --epoch-size 1 --max-epochs 1 \
  --train-interval 1 --val-interval 1 --val-sample-size 0 \
  --val-ratio 0.5 \
  --episode-timeout 45 --val-episode-timeout 45 \
  --no-progress-limit 30 \
  --experiment-id graphgrpo_fresh_smoke_20261009
```

## 结果

| 阶段 | 实例 | 状态 | makespan | 步数 |
|---|---|---|---:|---:|
| train | mk02 | completed | 499 | 500 |
| validation | mk03 | completed | 956 | 957 |

训练期间触发了 3 次 rollout 更新；策略损失日志非零。另用 `rollout_steps=8` 的短测比较训练前后参数，确认 GNN 参数发生变化。检查点写入：

`training_logs/results/graphgrpo_fresh_smoke_20261009_v2/checkpoints/last.pt`

## 解释边界

该运行只验证训练—验证闭环、GNN 梯度路径、实验目录 checkpoint 和小实例终止条件，不支持任何算法性能结论。训练集和验证集只有各 1 个实例，训练仅 1 epoch，且使用 clean 场景。正式实验必须重新划分数据集，增加训练 seed、epoch 和测试实例，并运行 uncertain 配对与消融实验。

此外，严格组采样的 buffer 语义已经通过合成数据测试，训练循环也已接入同一基础实例的多个独立环境；`graphgrpo_strict_seed_smoke_20261009` 已验证 1 组 2 条轨迹和非零组相对优势。本文仍将严格组版本与默认决策批次分组变体分开报告。
