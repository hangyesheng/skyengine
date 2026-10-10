# HGS-adapted 首轮验收记录

## 运行配置

- 日期：2026-10-09
- 实验 ID：`hgs_adapted_smoke_20261009`
- Agent：`HGSAdaptedAgent`
- 初始化：`--fresh`
- 设备：CPU
- 数据族：`brandimarte`
- 实例：`mk02`、`mk03`
- 划分：1 个训练实例、1 个验证实例
- 场景：clean

## 结果

| 阶段 | 实例 | 状态 | makespan | 步数 |
|---|---|---:|---:|---:|
| train | mk02 | completed | 492 | 492 |
| validation | mk03 | completed | 916 | — |

训练日志记录了非零 policy loss，模型 checkpoint 写入：

`training_logs/results/hgs_adapted_smoke_20261009/checkpoints/last.pt`

## 验收结论

HGS-adapted 已通过最小 `bootstrap → reset → step → terminated` 验收，三阶段 O-M-V 动作能够执行，训练和验证 episode 均正常结束，并且模型保存/加载回归测试通过。

该结果只证明适配层可运行，不代表 HGS 已经在 PacketFactory 上优于其他方法。正式纳入 benchmark 前仍需完成多 seed、统一训练预算、与 GraphGRPO 相同的测试实例和 uncertain 配对实验。
