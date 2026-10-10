# 种子、终局 transition 与严格组 smoke 验证记录

## 本轮工程修正

- `auto_train_loop.py` 增加 `--seed`，统一固定 Python、NumPy 和 PyTorch（含 CUDA）随机源。
- 严格组中的每条轨迹根据 `(seed, epoch, episode_index, trajectory_index)` 派生稳定 32 位 seed，并写入 epoch JSONL。
- 普通训练 episode 结束时统一调用 `flush_pending_transition()`。
- `HGSAdaptedAgent` 增加终局 transition flush 接口，与 GraphGRPO 共用训练循环。

## GraphGRPO 普通训练 smoke

- 实验 ID：`graphgrpo_flush_seed_smoke_20261009`
- seed：`101`
- `brandimarte/mk02` 训练、`mk03` 验证，均 `completed`
- makespan：训练 `627`，验证 `1000`
- checkpoint `train_step=4`，策略损失和 GNN 更新均非零

## HGS-adapted smoke

- 实验 ID：`hgs_flush_seed_smoke_20261009`
- seed：`102`
- `brandimarte/mk02` 训练、`mk03` 验证，均 `completed`
- makespan：训练 `523`，验证 `1079`
- 训练和验证均在共享循环中完成终局 transition flush

## GraphGRPO 严格组 smoke

- 实验 ID：`graphgrpo_strict_seed_smoke_20261009`
- seed：`103`
- `brandimarte/mk02` 建立 1 个严格组，包含 2 条独立轨迹；两条轨迹 makespan 分别为 `575` 和 `619`
- checkpoint 诊断：`group_count=1`、`trajectory_count=2`、`normalized_group_count=1`、`nonzero_advantage_count=116`
- epoch 结果保存两条轨迹的派生 seed：`1918090959`、`30445992`

## 序列动作一致性修正后的 smoke

- 实验 ID：`graphgrpo_seqalign_smoke_20261009`
- 修正内容：序列头采样的首个工序现在就是实际优先级的首位；其余工序按当前 logits 完成排序，只有首位作为序列策略 transition。
- `brandimarte/mk02` 训练、`mk03` 验证均 `completed`，makespan 分别为 `530` 和 `1023`。
- 训练期间策略损失非零，未出现动作维度或环境终止异常。

## 解释边界

这些运行只验证随机性记录、终局 transition 收集和严格组训练链路。样本量、训练预算和 seed 数量都不足以支持论文中的方法优劣结论；正式实验仍需冻结数据划分后多 seed 重训、clean/uncertain 配对测试及消融实验。

## HGS-adapted 多 seed 小规模验收

使用相同的 `brandimarte/mk02` 训练、`mk03` 验证划分，分别以 seed `301`、`302`、`303` 运行 1 epoch，三次训练和验证均 `completed`。验证 makespan 分别为 `902`、`936`、`1000`，说明训练随机性已进入记录结果；这仍是 smoke 验收，不是正式性能比较。

以 seed `301` 完整重复训练一次，训练 gap、验证 makespan 和 `last.pt` SHA-256 均与第一次运行一致（验证 makespan `902`，SHA-256 为 `07140eec4134d9e8575f91c5d7cafb0a96c0a8fb72d572d510afc4e9887a0fd5`）。

## GraphGRPO 严格组多 seed 小规模验收

seed `401` 和 `402` 均完成 1 个严格组、2 条轨迹和 1 个验证 episode。两次 checkpoint 诊断均为 `trajectory_count=2`、`normalized_group_count=1`、`nonzero_advantage_count=116`；验证 makespan 分别为 `1164` 和 `1088`。这些结果确认严格组 seed 记录和训练更新链路可重复运行，但不构成性能优劣结论。
