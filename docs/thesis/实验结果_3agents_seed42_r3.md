# 三方法 3-run 配对统计记录

这是一次用于检验统计脚本和事件追踪的阶段性结果，尚未替代按训练/验证/测试划分完成的最终论文实验。

## 运行协议

- 数据集：`brandimarte/mk02`–`mk09`，8 个实例。
- 方法：`ORToolsAgent`、`GraphPPOAgent`、`GraphGRPOAgent`。
- 每个方法、每个实例、每个场景运行 3 次，共 72 条 clean 记录和 72 条 uncertain 记录。
- uncertain 场景：`default`；固定 `--seed 42`，每个 `(family, instance, run)` 派生并记录 trial seed。
- OR-Tools 固定 `--ortools-workers 1`、时间限制 30 秒；所有方法使用 CPU、process 模式 2 workers。
- clean 与 uncertain 使用相同实例、run 和 seed，并按四元组加 seed 严格配对。

## 配对结果

| 方法 | 配对数 | clean 均值 | uncertain 均值 | 差值均值 | 相对退化均值 | 相对退化标准差 | 正退化比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GraphGRPOAgent | 24 | 1206.875 | 1306.000 | 99.125 | 7.70% | 5.67% | 75.0% |
| GraphPPOAgent | 24 | 1096.500 | 1141.000 | 44.500 | 5.34% | 9.01% | 50.0% |
| ORToolsAgent | 24 | 1223.750 | 1298.125 | 74.375 | 5.29% | 5.48% | 62.5% |

所有 uncertain trial 均完成，事件执行率为 100%。每种方法共计划并执行：`AGV_FAIL` 18 次、`JOB_ADD` 21 次、`MACHINE_FAIL` 84 次。

## 结果文件

- [clean JSONL](../../benchmark_results/thesis_clean_3agents_seed42_r3/results.jsonl)
- [uncertain JSONL](../../benchmark_results/thesis_uncertain_3agents_seed42_r3/results.jsonl)
- [配对明细](../../benchmark_plots/thesis_clean_uncertain_3agents_seed42_r3.csv)
- [配对统计](../../benchmark_plots/thesis_clean_uncertain_3agents_seed42_r3_stats.csv)
- [状态统计](../../benchmark_plots/thesis_clean_uncertain_3agents_seed42_r3_status_summary.csv)
- [事件统计](../../benchmark_plots/thesis_clean_uncertain_3agents_seed42_r3_event_summary.csv)

## 解释边界

三次 run 的每个方法—实例结果相同，因为 benchmark 强制 inference 模式，GraphPPO/GraphGRPO 使用固定检查点并采用贪心动作；因此这不是三次独立策略采样。表中的标准差来自 8 个实例和 3 次重复后的记录，不能单独用来声称算法具有随机稳定性。该结果生成于 GNN 更新路径修复前，且使用旧模型检查点，只能作为流程验证的历史记录，不能作为修复后 GraphGRPO 的正式结论。
