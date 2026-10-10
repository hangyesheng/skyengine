# 三方法 clean/uncertain 配对验证记录

本记录用于确认正式实验协议和结果追踪链路已经可运行，不替代多 run 正式结果。

## 协议

- 数据集：`brandimarte/mk02`–`mk09`，共 8 个基础实例。
- 方法：`ORToolsAgent`、`GraphPPOAgent`、`GraphGRPOAgent`。
- clean 与 uncertain 使用相同实例、相同 `run=1` 和相同派生 trial seed。
- uncertain 场景：`default`；事件文件来自 `dataset/uncertain-events/default/brandimarte/`。
- benchmark：`--seed 42 --workers 2 --parallel-mode process --ortools-workers 1 --device cpu`。
- 每种方法每个实例运行 1 次；模型均为现有检查点，benchmark 使用 inference 模式。

## 结果摘要

| 方法 | 配对数 | clean 均值 | uncertain 均值 | 平均差值 | 平均相对退化 | 正退化比例 | 事件执行率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GraphGRPOAgent | 8 | 1206.875 | 1306.000 | 99.125 | 7.70% | 75.0% | 100% |
| GraphPPOAgent | 8 | 1096.500 | 1141.000 | 44.500 | 5.34% | 50.0% | 100% |
| ORToolsAgent | 8 | 1223.750 | 1298.125 | 74.375 | 5.29% | 62.5% | 100% |

事件类型汇总为：`AGV_FAIL` 6 次、`JOB_ADD` 7 次、`MACHINE_FAIL` 28 次；三种方法的计划数和实际执行数均一致。

## 结果文件

- [clean JSONL](../../benchmark_results/thesis_clean_3agents_seed42_r1/results.jsonl)
- [uncertain JSONL](../../benchmark_results/thesis_uncertain_3agents_seed42_r1/results.jsonl)
- [配对明细 CSV](../../benchmark_plots/thesis_clean_uncertain_3agents_seed42_r1.csv)
- [配对统计 CSV](../../benchmark_plots/thesis_clean_uncertain_3agents_seed42_r1_stats.csv)
- [状态统计 CSV](../../benchmark_plots/thesis_clean_uncertain_3agents_seed42_r1_status_summary.csv)
- [事件统计 CSV](../../benchmark_plots/thesis_clean_uncertain_3agents_seed42_r1_event_summary.csv)

## 解释边界

该记录只有每个方法每个实例一次运行，不能支持显著性检验或稳定性结论。GraphPPO 和 GraphGRPO 以固定检查点进行推理，当前 benchmark 强制 inference 模式。该记录生成于 GraphGRPO GNN 更新路径修复前，正式论文结果需要使用新代码重新训练，并增加独立训练/验证/测试划分、多 run 或多 checkpoint。
