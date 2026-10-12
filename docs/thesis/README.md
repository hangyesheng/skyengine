# 毕业论文资料索引

论文正文采用以下概念名称：基础场景、动态事件场景、机器故障场景、AGV 故障场景和新作业到达场景。实现名称、实验目录和结构化记录格式只用于证据追踪，不直接写入正文叙述。

## 正文草稿

- [论文终稿草案 v1](论文终稿草案_v1.md)：当前冻结结果对应的六章统一终稿草案。
- [论文终稿格式检查清单 v1](论文终稿格式检查清单_v1.md)：内容冻结后的格式与提交前检查项。
- [第 1—6 章统一整合稿 v2](论文整合稿_v2.md)：历史整合版本，保留用于回溯。
- [第 1—6 章统一整合稿 v1](论文整合稿_v1.md)：历史整合版本。

- [第 1 章：绪论](chapters/论文第1章_绪论草稿.md)
- [第 2 章：相关工作](chapters/论文第2章_相关工作草稿.md)
- [第 3 章：平台设计与实现](chapters/论文第3章_平台设计与实现草稿.md)
- [第 4 章：基于关系图与组相对优势的加工—物流协同调度方法](chapters/论文第4章_GraphGRPO协同调度方法草稿.md)
- [第 5 章：实验设计与结果分析](chapters/论文第5章_实验设计与结果分析草稿.md)
- [第 6 章：结论与展望](chapters/论文第6章_结论与展望草稿.md)

正文主线是：统一表示加工资源、运输资源和动态事件，构建动态图状态与工序—机器—AGV 三阶段决策，并以总完工时间及配对性能退化评价本文方法。主对比方法为 OR-Tools、HGS 适配实现和本文方法；早期图策略实现只用于受控消融或附加分析。

## 写作与实验管理

- [论文推进任务清单](论文推进任务清单.md)：当前工作状态、前置条件和提交前检查。
- [毕业论文写作计划](毕业论文写作计划.md)：研究定位、章节安排和实验协议素材。
- [论文术语规范与正文重构计划](论文术语规范与正文重构计划.md)：工程术语到正文术语的映射。
- [论文研究问题与研究定位梳理](论文研究问题与研究定位梳理.md)：问题边界、贡献边界和可检验假设。
- [正式实验协议](正式实验协议_v1.md)：实例划分、随机性、配对规则和消融因素。
- [正式训练与消融协议 v2](正式训练与消融协议_v2.md)：最终训练轮数、每轮验证、严格组样本量和结果冻结规则。
- [本文方法优化与外部复现计划](GraphGRPO算法优化与外部复现计划.md)：组语义、奖励、实现修正和第二个外部方法验收顺序。
- [奖励模式公平筛选记录](奖励模式公平筛选记录_20261012.md)：目标一致奖励、复合奖励和严格组短跑的统一验证证据。

## 文献材料

- [唯一文献表与 DOI 核验](literature/唯一文献表_DOI核验.md)
- [文献结构化比较与综述证据](literature/文献结构化比较与综述证据.md)
- [论文相关工作与算法对比矩阵](literature/论文相关工作与算法对比矩阵.md)
- [Zotero 文献矩阵](literature/文献矩阵_Zotero.md)
- [HGS 复现卡](literature/reproduction/HGS_复现卡.md)
- [Wang 等（2024）QMIX 复现卡](literature/reproduction/Wang_2024_QMIX_复现卡.md)

## 实验证据

- [正式结果归档](实验验证记录_正式首轮结果.md)：正式方法对照、消融和统计限制；文件名保留历史名称以避免破坏已有引用。
- [正式实验证据清单](formal_artifact_manifest_v1.json)：检查点、测试结果、事件文件和源文件哈希。
- [保留测试实例验收记录](保留测试实例验收记录_20261012.md)：dauzere/01a 跨数据族链路证据，不进入主表。
- `../../benchmark_plots/formal_main_frozen_comparison_v1.csv`：冻结奖励配置下主对比的完成率、完工时间、配对退化和事件执行率。
- `../../benchmark_plots/formal_main_frozen_instance_v1.csv`：按测试实例汇总的基础场景与动态事件场景结果。
- `../../benchmark_plots/formal_main_frozen_paired_comparison_v1.csv`：本文方法、HGS 适配实现和 OR-Tools 的配对比较结果。
- `../../benchmark_plots/formal_main_frozen_runtime_v1.csv`：平均决策时间、总运行时间和环境步数。
- `../../benchmark_plots/formal_test_instance_metadata_v1.csv`：测试实例规模、运输参数、计划事件数和实际新增工作量。
- `../../benchmark_plots/formal_all_methods_summary_v3.csv`：主对比与附加消融的统一结果（严格组当前为折扣修正前历史结果，不能用于正式因果比较）。
- `../../benchmark_plots/formal_graphgrpo_frozen_machine_only_seed601_603_pair_v1_stats.csv`、`formal_graphgrpo_frozen_agv_only_seed601_603_pair_v1_stats.csv` 和 `formal_graphgrpo_frozen_job_add_only_seed601_603_pair_v1_stats.csv`：冻结主方法检查点下的三类专项事件结果。
- `../../benchmark_plots/formal_ablation_frozen_comparison_v1.csv`：冻结奖励下固定图编码器、去关系图状态、时间增量代价和固定 AGV 选择四项单因素消融的统一比较。
- `../../benchmark_plots/formal_ablation_frozen_paired_statistics_v1.csv` 和 `formal_ablation_frozen_paired_comparisons_v1.csv`：新版消融的配对区间与相对于本文方法的描述性比较。

实验归档可保留实现名称，以便从论文表格回溯到具体模型和原始记录；论文正文只引用经过整理的实验协议、方法名称和统计结果。
