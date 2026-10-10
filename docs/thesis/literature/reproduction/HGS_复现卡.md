# HGS（Moon et al., 2024）复现卡

## 1. 文献身份与证据

- 论文：Moon, Sihoon; Lee, Sanghoon; Park, Kyung-Joon. *Learning-enabled Flexible Job-shop Scheduling for Scalable Smart Manufacturing*。
- 期刊版本：Journal of Manufacturing Systems, 77, 356--367, 2024。
- DOI：`10.1016/j.jmsy.2024.09.011`。
- 公开预印本：<https://arxiv.org/abs/2402.08979>。
- 本次核验日期：2026-10-09。
- 方法名称：Heterogeneous Graph Scheduler（HGS）。
- 证据范围：下载 arXiv 预印本及其源码包，依据正文的 MDP、encoder、decoder 和 RL algorithm 小节整理。原论文的 benchmark 数值不直接并入 PacketFactory 结果。

## 2. 原论文中已核验的算法语义

### 问题

HGS 处理带运输约束的柔性作业车间调度（FJSPT）。每个决策动作同时选择一个可执行工序、兼容机器和可用车辆，目标是最小化 makespan。图节点包括 operation、machine 和 vehicle；边表示工序—机器加工时间、工序—车辆空载运输时间以及机器之间的有载运输时间。

### 状态

状态是随决策时刻变化的异构图。正文给出的节点特征包括：

- operation：当前可用机器数、可用车辆数、加工时间或兼容机器平均加工时间、作业剩余工序数、作业完成时间估计和工序开始时间估计；
- machine：邻接工序数、可用时间、机器位置；
- vehicle：邻接工序数、可用时间、当前位置和载荷/空载相关状态；
- 边：加工时间、空载运输时间和有载运输时间。

HGS 的结构感知编码器由 operation、machine、vehicle 三个局部子编码器和一个全局编码器构成。局部编码器使用带边特征的异构多头注意力，避免将不相关的加工和运输关系混在同一个邻域中。

### 动作

复合动作为：

\[
a_t=(O_{ij},M_k,V_u),
\]

即工序排序、机器分配和车辆分配在一个决策时刻内依次完成。论文明确使用动态可行性掩码：工序前置工序未完成、机器忙碌、车辆不可用的候选都不能被选中。解码器依次执行 operation、machine、vehicle 三个选择阶段，并使用上一阶段的上下文和边嵌入影响下一阶段。

### 奖励与训练

论文定义状态下的 makespan 估计为未调度工序完成时间下界的最大值：

\[
C_{\max}(s_t)=\max_{i,j} C_{LB}(O_{ij},s_t),
\]

奖励是相邻状态下界之差：

\[
r(s_t,a_t,s_{t+1})=C_{\max}(s_t)-C_{\max}(s_{t+1}).
\]

在 \(\gamma=1\) 时，完整轨迹回报等于初始下界减去最终 makespan，因此最大化回报与最小化 makespan 等价。训练算法是基于完整轨迹回报的 REINFORCE，而不是 PPO 或 GRPO；策略参数包含异构图 encoder 和三阶段 decoder。

## 3. PacketFactory 适配方案

本文实现的首个外部方法应命名为 **HGS-adapted** 或“基于 HGS 的 PacketFactory 适配实现”，不能直接称作原论文 HGS 复现。适配表如下：

| 项目 | HGS 原设定 | PacketFactory 适配设定 | 影响 |
|---|---|---|---|
| 节点 | operation / machine / vehicle | Operation / Machine / AGV，并保留 Job 节点 | 增加 Job 节点，需做类型投影 |
| 图边 | 加工、空载运输、有载运输边 | 当前工厂图中的工序—机器、工序前后、AGV—机器和作业—工序关系 | 边语义不完全相同 |
| 动作 | 单个 O-M-V 复合动作 | 复用统一 Agent 接口，按工序、机器、AGV 三阶段产生动作 | 决策接口可对齐 |
| 候选 | HGS 的可用机器/车辆邻域 | PacketFactory 当前时刻的 READY 工序、可用机器和 READY AGV | 需要统一动作掩码 |
| 运输 | 论文区分空载/有载运输时间 | 使用 PacketFactory 路网最短路径和 AGV 速度 | 运输模型发生适配 |
| 奖励 | makespan 下界差分 | 首版使用 HGS 风格下界差分；不加入 GraphGRPO 的利用率/队列塑形项 | 保持目标语义可解释 |
| 训练 | REINFORCE，完整轨迹回报 | 先实现独立 `HGSAdaptedAgent`；训练和推理均记录 seed、checkpoint 和配置 | 不与 GraphGRPO 的 GRPO 更新混用 |
| 动态事件 | 原论文重点是规模泛化 | 初始验收只用 clean；通过后再评估固定 uncertain 事件 | 动态结果单独归档 |

## 4. 实现边界与验收标准

首版适配实现必须满足：

1. `bootstrap → reset → step → terminated` 在 `brandimarte/mk02` 上正常完成；
2. 每个动作都能映射为 `(Operation, AGV, Machine)`，不产生无候选、重复占用或越过工艺前置约束的动作；
3. 记录每一步的 O-M-V 候选数量、动作掩码和奖励，便于与 GraphGRPO 对照；
4. 同一实例、同一 seed 下重复运行结果可复现；
5. 与 GraphGRPO 使用相同的 PacketFactory 终止条件、事件文件、超时和 `C_{\max}` 统计；
6. 只有通过上述验收后，才进入统一 benchmark；失败或超时记录为方法状态，不从均值中静默删除。

当前状态：已完成文献正文和源码包核验、复现卡和适配边界定义；`HGSAdaptedAgent` 首版已经实现，并在 `brandimarte/mk02` 上完成 clean 训练模式的动作可执行性、正常终止和 checkpoint 保存/加载验收。它尚未通过多 seed 和统一 benchmark 验收，因此目前仍不把 HGS 写入正式对比结果。

## 5. 与本文主方法的公平比较

HGS-adapted 的结果用于回答“与问题结构相近的图调度方法在同一环境中的表现如何”，不用于声称复现原论文的原始数值。正式表格必须同时报告：

- 方法名称和适配标记；
- clean/uncertain 的 `C_{\max}`、完成率和决策时间；
- HGS 原设定与 PacketFactory 的差异；
- 模型参数量、训练预算和随机种子；
- 事件计划数与实际执行数。

