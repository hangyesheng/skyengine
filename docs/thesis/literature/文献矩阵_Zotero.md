# Zotero 文献矩阵与结构化笔记

> 本文件是从原始 Zotero 整理过程提取的文献矩阵快照，包含 33 行原始比较记录、30 条结构化笔记和 8 篇重点文献证据摘要。它是综述素材库，不是已经完成全文核验的论文正文；字段写为“未说明”或“待核验”时，不得在论文中直接当作事实引用。
>
> 书目元数据以当前 [Zotero 文库](zotero_literature.bib) 和 [`唯一文献表_DOI核验.md`](唯一文献表_DOI核验.md) 为准；重复记录和 DOI 状态见后者。

## 整理说明

本次仅提取原始矩阵、逐篇笔记和重点证据摘要，保留其顺序和研究字段。原输出的“30 篇”同时用于矩阵和笔记，实际范围不同：

| 材料 | 原始记录数 | 按标准化标题去重后 | 重复位置 |
|---|---:|---:|---|
| 比较矩阵 | 33 行 | 30 个唯一标题 | 数据行 11/19、14/26、16/29 |
| 逐篇笔记 | 30 条 | 28 个唯一标题 | 笔记 11/19、14/26 |

“数据行”从表头之后的第一篇文献起计数。笔记 29 是多 AGV 管理综述，并非 MachineRank 文献；矩阵行号与笔记编号不能互换。矩阵另有工业组合优化 RL 综述和 JSSP RL 环境两篇，未出现在逐篇笔记中。

引用前请结合 [DOI 与元数据台账](唯一文献表_DOI核验.md) 及 [证据审查](文献结构化比较与综述证据.md) 核对。

## 一、原始文献比较矩阵（33 行记录）

> DOI 仅在当前 Zotero 元数据或正文证据中能够明确核验时列出；否则标记为“当前证据未显示”。“是/否”只表示文献明确报告的内容，不根据标题推断。
>
| 文献 | 研究对象 | 是否包含 AGV | 是否包含加工-物流协同 | 是否包含动态事件 | 是否包含机器故障 | 是否包含 AGV 故障 | 是否包含插单 | 是否使用图模型 | 是否使用强化学习 | 是否使用 PPO/GRPO | 主目标 | 数据集/环境 | 主要基线 | 与本文的主要差异 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| *Dynamic integrated scheduling of machines and AGVs in flexible job shop considering AGV failures* | 动态 FJSP-AGV | 是 | 是 | 是 | 未说明 | 是 | 未说明 | 未说明 | 未说明 | 未说明 | $C_{\max}$及鲁棒性 | 动态 FJSP-AGV 仿真/实例 | 未说明 | 聚焦 AGV 故障；未核验包含机器故障、JOB_ADD、GNN 或 RL |
| *Dynamic Integrated Scheduling of Production Equipment and Automated Guided Vehicles in a Flexible Job Shop Based on Deep Reinforcement Learning* | 动态 FJSP-AGV | 是 | 是 | 是 | 未说明 | 未说明 | 是，紧急订单插入 | 未说明 | 是，MARL | QMIX；非 PPO/GRPO | $C_{\max}$和能耗 | 案例与仿真 | 典型优化算法 | 本文唯一主目标为 $C_{\max}$，并使用 $\Delta C_{\max}$、$D_{C_{\max}}$ |
| *Deep Reinforcement Learning for Dynamic Flexible Job Shop Scheduling with Random Job Arrival* | 动态 FJSP | 运输资源存在，但 AGV 细节未说明 | 部分 | 是 | 未说明 | 未说明 | 是，随机到达 | 未说明 | 是，DRL | PPO 相关实现信息未完整核验 | $C_{\max}$ | 动态随机到达环境；初始作业和后续作业规模在正文中给出 | MILP/规则方法，具体需核验 | 本文进一步显式建模多 AGV、路网及多类故障 |
| *Solving Flexible Job-Shop Scheduling Problem with Heterogeneous Graph Neural Network Based on Relation and Deep Reinforcement Learning* | FJSP | 否 | 否 | 未说明 | 否 | 否 | 否 | 是，关系异构图 | 是 | PPO | makespan/调度效率 | 4 个公开基准数据集 | 4 种 DRL、3 种规则算法 | 只有工序—机器关系，未包含 AGV、运输网络或动态事件；DOI：10.3390/machines12080584 |
| *Learning-enabled flexible job-shop scheduling for scalable smart manufacturing* | AGV-based FJSP/FJSPT | 是 | 是 | 主要研究规模变化 | 未说明 | 未说明 | 未说明 | 图结构/节点编码，具体图模型未说明 | 是，DRL | 未说明 | 调度质量与规模泛化 | AGV 智能制造环境，具体数据划分需核验 | 未说明 | 与本文均涉及 AGV 和图状态，但本文重点是动态故障、JOB_ADD 和在线联合动作 |
| *Flexible Job Shop Scheduling Optimization with Machine and AGV Integration Based on Improved NSGA-II* | FJSP-AGV | 是 | 是 | 否或未说明 | 否 | 否 | 否 | 否 | 否 | 否 | 多目标：$C_{\max}$、机器负载、能耗 | 8 台机器、2 台 AGV、14 类活塞环工件的车间案例 | NSGA-II 及改进算法 | 本文不是多目标优化，也不使用 NSGA-II；本文动态事件更丰富 |
| *Multi-Agent Reinforcement Learning for Job Shop Scheduling in Dynamic Environments* | 动态 JSSP | AGV 未说明 | 未说明 | 是 | 未说明 | 未说明 | 订单扰动，具体类型需核验 | 未说明 | 是，MARL | 未说明 | 多目标调度，具体主目标需核验 | 动态制造环境 | 未说明 | 非明确 FJSP-AGV 协同研究 |
| *Multi-faceted dynamic flexible job shop scheduling via heterogeneous graph transformer and deep reinforcement learning* | 动态 FJSP | AGV 未说明 | 未说明 | 是 | 包含在动态事件参数中，但具体故障类型需核验 | 未说明 | 动态作业/资源事件，具体需核验 | 是，异构图 Transformer | 是，DRL | PPO 信息未完整核验 | $C_{\max}$或调度性能，具体目标需核验 | 模拟生成 MDFJSSP 数据集；加工时间、故障持续时间和事件参数随机生成 | 规则和 DRL 方法，具体列表需核验 | 与本文均使用动态事件和图状态，但本文包含加工—AGV 联合动作；DOI：10.1016/j.eswa.2025.130532 |
| *Digital Twin-Driven Adaptive Scheduling for Flexible Job Shops* | 数字孪生 FJSP | 未说明 | 未说明 | 是 | 机器故障或环境扰动 | 未说明 | 随机到达作业 | 未说明 | 是，RL 增强遗传算法 | 未说明 | 调度质量和适应性 | 数字孪生车间、随机到达仿真和实验室验证 | 遗传算法 | 本文显式建模 AGV、路网和多资源联合动作；DOI：10.3390/su14095340 |
| *Dynamic scheduling of flexible job-shop considering automatic guided vehicle transportation via deep reinforcement learning* | 动态 FJSP-AGV | 是 | 是 | 是 | 未说明 | AGV 状态变化；故障类型需核验 | 未说明 | 未说明 | 是，DRL | 未说明 | 完工率、机器利用率、AGV 利用率；$C_{\max}$是否主目标需核验 | 动态 FJSP-AGV 仿真 | 未说明 | 与本文较接近，但当前证据未确认其事件集合、联合动作和评价协议全部一致 |
| *A Dynamic Scheduling Method Combining Iterative Optimization and Deep Reinforcement Learning to Solve Sudden Disturbance Events in a Flexible Manufacturing Process* | 动态 FMS/FJSP | 未说明 | 未说明 | 是 | 是 | 未说明 | 订单变化 | 未说明 | 是，DRL | 未说明 | 动态调度质量、响应性和稳定性 | 事件驱动与周期驱动混合重调度环境 | 迭代优化和 DRL 对比 | 本文将机器故障、AGV 故障和 JOB_ADD 统一建模，并以 $C_{\max}$ 为唯一主目标 |
| *Flexible job shop scheduling via deep reinforcement learning with meta-path-based heterogeneous graph neural network* | FJSP | 否 | 否 | 未说明 | 否 | 否 | 否 | 是，元路径异构图 | 是 | GIN-A3C、GIN-PPO、GAT-PPO 等 | makespan/相对最优差距 | Brandimarte 等公开基准 | Rule 4、IPSO、SLGA、GIN-A3C、GIN-PPO、GAT-PPO | 图和 DRL 方法相关，但无 AGV 和动态扰动；DOI：10.1016/j.knosys.2024.111940 |
| *A Tree neural network deep reinforcement learning for flexible job shop scheduling with transportation constraints* | FJSP-T | 运输约束存在；AGV 类型未明确 | 是，至少考虑运输约束 | 未说明 | 未说明 | 未说明 | 未说明 | 是，运输树/HSTT | 是，DRL | 未说明 | makespan | FJSP-T 仿真/基准，规模需核验 | 规则、元启发式和 DRL 方法，具体需核验 | 与本文均有加工—运输耦合，但本文明确包含多 AGV、路网和动态故障 |
| *Flexible Manufacturing Systems intralogistics: Dynamic optimization of AGVs and tool sharing using Colored-Timed Petri Nets and actor–critic RL with actions masking* | FMS 内物流 | 是 | 是，AGV 与工具共享 | 是，动态物流状态 | 未说明 | 未说明 | 未说明 | Colored-Timed Petri Net，不是 GNN | 是，actor–critic RL | 非 PPO/GRPO；动作掩码 | 物流效率、资源共享和调度质量 | FMS 内物流仿真 | 未说明 | 对 AGV 状态、资源共享和动作掩码有参考价值，但加工工序联合决策和 $C_{\max}$ 设定不同 |
| *Dynamic Scheduling for Flexible Manufacturing Systems Based on Multi-Agent Deep Reinforcement Learning and Petri Nets* | 动态 FMS | 运输资源是否为 AGV 未说明 | 未说明 | 是：新订单、订单取消、机器故障 | 是 | 未说明 | 是，新订单到达 | Petri 网，不是 GNN | 是，MARL | 未说明 | 动态调度性能，主目标需核验 | FMS 离散事件/Petri 网环境 | 未说明 | 动态事件覆盖较好，但当前证据未确认 AGV 和加工—运输联合决策 |
| *Dynamic scheduling for flexible job shop based on MachineRank algorithm and reinforcement learning* | 动态 FJSP-AGV | 是，考虑 AGV 状态 | 部分；机器、工序和 AGV 状态共同进入调度 | 是 | 是 | AGV 状态，是否故障需核验 | 是，新作业插入 | 未说明 | 是，D3QN/RL | D3QN；非 PPO/GRPO | 机器利用率、调度性能；$C_{\max}$需核验 | 动态 FJSP 仿真 | MachineRank 规则、组合调度规则 | 与本文事件集合接近，但本文使用图状态和三阶段联合动作；当前证据支持新作业、机器故障、加工时间变化和 AGV 状态  |
| *Learning to schedule job-shop problems: Representation and policy learning using graph neural network and reinforcement learning* | JSSP | 否 | 否 | 否或未说明 | 否 | 否 | 否 | 是，GNN/析取图 | 是，RL | 未说明 | makespan | 随机生成 JSSP；最大测试规模为 20 台机器、100 个作业 | 传统启发式/规则，具体需核验 | 图状态和策略学习可借鉴，但不包含 FJSP、AGV 或动态事件；最大规模证据见  |
| *Integrated Scheduling of Multi-Objective Job Shops and Material Handling Robots with Reinforcement Learning Guided Meta-Heuristics* | JSSP-MHR | 机器人/MHR，未明确为 AGV | 是 | 否或未说明 | 否 | 否 | 否 | 未说明 | 是，用于选择局部搜索算子 | Q-learning、SARSA；非 PPO/GRPO | $C_{\max}$、提前/拖期、能耗 | 82 个基准实例 | GA、DE、HS 及 27 种算法 | 明确是加工—物流集成，但为静态多目标问题；正文明确报告三类目标和 82 个实例  |
| *A Dynamic Scheduling Method Combining Iterative Optimization and Deep Reinforcement Learning to Solve Sudden Disturbance Events in a Flexible Manufacturing Process* | 动态 FMS/FJSP | 未说明 | 未说明 | 是 | 是 | 未说明 | 订单变化 | 未说明 | 是，DRL | 未说明 | 响应性、稳定性和调度质量 | 机器故障、加工时间滞后、订单变化的案例仿真 | 迭代优化、DRL | 与本文动态重调度方向一致，但缺少已核验的 AGV 联合建模 |
| *A heterogeneous graph attention-enhanced deep reinforcement learning framework for flexible job shop scheduling problem with variable sublots* | 带可变批量的 FJSP | 否 | 否 | 未说明 | 否 | 否 | 否 | 是，异构图注意力 | 是，DRL | 具体策略算法需核验 | 调度效率和 makespan 相关指标 | 模拟/基准 FJSP | 未说明 | 对图注意力和扩大动作空间有方法参考，但不包含 AGV 或故障 |
| *Meta-relation-based heterogeneous graph neural network with deep reinforcement learning for flexible job shop scheduling* | FJSP | 否 | 否 | 未说明 | 否 | 否 | 否 | 是，元关系异构图 | 是，DRL | 具体 PPO 信息需核验 | makespan | FJSP 基准 | 规则和 DRL 方法 | 支持工序—机器关系编码，不支持本文的加工—物流联合建模 |
| *Dynamic Scheduling Method for Job-Shop Manufacturing Systems by Deep Reinforcement Learning with Proximal Policy Optimization* | 动态 JSSP | 运输代理存在，但 AGV 未明确 | 部分 | 是 | 机器故障 | 未说明 | 未说明 | 未说明 | 是，DRL | PPO | 调度性能，$C_{\max}$需核验 | 真实生产环境动态仿真 | 现有方法/规则 | 可支持 PPO 动态调度，但不能直接支持多 AGV 协同；正文摘要明确为 PPO 和机器故障  |
| *Digital Twin-Driven AGV Scheduling and Routing in Automated Container Terminals* | 数字孪生 AGV 调度与路径规划 | 是 | 否，非加工车间协同 | 是，不确定因素、冲突和动态障碍 | 否 | 未说明 | 未说明 | 路网图/路径算法，非 GNN | 否 | 否 | AGV 调度效率、运输成本、冲突 | 自动化集装箱码头数字孪生 | IAFSA-Dijkstra、Yen 算法 | 可借鉴路网、冲突预测和虚实闭环，但不包含加工工序和机器故障 |
| *Integrated scheduling of multi-objective lot-streaming hybrid flowshop with AGV under dynamic environments* | 动态混合流水车间-AGV | 是 | 是 | 是 | 未说明 | 是或动态 AGV 状态，具体需核验 | 未说明 | 未说明 | 未说明 | 未说明 | 多目标 | 动态混合流水车间仿真 | 未说明 | 与本文均处理 AGV 动态扰动，但不是 FJSP，且目标不限定为单一 $C_{\max}$ |
| *Machine learning-driven solutions for sustainable and dynamic flexible job shop scheduling under worker absences and renewable energy variability* | 动态、可持续 FJSP | 否 | 否 | 是：工人缺勤、能源变化 | 否 | 否 | 未说明 | 未说明 | 是，机器学习方法 | 未说明 | 生产效率、能源与可持续性指标 | 动态 FJSP 仿真 | 未说明 | 动态性和不确定性相关，但不包含 AGV；目标体系不是本文的单一 $C_{\max}$ |
| *Flexible Manufacturing Systems intralogistics: Dynamic optimization of AGVs and tool sharing using Colored-Timed Petri Nets and actor–critic RL with actions masking* | FMS 内物流 | 是 | 是，物流资源共享 | 是 | 未说明 | 未说明 | 未说明 | Colored-Timed Petri Net | 是，actor–critic | 非 PPO/GRPO | 内物流和工具共享效率 | FMS 仿真 | 未说明 | 更偏物流控制；本文进一步联合作业、工序、机器和 AGV |
| *Edge computing-based real-time scheduling for digital twin flexible job shop with variable time window* | 数字孪生动态 FJSP | 未说明 | 未说明 | 是，时间窗口和实时调度 | 未说明 | 未说明 | 未说明 | 未说明 | 未说明或使用智能优化 | 未说明 | 实时调度和时间窗性能 | 数字孪生、边缘计算仿真 | 未说明 | 对实时仿真和数字孪生有背景价值，未确认 AGV 协同 |
| *Task Scheduling with Mobile Robots—A Systematic Literature Review* | 移动机器人任务调度综述 | 是/移动机器人 | 部分 | 覆盖动态调度文献 | 未说明 | 未说明 | 未说明 | 未说明 | 覆盖 RL 文献 | 包含 PPO 等既有方法，但非单一方法 | 综述对象多样 | 71 篇 2014—2024 文献 | 文献综述 | 适合背景和分类，不是单一实验方法；不能直接证明某种算法优越 |
| *Dynamic scheduling for flexible job shop based on MachineRank algorithm and reinforcement learning* | 动态 FJSP-AGV | 是，AGV 状态 | 部分 | 是 | 是 | AGV 状态 | 是 | 未说明 | 是，D3QN | 非 PPO/GRPO | 动态调度性能和机器利用率 | 动态 FJSP 仿真 | MachineRank 和复合规则 | 与本文动态事件较接近，但图表示、三阶段动作和退化率评价未确认 |
| *Analysis of Multi-AGVs Management System and Key Issues: A Review* | 多 AGV 管理综述 | 是 | 可能覆盖，但非单一模型 | 覆盖动态调度、路径和任务分配文献 | 未说明 | 未说明 | 未说明 | 路网/路径关系，非 GNN | 覆盖 RL 文献 | 多种算法 | 成本、效率、资源利用率 | 多 AGV 管理文献 | 综述 | 适合路网、死锁、路径冲突和任务分配背景，不直接支持本文方法 |
| *A review of the applications of multi-agent reinforcement learning in smart factories* | 智能工厂 MARL 综述 | 可能包含 | 视被综述研究而定 | 是 | 视被综述研究而定 | 视被综述研究而定 | 视被综述研究而定 | 未说明 | 是，MARL | QMIX、DQN 等多种方法 | 综述 | 智能工厂应用文献 | 综述 | 支持 MARL 应用于复杂动态制造的背景，但不能直接支持本文 GraphGRPO |
| *A Systematic Review on Reinforcement Learning for Industrial Combinatorial Optimization Problems* | 工业组合优化 RL 综述 | 视具体研究而定 | 视具体研究而定 | 覆盖动态问题 | 未说明 | 未说明 | 未说明 | 覆盖多种表示 | 是 | 包括 PPO 等 | 综述状态、动作、奖励设计 | 715 篇检索，298 篇纳入分析 | 文献综述 | 可用于 RL 状态、动作和奖励设计综述，不能替代具体调度实验 |
| *A Reinforcement Learning Environment For Job-Shop Scheduling* | JSSP RL 环境 | 否 | 否 | 环境支持序贯决策，动态事件未说明 | 否 | 否 | 否 | 未说明 | 是 | 未说明 | makespan/调度质量 | RL JSSP 环境和随机实例 | 规则/组合优化方法 | 可借鉴环境接口和 transition 采样，不包含 FJSP-AGV 动态扰动 |

---

## 二、原始文献结构化笔记（30 条记录）

### 1. *Dynamic integrated scheduling of machines and AGVs in flexible job shop considering AGV failures*

**DOI：** 当前证据未显示。
**研究问题：** 研究 AGV 故障下机器与 AGV 的动态集成调度。
**问题设定：** 柔性作业车间，明确包含加工和运输资源。
**加工资源：** 机器及其加工任务；机器候选集合的具体形式未说明。
**物流资源：** 多 AGV；故障发生后由其他可用 AGV 接续运输任务。
**状态与动作：** 状态、动作编码和联合决策粒度未说明。
**算法：** 具体算法名称未说明。
**奖励或目标函数：** 目标包括最小化 makespan，并关注调度鲁棒性；作者将 AGV 故障传播及重调度纳入模型 。
**动态事件：** AGV 故障；事件发生后执行重调度。机器故障、JOB_ADD 和加工时间变化未说明。
**实验设计：** 动态 FJSP-AGV 实例或仿真，规模、训练测试划分和随机种子未说明。
**主要结果：** 当前证据支持其能够研究故障后的 AGV 接续和重调度，但具体数值和统计协议未说明。
**作者明确说明的局限性：** 当前可见证据未显示。
**与本文 SkyEngine/PacketFactory 的关系：** 这是与本文最接近的故障场景文献之一。共同点是 FJSP、AGV、动态事件和 makespan；差异是本文还统一考虑机器故障、JOB_ADD、路网图、三阶段动作、clean/uncertain 配对及退化指标。
**可引用位置：** 第 1 章 1.3.3“动态扰动与实时重调度研究”；第 5 章 5.5“故障场景对比”。

---

### 2. *Dynamic Integrated Scheduling of Production Equipment and Automated Guided Vehicles in a Flexible Job Shop Based on Deep Reinforcement Learning*

**DOI：** 10.3390/pr12112423。
**研究问题：** 研究柔性作业车间生产设备与 AGV 的动态集成调度，重点处理紧急订单插入。
**问题设定：** FJSP-AGV；明确同时建模生产设备和 AGV。
**加工资源：** 工件、工序、机器。
**物流资源：** AGV；摘要明确设置工件选择、机器选择、AGV 选择和目标选择代理。
**状态与动作：** 观测包括车间生产状态；动作由四类代理分别选择工件、机器、AGV 和目标。
**算法：** 多智能体强化学习 QMIX，离线训练后用于动态决策。
**奖励或目标函数：** 静态模型优化最大完工时间和总生产能耗；动态实验考虑紧急订单插入。
**动态事件：** 紧急订单插入；事件驱动决策。机器故障和 AGV 故障未说明。
**实验设计：** 多种案例和仿真场景；训练测试细节、随机种子和重复次数未说明。
**主要结果：** 作者报告 QMIX 方法与其他典型优化算法进行了比较；具体优越性范围和统计信息当前证据未完整显示。
**作者明确说明的局限性：** 当前证据未显示。
**与本文 SkyEngine/PacketFactory 的关系：** 与本文最接近的部分是加工—AGV 联合决策、动态插单和多代理分解。重要差异是其目标包含能耗，而本文将 $C_{\max}$ 设为唯一主目标；其算法是 QMIX，本文是基于 transition 分组组相对优势的 GraphGRPO 实现，不能将二者视为同一方法。
**可引用位置：** 第 1 章 1.3.2；第 2 章 2.5；第 5 章 5.5。

---

### 3. *Deep Reinforcement Learning for Dynamic Flexible Job Shop Scheduling with Random Job Arrival*

**DOI：** 当前证据未显示。
**研究问题：** 研究随机新作业到达条件下的动态 FJSP。
**问题设定：** 动态柔性作业车间；运输代理存在，但 AGV 的具体建模未完全核验。
**加工资源：** 多工件、多工序、多候选机器。
**物流资源：** 运输资源是否为显式 AGV、是否包含路网和路径冲突，未说明。
**状态与动作：** 动态车间状态和新到达作业进入决策环境；具体状态向量、动作空间和动作掩码未说明。
**算法：** 深度强化学习；具体策略实现需结合全文进一步核验。
**奖励或目标函数：** 以动态调度中的完工时间或相关性能为核心；奖励具体形式未说明。
**动态事件：** 随机工件到达；事件在运行过程中产生。
**实验设计：** 当前证据显示包括初始作业和后续到达作业的动态实验；训练/测试划分、随机种子和重复运行未说明。
**主要结果：** 研究验证了 DRL 对随机到达场景的适应性，但当前证据不足以说明其包含本文全部资源和事件。
**作者明确说明的局限性：** 当前证据未显示。
**与本文 SkyEngine/PacketFactory 的关系：** 共同点是 JOB_ADD/随机到达和在线重决策；差异是本文把加工和多 AGV 运输统一为联合动作，并同时处理机器故障和 AGV 故障。
**可引用位置：** 第 1 章 1.3.3；第 5 章 5.5。

---

### 4. *Solving Flexible Job-Shop Scheduling Problem with Heterogeneous Graph Neural Network Based on Relation and Deep Reinforcement Learning*

**DOI：** 10.3390/machines12080584。
**研究问题：** 采用关系异构图神经网络和 DRL 求解 FJSP。
**问题设定：** 静态 FJSP，工序和机器被建模为不同节点及关系。
**加工资源：** 工序节点、机器节点、工序依赖关系和机器兼容关系。
**物流资源：** 未包含 AGV 或其他运输资源。
**状态与动作：** 从 FJSP 实例提取异构图；具体动作是工序/机器调度选择，完整动作编码未说明。
**算法：** HGNNR 与 PPO。
**奖励或目标函数：** 以调度效率和基准实例性能为核心，具体奖励表达式未说明。
**动态事件：** 未说明。
**实验设计：** 4 个公开基准数据集；与 4 种 DRL 方法及 3 种规则方法比较。训练细节、随机种子和统计显著性未说明。
**主要结果：** 摘要声称方法在公开基准上优于所比较的 DRL 和规则方法，但该结论只适用于其 FJSP 基准设定。
**作者明确说明的局限性：** 当前证据未显示。
**与本文 SkyEngine/PacketFactory 的关系：** 可借鉴异构工序—机器关系编码和 PPO 对照；不能直接支持加工—物流协同、AGV 故障或 JOB_ADD。
**可引用位置：** 第 2 章 2.4“图神经网络”；第 5 章 5.5。

---

### 5. *Learning-enabled flexible job-shop scheduling for scalable smart manufacturing*

**DOI：** 当前证据未显示。
**研究问题：** 研究 AGV 型智能制造系统中 DRL 调度的规模泛化问题。
**问题设定：** 柔性作业车间与 AGV-based smart manufacturing system。
**加工资源：** 工件、工序和机器，具体机器柔性设置需核验。
**物流资源：** AGV；其节点或运输关系被纳入可扩展调度问题。
**状态与动作：** 采用结构化调度表示；具体状态、动作和解码过程未说明。
**算法：** 学习型 DRL 调度方法。
**奖励或目标函数：** 调度性能与规模泛化；具体奖励和是否唯一优化 $C_{\max}$ 未说明。
**动态事件：** 主要是系统规模动态变化；机器/车辆增减不应直接等同于故障事件。
**实验设计：** AGV 智能制造环境；训练测试规模、实例生成和基线需核验。
**主要结果：** 论文指出 DRL 在规模扩展时面临长且昂贵的学习过程，研究重点是提高规模泛化 。
**作者明确说明的局限性：** DRL 的学习曲线长、训练成本高。
**与本文 SkyEngine/PacketFactory 的关系：** 可用于讨论 AGV 调度和规模泛化；本文更关注动态故障、JOB_ADD 和 clean/uncertain 配对，而非仅系统规模变化。
**可引用位置：** 第 1 章 1.3.2；第 2 章 2.5；第 5 章 5.6“泛化实验”。

---

### 6. *Flexible Job Shop Scheduling Optimization with Machine and AGV Integration Based on Improved NSGA-II*

**DOI：** 当前证据未显示。
**研究问题：** 研究机器与 AGV 集成的柔性作业车间多目标调度。
**问题设定：** 静态 FJSP-AGV。
**加工资源：** 8 台机器和 14 类活塞环工件。
**物流资源：** 2 台 AGV；考虑载荷运输时间和空载时间。
**状态与动作：** 三层编码分别表示工序、机器和 AGV；这是编码结构，不等同于在线 RL 动作空间。
**算法：** 改进 NSGA-II、变邻域搜索及反向个体策略。
**奖励或目标函数：** 最小化 $C_{\max}$、机器负载和总能耗。
**动态事件：** 未说明。
**实验设计：** 基准实验和活塞环制造车间案例；训练/测试划分不适用或未说明。
**主要结果：** 摘要声称改进 NSGA-II 在 FJSP-AGV 问题上具有较好性能，并观察到增加 AGV 的边际收益递减。该结论限于其模型和实例。
**作者明确说明的局限性：** 当前证据未显示。
**与本文 SkyEngine/PacketFactory 的关系：** 适合作为静态加工—AGV 联合优化基线；本文只将 $C_{\max}$ 作为主目标，并增加机器故障、AGV 故障、JOB_ADD 和在线重决策。
**可引用位置：** 第 1 章 1.3.2；第 5 章 5.5。

---

### 7. *Multi-Agent Reinforcement Learning for Job Shop Scheduling in Dynamic Environments*

**DOI：** 当前证据未显示。
**研究问题：** 使用 MARL 处理动态环境中的作业车间调度。
**问题设定：** 动态 JSSP，是否为 FJSP 未说明。
**加工资源：** 机器和工件/工序。
**物流资源：** AGV 和运输网络未说明。
**状态与动作：** 多智能体状态和动作机制未说明。
**算法：** MARL；具体算法未说明。
**奖励或目标函数：** 动态调度性能，是否以 $C_{\max}$ 为唯一主目标未说明。
**动态事件：** 动态订单或资源扰动，具体类型需核验。
**实验设计：** 动态制造环境；数据划分、基线、随机种子和统计协议未说明。
**主要结果：** 当前证据只能确认其讨论动态 JSSP 的多智能体调度，不能确认其对本文 AGV 联合问题的直接适用性。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 为 MARL 动态决策提供方法背景，但不能直接支持多 AGV 加工—物流协同或 GraphGRPO。
**可引用位置：** 第 2 章 2.5。

---

### 8. *Multi-faceted dynamic flexible job shop scheduling via heterogeneous graph transformer and deep reinforcement learning*

**DOI：** 10.1016/j.eswa.2025.130532。
**研究问题：** 研究多因素动态 FJSP，并用异构图 Transformer 和 DRL 处理动态状态。
**问题设定：** 动态 FJSP；数据生成器考虑作业特征、资源可用性、调度约束和动态事件。
**加工资源：** 多作业、多工序、多机器及加工时间分布。
**物流资源：** AGV 是否包含未说明。
**状态与动作：** 异构图状态；具体动作分解和动作掩码未说明。
**算法：** 异构图 Transformer 与 DRL；具体 PPO 变体需核验。
**奖励或目标函数：** 动态调度性能，$C_{\max}$ 是否唯一主目标未说明。
**动态事件：** 生成数据包括故障持续时间、故障间隔和作业相关动态参数；具体事件分类需核验 。
**实验设计：** 模拟生成 MDFJSSP 数据；工序数服从均匀分布，处理时间采用截断正态分布。训练/测试划分和统计重复未说明。
**主要结果：** 当前证据支持其覆盖动态 FJSP 和异构图表示，但不能确认 AGV 协同。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 可参考动态数据生成和异构图 Transformer；本文额外统一表示机器、工序、作业和 AGV，并采用 GraphGRPO 进行三阶段在线决策。
**可引用位置：** 第 2 章 2.4；第 5 章 5.6。

---

### 9. *Digital Twin-Driven Adaptive Scheduling for Flexible Job Shops*

**DOI：** 10.3390/su14095340。
**研究问题：** 使用数字孪生实时监控生产线，并在动态事件出现时自适应更新调度。
**问题设定：** 数字孪生驱动的动态 FJSP。
**加工资源：** 柔性作业车间生产线和机器。
**物流资源：** AGV 未说明。
**状态与动作：** 数字孪生采集实际生产状态；调度算法根据异常事件调整排程。
**算法：** 强化学习增强遗传算法。
**奖励或目标函数：** 调度质量和求解效率；唯一 $C_{\max}$ 主目标未说明。
**动态事件：** 随机作业到达、机器或生产过程异常；事件发生后触发自适应重调度。
**实验设计：** 动态调度实验和实验室环境验证。
**主要结果：** 作者报告数字孪生能够提供实时生产状态，并帮助捕获异常事件和更新调度；这支持数字孪生闭环，但不等于已经满足特定在线决策时间约束。
**作者明确说明的局限性：** 当前证据未显示。
**与本文 SkyEngine/PacketFactory 的关系：** 可支持仿真—调度闭环和动态事件捕获的研究背景；本文进一步将 AGV、路网和加工资源统一到一个仿真平台。
**可引用位置：** 第 1 章 1.3.4；第 3 章 3.1；第 5 章 5.4。

---

### 10. *Dynamic scheduling of flexible job-shop considering automatic guided vehicle transportation via deep reinforcement learning*

**DOI：** 当前证据未显示。
**研究问题：** 研究考虑 AGV 运输的动态 FJSP，并使用 DRL 进行调度。
**问题设定：** 动态 FJSP-AGV。
**加工资源：** 工件、工序、候选机器。
**物流资源：** AGV 和运输任务；AGV 利用率被作为评价指标。
**状态与动作：** 动态加工和 AGV 状态；联合动作的具体定义未说明。
**算法：** DRL，具体网络和策略算法未说明。
**奖励或目标函数：** 评价指标包括作业按时完成率、机器利用率和 AGV 利用率；$C_{\max}$ 是否主目标未说明 。
**动态事件：** 动态作业和 AGV 状态变化；机器故障、AGV 故障、JOB_ADD 的具体区分未说明。
**实验设计：** 多批次加工任务和仿真比较；训练测试划分、基线和随机种子未说明。
**主要结果：** 当前证据支持其考察加工和 AGV 的动态调度性能，但不支持将其直接视为与 PacketFactory 完全同构。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 是最接近本文的文献之一；主要差异是本文显式设置三阶段工序—机器—AGV 动作、机器/AGV 故障、JOB_ADD、clean/uncertain 和退化率。
**可引用位置：** 第 1 章 1.3.2、1.3.3；第 5 章 5.5。

---

### 11. *A Dynamic Scheduling Method Combining Iterative Optimization and Deep Reinforcement Learning to Solve Sudden Disturbance Events in a Flexible Manufacturing Process*

**DOI：** 当前证据未显示。
**研究问题：** 处理柔性制造过程中机器故障、加工时间滞后和订单变化等突发扰动。
**问题设定：** 动态 FMS/FJSP。
**加工资源：** 机器和加工任务。
**物流资源：** 未说明。
**状态与动作：** 建立实时 DRL 生产环境；具体编码未说明。
**算法：** 迭代优化与 DRL；采用事件驱动和周期驱动相结合的混合重调度触发策略。
**奖励或目标函数：** 强调响应性、反馈速度、质量和稳定性；唯一 $C_{\max}$ 主目标未说明。
**动态事件：** 机器故障、处理时间滞后、订单变化；机器故障案例在 25 秒时触发。
**实验设计：** 不同规模订单问题和多种扰动案例。
**主要结果：** 作者声称所提方法能够对突发扰动及时反馈并生成调整后的排程，但当前证据不支持其在多 AGV 环境中的结论。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 可借鉴事件—周期混合重调度触发机制；本文将故障和 JOB_ADD 统一为仿真事件，并使用配对 clean/uncertain 指标。
**可引用位置：** 第 1 章 1.3.3；第 3 章 3.4。

---

### 12. *Flexible job shop scheduling via deep reinforcement learning with meta-path-based heterogeneous graph neural network*

**DOI：** 10.1016/j.knosys.2024.111940。
**研究问题：** 用元路径异构图神经网络和 DRL 求解 FJSP。
**问题设定：** 静态 FJSP。
**加工资源：** 工序、机器以及加工分配关系。
**物流资源：** 未包含 AGV。
**状态与动作：** 采用异构图表示全局调度状态；动作具体编码未说明。
**算法：** GIN-A3C、GIN-PPO、GAT-PPO 等图强化学习基线和提出方法。
**奖励或目标函数：** 主要比较 Brandimarte 数据集上的相对差距和调度性能。
**动态事件：** 未说明。
**实验设计：** Brandimarte 数据集；与 Rule 4、IPSO、SLGA 以及多个图 DRL 方法比较。
**主要结果：** 当前正文片段显示其使用异构图和 DRL，并报告不同图网络与策略组合的相对差距；不应据此推断其适合 AGV 动态调度。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 可作为图 DRL 和 PPO 对照；不能直接支持加工—运输协同。
**可引用位置：** 第 2 章 2.4；第 5 章 5.5。

---

### 13. *A Tree neural network deep reinforcement learning for flexible job shop scheduling with transportation constraints*

**DOI：** 当前证据未显示。
**研究问题：** 研究含运输约束的 FJSP-T。
**问题设定：** 加工和运输存在层级耦合。
**加工资源：** 工件、工序和候选机器。
**物流资源：** 运输资源；是否为 AGV、是否有多车和路网冲突未说明。
**状态与动作：** 使用 hierarchical scheduling and transport tree（HSTT）以及局部注意机制；具体动作空间未说明。
**算法：** Tree neural network 与 DRL。
**奖励或目标函数：** 文献综述部分将该类问题与 makespan 联系起来，但本文具体奖励表达式未说明。
**动态事件：** 未说明。
**实验设计：** FJSP-T 实例和仿真；训练划分、随机种子和统计协议未说明。
**主要结果：** 研究重点是利用树结构表达调度和运输约束，而不是动态故障恢复。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 可参考加工—运输树状关系和层级动作设计；本文进一步显式加入多 AGV、工厂路网和动态事件。
**可引用位置：** 第 1 章 1.3.2；第 2 章 2.4。

---

### 14. *Flexible Manufacturing Systems intralogistics: Dynamic optimization of AGVs and tool sharing using Colored-Timed Petri Nets and actor–critic RL with actions masking*

**DOI：** 当前证据未显示。
**研究问题：** 研究 FMS 内 AGV 和工具共享的动态优化。
**问题设定：** FMS 内物流系统。
**加工资源：** 工具共享和生产资源；加工工序联合模型未说明。
**物流资源：** 多 AGV、运输任务和共享工具。
**状态与动作：** Colored-Timed Petri Net 表示离散事件状态；动作掩码约束不可行动作。
**算法：** actor–critic RL。
**奖励或目标函数：** 内物流动态优化和资源共享效率；$C_{\max}$ 未说明。
**动态事件：** 动态任务和资源状态；故障类型未说明。
**实验设计：** FMS 仿真和算法比较；规模、训练划分和统计信息未说明。
**主要结果：** 可支持 Petri 网和动作掩码适用于动态 AGV 物流控制，但不能直接证明图模型优于其他模型。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 可借鉴运输资源状态、动作可行性掩码和离散事件仿真；本文同时决策工序、机器和 AGV。
**可引用位置：** 第 2 章 2.3；第 3 章 3.2。

---

### 15. *Dynamic Scheduling for Flexible Manufacturing Systems Based on Multi-Agent Deep Reinforcement Learning and Petri Nets*

**DOI：** 当前证据未显示。
**研究问题：** 研究动态 FMS 中新订单、订单取消和机器故障下的调度。
**问题设定：** FMS 离散事件系统。
**加工资源：** 机器、工件和订单。
**物流资源：** 是否显式包含 AGV 未说明。
**状态与动作：** Petri 网表示系统状态；MARL 进行动态调度。
**算法：** 多智能体深度强化学习与 Petri 网。
**奖励或目标函数：** 动态调度性能，具体奖励和主目标未说明。
**动态事件：** 新订单到达、临时订单取消、机器故障 。
**实验设计：** FMS 仿真；基线、训练划分和重复运行未说明。
**主要结果：** 研究明确面向实时扰动下的动态 FMS 调度，但当前证据不能确认其包含多 AGV 加工—物流联合决策。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 可支持统一事件机制和 MARL 动态调度背景；本文进一步加入 AGV 故障和路网图。
**可引用位置：** 第 1 章 1.3.3；第 2 章 2.5。

---

### 16. *Dynamic scheduling for flexible job shop based on MachineRank algorithm and reinforcement learning*

**DOI：** 当前证据未显示。
**研究问题：** 动态 FJSP 中处理新作业插入、机器故障、加工时间变化和 AGV 状态变化。
**问题设定：** 动态 FJSP-AGV。
**加工资源：** 机器、工序和候选加工资源。
**物流资源：** AGV 状态被纳入问题，但 AGV 动作和路网细节未说明。
**状态与动作：** 机器利用率和 AGV 状态等动态信息；复合规则选择未加工工序和机器。
**算法：** MachineRank 规则与 D3QN。
**奖励或目标函数：** 调度性能和机器利用率；是否唯一优化 $C_{\max}$ 未说明。
**动态事件：** 新作业插入、机器故障、加工时间变化、AGV 状态变化 。
**实验设计：** 动态 FJSP 仿真；详细规模、随机种子和统计协议未说明。
**主要结果：** 论文提出 D3QN 和 MachineRank 组合以处理多类动态事件。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 是动态事件集合最接近本文的文献之一；差异是本文采用图状态和三阶段联合动作，并使用 clean/uncertain 的性能退化分析。
**可引用位置：** 第 1 章 1.3.3；第 5 章 5.5。

---

### 17. *Learning to schedule job-shop problems: Representation and policy learning using graph neural network and reinforcement learning*

**DOI：** 当前证据未显示。
**研究问题：** 使用 GNN 和 RL 学习 JSSP 调度策略。
**问题设定：** 静态 JSSP。
**加工资源：** 工件、工序和机器。
**物流资源：** 未包含 AGV。
**状态与动作：** 将调度状态表示为图，GNN 生成节点嵌入并用于策略决策。
**算法：** GNN + RL。
**奖励或目标函数：** 以 JSSP 调度质量/makespan 为主要方向。
**动态事件：** 未说明。
**实验设计：** 随机生成 JSSP；最大测试规模为 20 台机器和 100 个作业，训练采样 transition。
**主要结果：** 研究表明图表示能够用于学习 JSSP 策略，但当前证据没有证明其在 FJSP-AGV 动态场景中有效。
**作者明确说明的局限性：** GNN 层数增加会提高计算量，超过一定范围后性能提升有限 。
**与本文 SkyEngine/PacketFactory 的关系：** 可借鉴图状态、transition 采样和策略训练；本文将图扩展到机器、工序、作业和 AGV 异构节点。
**可引用位置：** 第 2 章 2.4；第 2 章 2.5。

---

### 18. *Integrated Scheduling of Multi-Objective Job Shops and Material Handling Robots with Reinforcement Learning Guided Meta-Heuristics*

**DOI：** 当前证据未显示。
**研究问题：** 集成作业车间和物料搬运机器人调度。
**问题设定：** JSSP-MHR。
**加工资源：** 作业、工序和机器。
**物流资源：** MHR；是否为 AGV 未说明。
**状态与动作：** RL 用于在元启发式迭代中选择局部搜索算子。
**算法：** GA、差分进化、和声搜索与 Q-learning/SARSA。
**奖励或目标函数：** makespan、提前/拖期和总能耗；属于多目标问题 。
**动态事件：** 未说明。
**实验设计：** 82 个基准实例，比较 27 种算法。
**主要结果：** 作者报告 GA 与 SARSA 的某一奖励设置具有竞争力，但该结果不能直接转化为动态 AGV 调度结论。
**作者明确说明的局限性：** 未来工作内容当前证据未显示。
**与本文 SkyEngine/PacketFactory 的关系：** 明确支持加工资源和物流资源联合考虑；本文将其扩展到动态单目标在线调度。
**可引用位置：** 第 1 章 1.3.2；第 5 章 5.5。

---

### 19. *A Dynamic Scheduling Method Combining Iterative Optimization and Deep Reinforcement Learning to Solve Sudden Disturbance Events in a Flexible Manufacturing Process*

与第 11 篇为同一文献，不重复添加新的事实。其核心价值是事件驱动和周期驱动重调度触发机制，以及机器故障、加工时间滞后和订单变化场景。建议在综述中只保留一条书目记录，避免重复计数。

---

### 20. *A heterogeneous graph attention-enhanced deep reinforcement learning framework for flexible job shop scheduling problem with variable sublots*

**DOI：** 当前证据未显示。
**研究问题：** 研究可变批量/分批加工扩大动作空间时的 FJSP 调度。
**问题设定：** FJSP，未包含 AGV。
**加工资源：** 工序、机器和可变批量。
**物流资源：** 未说明。
**状态与动作：** 异构图注意力编码；动作包括与分批和机器分配相关的决策，具体定义未说明。
**算法：** HGADRL。
**奖励或目标函数：** 调度效率和相对最优性能，具体奖励未说明。
**动态事件：** 未说明。
**实验设计：** FJSP 模拟/基准实例；训练划分和统计信息未说明。
**主要结果：** 作者关注可变批量带来的组合空间扩张，并提出图注意力 DRL 框架。
**作者明确说明的局限性：** 可变批量会增加搜索空间和收敛难度。
**与本文 SkyEngine/PacketFactory 的关系：** 可作为复杂动作空间和异构图编码的参考，但不支持 AGV 协同和动态故障。
**可引用位置：** 第 2 章 2.4。

---

### 21. *Meta-relation-based heterogeneous graph neural network with deep reinforcement learning for flexible job shop scheduling*

**DOI：** 当前证据未显示。
**研究问题：** 通过元关系异构图神经网络提取 FJSP 中机器和工序关系，并进行 DRL 调度。
**问题设定：** 静态 FJSP。
**加工资源：** 工序和机器节点及其关系。
**物流资源：** 未包含 AGV。
**状态与动作：** 元关系异构图；具体动作和奖励未说明。
**算法：** 元关系 HGNN + DRL。
**奖励或目标函数：** FJSP 调度性能，具体目标需核验。
**动态事件：** 未说明。
**实验设计：** FJSP 基准和 DRL/启发式对照；详细协议未说明。
**主要结果：** 研究支持在关系类型较多的 FJSP 中使用异构图表示，但不能推出图模型在动态 AGV 场景中必然更优。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 可参考机器—工序关系建模；本文图中还加入作业、AGV 和运输网络节点。
**可引用位置：** 第 2 章 2.4。

---

### 22. *Dynamic Scheduling Method for Job-Shop Manufacturing Systems by Deep Reinforcement Learning with Proximal Policy Optimization*

**DOI：** 当前证据未显示。
**研究问题：** 研究机器故障下的动态 JSSP。
**问题设定：** 动态 JSSP 和真实生产环境仿真。
**加工资源：** 机器、工件和工序。
**物流资源：** 运输代理存在，但 AGV 类型和联合建模未说明。
**状态与动作：** 生产状态和运输代理状态进入 DRL 环境；具体动作未说明。
**算法：** PPO。
**奖励或目标函数：** 动态调度性能；是否将 $C_{\max}$ 作为唯一主目标未说明。
**动态事件：** 机器故障。
**实验设计：** 真实生产环境动态仿真；比较方法和统计协议未说明。
**主要结果：** 作者报告 PPO-DRL 在所测试的动态生产环境中表现较好 。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 可作为 PPO 动态调度背景和对比方法，但不能直接支持多 AGV、GraphGRPO 或统一故障机制。
**可引用位置：** 第 2 章 2.5；第 5 章 5.5。

---

### 23. *Digital Twin-Driven AGV Scheduling and Routing in Automated Container Terminals*

**DOI：** 当前证据未显示。
**研究问题：** 数字孪生驱动的 AGV 调度、路径规划和冲突处理。
**问题设定：** 自动化集装箱码头，不是 FJSP。
**加工资源：** 未包含加工机器和工序。
**物流资源：** AGV、运输任务、路网和路径冲突。
**状态与动作：** 虚实空间交互，比较物理与虚拟状态；采用路径算法解决冲突。
**算法：** IAFSA-Dijkstra、Yen 算法。
**奖励或目标函数：** AGV 调度效率和运输成本。
**动态事件：** 不确定环境因素、冲突和动态障碍；新作业和故障类型未说明。
**实验设计：** 数字孪生码头案例。
**主要结果：** 案例结果支持数字孪生用于检测物理—虚拟差异、预测冲突和更新 AGV 路径。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 可支持路网、路径冲突和数字孪生建模；不能支持加工—物流协同调度的完整论断。
**可引用位置：** 第 1 章 1.3.4；第 3 章 3.2。

---

### 24. *Integrated scheduling of multi-objective lot-streaming hybrid flowshop with AGV under dynamic environments*

**DOI：** 当前证据未显示。
**研究问题：** 动态环境下带 AGV 的多目标批流混合流水车间调度。
**问题设定：** 混合流水车间，不是 FJSP。
**加工资源：** 流水车间机器和批流作业。
**物流资源：** AGV。
**状态与动作：** 动态加工和运输状态，具体表示未说明。
**算法：** 具体算法未说明。
**奖励或目标函数：** 多目标调度。
**动态事件：** 包含动态环境和 AGV 相关扰动；机器故障、AGV 故障和插单的具体设置未说明。
**实验设计：** 动态混合流水车间仿真，详细协议未说明。
**主要结果：** 可作为动态加工—物流协同的邻近问题参考。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 共同点是 AGV 和动态协同；差异是问题结构、目标体系和工序柔性不同。
**可引用位置：** 第 1 章 1.3.2、1.3.3。

---

### 25. *Machine learning-driven solutions for sustainable and dynamic flexible job shop scheduling under worker absences and renewable energy variability*

**DOI：** 当前证据未显示。
**研究问题：** 工人缺勤和可再生能源变化下的动态、可持续 FJSP。
**问题设定：** 动态 FJSP。
**加工资源：** 机器、工件、工序和工人可用性。
**物流资源：** AGV 未说明。
**状态与动作：** 动态资源和能源状态；具体表示未说明。
**算法：** 机器学习方法，具体算法需核验。
**奖励或目标函数：** 生产效率、能源和可持续性指标。
**动态事件：** 工人缺勤、能源波动；机器/AGV 故障和 JOB_ADD 未说明。
**实验设计：** 动态 FJSP 仿真；数据划分和基线未说明。
**主要结果：** 可支持动态资源可用性和不确定加工环境背景，但不能直接支持 AGV 协同。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 本文关注机器与运输故障，而该文关注人力和能源扰动；二者可在动态事件分类中形成互补。
**可引用位置：** 第 1 章 1.3.3。

---

### 26. *Flexible Manufacturing Systems intralogistics: Dynamic optimization of AGVs and tool sharing using Colored-Timed Petri Nets and actor–critic RL with actions masking*

与第 14 篇为同一文献。综述中建议合并为一条记录，避免重复计数。

---

### 27. *Edge computing-based real-time scheduling for digital twin flexible job shop with variable time window*

**DOI：** 当前证据未显示。
**研究问题：** 数字孪生、边缘计算和可变时间窗下的实时 FJSP 调度。
**问题设定：** 数字孪生 FJSP。
**加工资源：** 机器、作业和时间窗。
**物流资源：** AGV 未说明。
**状态与动作：** 数字孪生状态和边缘计算实时调度信息；具体表示未说明。
**算法：** 具体算法未说明。
**奖励或目标函数：** 实时调度与时间窗满足情况。
**动态事件：** 时间窗变化和实时状态变化；故障、插单未说明。
**实验设计：** 数字孪生/边缘计算仿真；基线和统计协议未说明。
**主要结果：** 可用于说明数字孪生与实时调度的系统背景。
**作者明确说明的局限性：** 未说明。
**与本文 SkyEngine/PacketFactory 的关系：** 本文也强调仿真验证和在线决策，但 PacketFactory 额外显式建模多 AGV 和故障事件。
**可引用位置：** 第 1 章 1.3.4；第 3 章 3.1。

---

### 28. *Task Scheduling with Mobile Robots—A Systematic Literature Review*

**DOI：** 当前证据未显示。
**研究问题：** 综述移动机器人任务调度，包括机器分配、机器人分配、运输和路径规划。
**问题设定：** 移动机器人调度综述。
**加工资源：** 视被综述研究而定。
**物流资源：** AGV/AMR/移动机器人。
**状态与动作：** 视具体文献而定。
**算法：** 涵盖启发式、元启发式、机器学习和 RL，包括 PPO 等方法。
**奖励或目标函数：** 多样化，包括完工时间、成本、利用率和路径质量。
**动态事件：** 综述覆盖动态调度，但各文献事件不统一。
**实验设计：** 系统分析 2014—2024 年的 71 篇文献。
**主要结果：** 综述指出任务调度与路径规划、障碍物规避的联合仍是实际部署中的重要问题。
**作者明确说明的局限性：** 移动机器人调度仍面临路径规划、障碍规避、可靠性和规模扩展问题。
**与本文 SkyEngine/PacketFactory 的关系：** 适合支撑 AGV/移动机器人调度分类和路网问题背景；不能直接支持本文具体算法。
**可引用位置：** 第 1 章 1.3.2；第 1 章 1.3.4。

---

### 29. *Analysis of Multi-AGVs Management System and Key Issues: A Review*

**DOI：** 当前证据未显示。
**研究问题：** 综述多 AGV 管理中的任务分配、路径规划和时间调度。
**问题设定：** 多 AGV 管理系统。
**加工资源：** 视应用场景而定。
**物流资源：** 多 AGV、任务、路网和共享路径。
**状态与动作：** 任务维度、空间维度和时间维度；具体算法依研究而定。
**算法：** 涵盖遗传算法、Dijkstra、死锁避免、机器学习等。
**奖励或目标函数：** 成本降低、效率提升和个性化需求。
**动态事件：** 讨论动态任务、路径冲突和死锁，但不是单一动态 FJSP 模型。
**实验设计：** 文献综述，不是单一实验。
**主要结果：** 综述将多 AGV 管理分为任务调度、路径规划和时间调度三个维度，并讨论死锁和共享路径问题 。
**作者明确说明的局限性：** 多 AGV 管理仍需要综合考虑任务、空间和时间层面的耦合。
**与本文 SkyEngine/PacketFactory 的关系：** 支持本文建立工厂路网和 AGV 资源关系图；不能直接支持 GraphGRPO 或 FJSP 加工决策。
**可引用位置：** 第 1 章 1.3.2；第 3 章 3.2。

---

### 30. *A review of the applications of multi-agent reinforcement learning in smart factories*

**DOI：** 当前证据未显示。
**研究问题：** 综述 MARL 在智能工厂任务中的应用。
**问题设定：** 智能工厂多智能体控制与调度。
**加工资源：** 视被综述研究而定。
**物流资源：** 部分研究包含 AGV 或移动设备，但不能转移到全部文献。
**状态与动作：** 综述讨论状态、动作、奖励和智能体协作。
**算法：** 覆盖 SARL、MARL、DQN、QMIX 等；不等同于本文 GraphGRPO。
**奖励或目标函数：** 各研究不同。
**动态事件：** 综述强调订单、设备和环境不确定性。
**实验设计：** 文献综述，非单一实验。
**主要结果：** 文章认为智能工厂的分布式决策和动态不确定性使 MARL 成为重要研究方向，但这是综述层面的判断。
**作者明确说明的局限性：** MARL 存在复杂度、协调、训练稳定性和可扩展性问题。
**与本文 SkyEngine/PacketFactory 的关系：** 可用于 MARL 背景和协同决策动机；不能直接证明本文 GraphGRPO 优于 QMIX 或 PPO。
**可引用位置：** 第 2 章 2.5；第 2 章 2.6。

---

## 三、最接近本文的 8 篇文献：证据核验摘要

### 1. *Dynamic integrated scheduling of machines and AGVs in flexible job shop considering AGV failures*

该文明确把柔性作业车间、机器、AGV 和 AGV 故障放在同一个动态集成调度问题中。正文片段说明，AGV 在运输过程中发生故障后，系统会调用其他可用 AGV 接续任务并进行重调度；目标包括最小化 makespan 和提高调度鲁棒性 。因此，它确实包含 AGV，也确实包含加工—物流协同问题，并且确实包含动态事件和 AGV 故障。不过，当前证据未确认其是否同时包含机器故障、JOB_ADD、图模型、强化学习或 PPO/QMIX。与本文最重要的相同点是都关注动态 FJSP-AGV 和扰动恢复；最重要的不同点是本文还统一处理机器故障、AGV 故障和 JOB_ADD，并采用 GraphGRPO、三阶段联合动作以及 clean/uncertain 配对评价。该文能够直接支持“AGV 故障会导致运输任务接续和重调度需求”，但不能直接支持“图模型适合该问题”或“GRPO 可用于该问题”。

### 2. *Dynamic Integrated Scheduling of Production Equipment and Automated Guided Vehicles in a Flexible Job Shop Based on Deep Reinforcement Learning*

该文明确包含生产设备和 AGV，并将动态问题设置为紧急订单插入。摘要显示，作者设置了工件选择、机器选择、AGV 选择和目标选择四类代理，使用 QMIX 进行离线训练，再在动态环境中执行决策。因此，它确实联合决定加工和运输，也确实包含动态事件和新订单插入；但当前证据没有确认机器故障和 AGV 故障。它使用强化学习，具体是多智能体 QMIX，而不是 PPO 或 GRPO。其静态模型同时优化最大完工时间和总能耗，因此不能直接作为本文“唯一主目标为 $C_{\max}$”的同设定文献。与本文的相同点是多资源联合决策和事件驱动调整；不同点是本文使用单一 $C_{\max}$，并用 $\Delta C_{\max}$ 与 $D_{C_{\max}}$ 表征动态退化。该文可以直接支持加工—物流协同和动态插单调度，但不能支持 GraphGRPO 的有效性。

### 3. *Deep Reinforcement Learning for Dynamic Flexible Job Shop Scheduling with Random Job Arrival*

该文明确研究随机作业到达下的动态 FJSP，适合支持 JOB_ADD 场景和在线重调度动机。当前证据显示其构造了初始作业与后续随机到达作业组成的动态测试环境，但没有充分确认 AGV 是否作为独立运输资源、是否存在多 AGV、路网和运输冲突，也没有确认机器故障或 AGV 故障。该文使用 DRL，但当前保留证据不足以准确填写其 PPO、DQN 或其他具体策略算法。其核心评价与动态完工时间有关，但奖励函数和是否以 $C_{\max}$ 为唯一主目标仍待核验。与本文最重要的相同点是随机新作业到达和事件后重新决策；差异是本文把作业、工序、机器和 AGV 纳入统一图状态，并同时测试机器故障和 AGV 故障。该文能支持“静态排程难以覆盖随机到达作业”，但不能直接支持 AGV 协同或 GraphGRPO。

### 4. *Learning-enabled flexible job-shop scheduling for scalable smart manufacturing*

该文面向 AGV-based smart manufacturing system，重点讨论 DRL 调度在系统规模变化下的可扩展性。当前证据确认其与 AGV 制造系统有关，也确认论文关注 DRL 的训练成本和规模泛化问题；但没有充分确认其是否处理故障、插单、机器—AGV 联合动作或特定路网冲突。图表示的存在可以从检索片段看出，但具体 GNN 架构未说明，不能直接把它归为某一种图神经网络。该文使用 DRL，但 PPO/QMIX/GRPO 未说明。与本文相似的是都关注 AGV 制造系统和可扩展调度；不同点是本文重点是动态扰动下的在线联合决策和性能退化，而不是单纯规模泛化。该文可支持“DRL 在大规模智能制造中面临训练和泛化挑战”，不能支持 GraphGRPO 或本文事件机制。

### 5. *Dynamic scheduling of flexible job-shop considering automatic guided vehicle transportation via deep reinforcement learning*

该文标题和正文检索片段均明确涉及 AGV 运输和动态 FJSP，因此可以确认其包含 AGV，并且研究方向是加工与运输耦合调度。正文片段还报告了多批次加工任务、作业按时完成率、机器利用率和 AGV 利用率等指标，说明其至少在实验评价中同时观察加工和物流资源表现 。不过，当前证据没有确认其是否具体处理机器故障、AGV 故障或 JOB_ADD，也没有确认采用 GNN、PPO、QMIX 或 DQN。它使用 DRL，但主目标是否严格为 $C_{\max}$ 未说明。与本文的相同点是动态 FJSP、AGV 和 DRL；差异是本文明确规定联合动作、故障事件、clean/uncertain 场景和退化率。该文适合支持“加工与 AGV 运输需要在动态环境中共同评价”，但不能直接支持 GraphGRPO。

### 6. *A Tree neural network deep reinforcement learning for flexible job shop scheduling with transportation constraints*

该文明确研究带运输约束的 FJSP，并提出 hierarchical scheduling and transport tree。它确实涉及加工—运输耦合，但当前证据没有确认运输资源是否明确为 AGV，也没有确认多 AGV、位置、速度、容量或路径冲突。该文使用树神经网络和 DRL，说明它可以为工序、机器和运输关系的层级表达提供方法参考；但当前证据未确认其具体使用 PPO、DQN 或其他策略，也未确认动态故障和新作业到达。与本文最重要的相同点是都将加工决策和运输约束放入同一调度框架；不同点是本文明确采用多 AGV、工厂路网和动态事件，并进行在线三阶段决策。该文可以支持“运输约束需要进入调度状态和决策”，但不能直接支持 AGV 故障、动态重调度或 GraphGRPO。

### 7. *Flexible Job Shop Scheduling Optimization with Machine and AGV Integration Based on Improved NSGA-II*

该文明确包含机器和 AGV，并采用三层编码同时表达工序、机器和 AGV；案例中包括 8 台机器、2 台 AGV 和 14 类活塞环工件。它确实实现了加工—物流集成，但属于静态多目标优化，目标包括 $C_{\max}$、机器负载和总能耗，并非本文的单目标 $C_{\max}$。当前证据没有动态事件、机器故障、AGV 故障或插单，也没有图模型或强化学习。它使用改进 NSGA-II、变邻域搜索等元启发式方法，不涉及 PPO/GRPO。与本文最重要的相同点是机器—AGV 联合建模和联合编码；不同点是本文研究动态在线调度，并以 clean/uncertain 的 $C_{\max}$ 及退化指标评价。该文可作为静态加工—AGV 集成基线，但不能直接支持动态 RL 或 GraphGRPO。

### 8. *Multi-faceted dynamic flexible job shop scheduling via heterogeneous graph transformer and deep reinforcement learning*

该文明确研究动态 FJSP，并使用异构图 Transformer 与 DRL。其数据生成过程考虑作业特征、资源可用性、调度约束、加工时间、故障持续时间和故障间隔等动态参数 。因此，可以确认动态事件和图模型，但当前证据没有确认 AGV 是否存在，也没有确认加工—物流联合决策、机器故障与 AGV 故障的具体分类。它使用 DRL，但 PPO 的具体使用方式未核验。与本文相同的是图状态、动态事件和 DRL；不同点是本文明确加入工序—机器—AGV 异构图、JOB_ADD、机器/AGV 故障以及 clean/uncertain 配对评价。该文能支持图模型用于动态 FJSP 表示的研究动机，但不能直接支持图模型优于其他模型，也不能支持 GraphGRPO。

---
