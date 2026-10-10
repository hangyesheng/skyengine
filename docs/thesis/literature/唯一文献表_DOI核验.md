# 唯一文献表：DOI 去重与元数据核验

## 1. 核验说明

本表以 [文献矩阵_Zotero.md](文献矩阵_Zotero.md) 中的 33 行文献比较矩阵为主输入。按明确的重复记录合并后，当前得到 **30 条唯一候选文献**。去重规则按以下优先级执行：

1. DOI 标准化后完全一致，视为同一文献；
2. DOI 缺失时，使用标准化标题作为临时去重键；
3. 仅题目相近但无法确认是同一篇论文时，不合并，保留为独立候选；
4. DOI、作者、年份、期刊和文献类型只有在当前 Zotero 输出或 Crossref/出版社元数据中明确出现时才填写；无法确认的字段写“未说明”或“待核验”。

“唯一”表示书目记录去重，不表示 30 条文献都已经完成全文方法核验。当前主表中的大多数无 DOI 条目仍需要通过出版社页面、PDF 首页或 Crossref 标题检索进一步确认。

## 2. 统计摘要

当前 BibTeX 文库包含 44 条唯一记录；本台账继续针对矩阵中的 30 条候选文献进行去重和证据边界管理。此前 Crossref 状态沿用原整理记录，本次只替换 BibTeX 文件入口，未新增联网或全文核验。

| 项目 | 数量/状态 |
|---|---:|
| 原始矩阵记录 | 33 |
| 明确重复组 | 3 组 |
| 去重后唯一候选 | 30 |
| 当前主表中 DOI 可从当前 BibTeX 对齐 | 30 条 |
| 当前主表中仍需在线或全文核验 | 30 条 |
| 当前已由 Crossref 完整核验且能直接用于书目信息的 DOI 页面 | 3 条 |
| Crossref 请求异常 | 1 条：`10.1016/j.jmsy.2024.09.011` |

> Crossref 已核验的 3 条为：`10.1080/00207543.2020.1870013`、`10.1016/j.ifacol.2018.08.474`、`10.1016/j.jmsy.2025.06.017`。它们是否属于主表中的某一条，必须以标题匹配为准；目前无法仅凭 DOI 出现在早期对话中的事实强行归入主表。

## 3. 去重后的唯一文献表

| 唯一ID | 原矩阵数据行 | 标准化标题 | DOI | 作者 | 年份 | 期刊/会议 | 类型 | 元数据状态 | 证据来源 | 备注 |
|---|---:|---|---|---|---:|---|---|---|---|---|
| U01 | 1 | Dynamic integrated scheduling of machines and AGVs in flexible job shop considering AGV failures | 10.1016/j.cor.2026.107524 | Li, Jia; Li, Minghui; Li, Guoqiang; Shi, Xiaoqiu | 2026 | Computers \& Operations Research | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | AGV 故障主题 |
| U02 | 2 | Dynamic Integrated Scheduling of Production Equipment and Automated Guided Vehicles in a Flexible Job Shop Based on Deep Reinforcement Learning | 10.3390/pr12112423 | Wang, Jingrui; Li, Yi; Zhang, Zhongwei; Wu, Zhaoyun; Wu, Lihui; Jia, Shun; Peng, Tao | 2024 | Processes | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | DOI 需要继续与 Crossref/出版社元数据匹配 |
| U03 | 3 | Deep Reinforcement Learning for Dynamic Flexible Job Shop Scheduling with Random Job Arrival | 10.3390/pr10040760 | Chang, Jingru; Yu, Dong; Hu, Yi; He, Wuwei; Yu, Haoyu | 2022 | Processes | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 随机新作业到达主题 |
| U04 | 4 | Solving Flexible Job-Shop Scheduling Problem with Heterogeneous Graph Neural Network Based on Relation and Deep Reinforcement Learning | 10.3390/machines12080584 | Tang, Hengliang; Dong, Jinda | 2024 | Machines | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 静态 FJSP 图 DRL |
| U05 | 5 | Learning-enabled flexible job-shop scheduling for scalable smart manufacturing | 10.1016/j.jmsy.2024.09.011 | Moon, Sihoon; Lee, Sanghoon; Park, Kyung-Joon | 2024 | Journal of Manufacturing Systems | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | AGV 与规模泛化主题 |
| U06 | 6 | Flexible Job Shop Scheduling Optimization with Machine and AGV Integration Based on Improved NSGA-II | 10.5755/j01.itc.53.4.37410 | Liu, Yong; Huang, Tao; Chen, Yong; Liu, Li; Guo, Tao | 2024 | Information Technology and Control | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 静态多目标 FJSP-AGV |
| U07 | 7 | Multi-Agent Reinforcement Learning for Job Shop Scheduling in Dynamic Environments | 10.3390/su16083234 | Pu, Yu; Li, Fang; Rahimifard, Shahin | 2024 | Sustainability | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | AGV 是否存在待核验 |
| U08 | 8 | Multi-faceted dynamic flexible job shop scheduling via heterogeneous graph transformer and deep reinforcement learning | 10.1016/j.eswa.2025.130532 | Liu, Guang; Liao, Mengqi; Chen, Wei; Zhang, Zhiyu; Wan, Huaiyu; Lin, Youfang | 2025 | Expert Systems with Applications | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 动态 FJSP 与异构图 Transformer |
| U09 | 9 | Digital Twin-Driven Adaptive Scheduling for Flexible Job Shops | 10.3390/su14095340 | Liu, Lilan; Guo, Kai; Gao, Zenggui; Li, Jiaying; Sun, Jiachen | 2022 | Sustainability | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 数字孪生动态调度 |
| U10 | 10 | Dynamic scheduling of flexible job-shop considering automatic guided vehicle transportation via deep reinforcement learning | 10.1016/j.future.2026.108450 | Ren, Fujie; Liu, Haibin; Huang, He; Cao, Yang | 2026 | Future Generation Computer Systems | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 动态 FJSP-AGV |
| U11 | 11, 19 | A Dynamic Scheduling Method Combining Iterative Optimization and Deep Reinforcement Learning to Solve Sudden Disturbance Events in a Flexible Manufacturing Process | 10.3390/math13010004 | Yan, Jun; Zhao, Tianzuo; Zhang, Tao; Chu, Hongyan; Yang, Congbin; Zhang, Yueze | 2025 | Mathematics | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 明确重复组 A |
| U12 | 12 | Flexible job shop scheduling via deep reinforcement learning with meta-path-based heterogeneous graph neural network | 10.1016/j.knosys.2024.111940 | Wan, Lanjun; Fu, Long; Li, Changyun; Li, Keqin | 2024 | Knowledge-Based Systems | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 元路径异构图 DRL |
| U13 | 13 | A Tree neural network deep reinforcement learning for flexible job shop scheduling with transportation constraints | 10.1016/j.swevo.2025.102102 | Geng, Yadian; Zhao, Ning | 2025 | Swarm and Evolutionary Computation | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 运输约束是否为 AGV 待核验 |
| U14 | 14, 26 | Flexible Manufacturing Systems intralogistics: Dynamic optimization of AGVs and tool sharing using Colored-Timed Petri Nets and actor-critic RL with actions masking | 10.1016/j.jmsy.2025.06.017 | Lassoued, Sofiene; Baheti, Laxmikant Shrikant; Weiß-Borkowski, Nathalie; Lier, Stefan; Schwung, Andreas | 2025 | Journal of Manufacturing Systems | 期刊论文 | 当前 BibTeX 元数据已对齐；Crossref 核验记录沿用此前整理结果 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 明确重复组 B；Crossref 页显示卷 82，页 405-419 |
| U15 | 15 | Dynamic Scheduling for Flexible Manufacturing Systems Based on Multi-Agent Deep Reinforcement Learning and Petri Nets | 10.48550/arXiv.2606.31737 | He, Zhou; Li, Ning; Liu, Ruotian; Li, Liang; Seatzu, Carla | 2026 | arXiv | 预印本/软件环境 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | Petri 网与 MARL |
| U16 | 16, 29 | Dynamic scheduling for flexible job shop based on MachineRank algorithm and reinforcement learning | 10.1038/s41598-024-79593-8 | Ren, Fujie; Liu, Haibin | 2024 | Scientific Reports | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 明确重复组 C |
| U17 | 17 | Learning to schedule job-shop problems: Representation and policy learning using graph neural network and reinforcement learning | 10.1080/00207543.2020.1870013 | Park, Junyoung; Chun, Jaehyeong; Kim, Sang Hun; Kim, Youngkook; Park, Jinkyoo | 2021 | International Journal of Production Research | 期刊论文 | 当前 BibTeX 元数据已对齐；Crossref 核验记录沿用此前整理结果 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | Crossref 页显示卷 59(11)，页 3360-3377 |
| U18 | 18 | Integrated Scheduling of Multi-Objective Job Shops and Material Handling Robots with Reinforcement Learning Guided Meta-Heuristics | 10.3390/math13010102 | Xu, Zhangying; Jia, Qi; Gao, Kaizhou; Fu, Ya-ping; Yin, Li; Sun, Qiangqiang | 2025 | Mathematics | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | MHR 是否为 AGV 待核验 |
| U19 | 20 | A heterogeneous graph attention-enhanced deep reinforcement learning framework for flexible job shop scheduling problem with variable sublots | 10.1016/j.engappai.2025.111375 | Yang, Zipeng; Li, Xinyu; Gao, Liang; Liu, Qihao | 2025 | Engineering Applications of Artificial Intelligence | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 可变批量 FJSP |
| U20 | 21 | Meta-relation-based heterogeneous graph neural network with deep reinforcement learning for flexible job shop scheduling | 10.1016/j.eswa.2025.128411 | Zhang, Yuzhi; Dong, Shidu; Yuan, Zhenfang; Wen, Ting; Xiao, Jianfeng; Diao, Zhuo | 2025 | Expert Systems with Applications | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 元关系异构图 |
| U21 | 22 | Dynamic Scheduling Method for Job-Shop Manufacturing Systems by Deep Reinforcement Learning with Proximal Policy Optimization | 10.3390/su14095177 | Zhang, Ming; Lü, Yang; Hu, Youxi; Amaitik, Nasser; Xu, Yuchun | 2022 | Sustainability | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | PPO 与机器故障主题 |
| U22 | 23 | Digital Twin-Driven AGV Scheduling and Routing in Automated Container Terminals | 10.3390/math11122678 | Lou, Ping; Zhong, Yutong; Hu, Jiwei; Fan, Chuannian; Chen, Xiao | 2023 | Mathematics | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | AGV 路由背景 |
| U23 | 24 | Integrated scheduling of multi-objective lot-streaming hybrid flowshop with AGV under dynamic environments | 10.5267/j.jiiec.2025.9.003 | Tang, Hongtao; Zhang, Xuewei; Huang, Jiawei; Xu, Xuesong; Lu, Jiansha | 2025 | International Journal of Industrial Engineering Computations | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 动态混合流水车间-AGV |
| U24 | 25 | Machine learning-driven solutions for sustainable and dynamic flexible job shop scheduling under worker absences and renewable energy variability | 10.1016/j.cor.2025.107323 | Destouet, Candice; Tlahig, Houda; Bettayeb, Belgacem; Mazari, Bélahcène | 2025 | Computers \& Operations Research | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 工人缺勤与能源变化 |
| U25 | 27 | Edge computing-based real-time scheduling for digital twin flexible job shop with variable time window | 10.1016/j.rcim.2022.102435 | Wang, Jin; Liu, Yang; Ren, Shan; Wang, Chuang; Ma, Shuaiyin | 2022 | Robotics and Computer-Integrated Manufacturing | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 数字孪生与边缘计算 |
| U26 | 28 | Task Scheduling with Mobile Robots—A Systematic Literature Review | 10.3390/robotics14060075 | Rema, Catarina; Costa, Pedro; Silva, Manuel Fernando; Pires, Eduardo J. Solteiro | 2025 | Robotics | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 移动机器人任务调度综述 |
| U27 | 30 | Analysis of Multi-AGVs Management System and Key Issues: A Review | 10.32604/cmes.2022.019770 | Lu, Wenhao; Guo, Shuai; Song, Tao; Li, Yuwen | 2022 | Computer Modeling in Engineering \& Sciences | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 多 AGV 管理综述 |
| U28 | 31 | A review of the applications of multi-agent reinforcement learning in smart factories | 10.3389/frobt.2022.1027340 | Bahrpeyma, Fouad; Reichelt, Dirk | 2022 | Frontiers in Robotics and AI | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 智能工厂 MARL 综述 |
| U29 | 32 | A Systematic Review on Reinforcement Learning for Industrial Combinatorial Optimization Problems | 10.3390/app15031211 | Martins, Miguel S. E.; Sousa, João M. C.; Vieira, Susana Margarida | 2025 | Applied Sciences | 期刊论文 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | 工业组合优化 RL 综述；原始逐篇笔记未覆盖 |
| U30 | 33 | A Reinforcement Learning Environment For Job-Shop Scheduling | 10.48550/arXiv.2104.03760 | Tassel, Pierre; Gebser, Martin; Schekotihin, Konstantin | 2021 | arXiv | 预印本/软件环境 | 与当前 BibTeX 的标题、作者、年份、期刊和 DOI 对齐；本次未新增在线或全文核验 | `文献矩阵_Zotero.md` 矩阵与结构化笔记；[`zotero_literature.bib`](zotero_literature.bib) | JSSP 强化学习环境 |

## 4. 明确重复记录合并表

| 重复组 | 原矩阵数据行 | 保留记录 | 合并依据 |
|---|---|---|---|
| A | 11、19 | U11 | 标准化标题完全一致：`A Dynamic Scheduling Method Combining Iterative Optimization and Deep Reinforcement Learning to Solve Sudden Disturbance Events in a Flexible Manufacturing Process` |
| B | 14、26 | U14 | 标准化标题完全一致；且 DOI `10.1016/j.jmsy.2025.06.017` 与 Crossref 标题一致 |
| C | 16、29 | U16 | 标准化标题完全一致：`Dynamic scheduling for flexible job shop based on MachineRank algorithm and reinforcement learning` |

## 5. 补充 DOI 候选记录

以下 DOI 保留自此前 Zotero 元数据整理记录，主比较矩阵未单列这些 DOI；其中未与 U01—U30 精确匹配的候选继续单独列出。

| 补充ID | 标题 | DOI | 当前状态 |
|---|---|---|---|
| S01 | Dynamic scheduling for flexible job shop with insufficient transportation resources via graph neural network and deep reinforcement learning | `10.1016/j.cie.2023.109718` | DOI 已显示；与 U01—U30 的标题未精确匹配，待核验是否为漏列记录 |
| S02 | Dynamic Integrated Scheduling of Production Equipment and Automated Guided Vehicles in a Flexible Job Shop Based on Deep Reinforcement Learning | `10.3390/pr12112423` | 与 U02 标题精确匹配，可回填 U02 的 DOI 来源 |
| S03 | A flexible job shop scheduling method based on heterogeneous disjunctive graph and deep reinforcement learning | `10.1016/j.engappai.2025.111356` | DOI 已显示；与 U01—U30 的标题未精确匹配，待核验是否为漏列记录 |
| S04 | Flexible job shop scheduling via deep reinforcement learning with meta-path-based heterogeneous graph neural network | `10.1016/j.knosys.2024.111940` | 与 U12 标题精确匹配，可回填 U12 的 DOI 来源 |
| S05 | Digital twin-enabled dynamic scheduling with preventive maintenance using a double-layer Q-learning algorithm | `10.1016/j.cor.2022.105823` | DOI 已显示；与 U01—U30 的标题未精确匹配，待核验是否为漏列记录 |

## 6. 已知 DOI 的标准化规则

- 删除 `https://doi.org/`、`http://dx.doi.org/`、Markdown 反引号和首尾空格；
- 删除 DOI 末尾由引用格式带入的句号、逗号、分号和右括号；
- DOI 比较统一转为小写；
- 保留原 DOI 作为主表值时，使用 `https://doi.org/` 之外的裸 DOI 格式；
- DOI 相同即使标题大小写、连字符或期刊缩写不同，也先视为同一记录，再用 Crossref 标题复核。

## 7. 当前限制与下一步核验边界

1. 30 条唯一候选并不等于 30 条都已完成全文核读；算法、状态、动作、奖励、动态事件和实验协议仍以结构化证据文件中的“未说明/待核验”为准。
2. 30 条候选文献的 DOI、作者、年份和期刊均已与当前 BibTeX 按标题对齐；U14、U17 的 Crossref 核验记录沿用此前结果，本次未重新联网核验。
3. 此前 `10.1016/j.jmsy.2024.09.011` 的 Crossref 请求返回 HTTP 500；当前 BibTeX 中保留了该条目的书目信息，但本次不将其标记为新的 Crossref 核验结果。
4. Crossref 页面中的 DOI 若无法与主表标题精确匹配，暂不并入主表；这避免把早期对话中出现的其他候选论文误认为矩阵中的同一条记录。
5. 本表不对不同论文的 makespan 数值做横向排名，也不把题目或摘要中的方法词扩展为正文已证实的算法细节。
