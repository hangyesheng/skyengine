+---------------------------------------------------+
|                 SkyDagEnvironment (env)           |
|---------------------------------------------------|
|                                                   |
|  +-------------------+                            |
|  |   Coordinator     |  <--- 调度核心             |
|  |-------------------|                            |
|  | + JobSolver       |  负责任务调度（分配机器）  |
|  | + RouteSolver     |  负责路径规划（移动AGV）   |
|  |-------------------|                            |
|  | + decide(obs) --> {agent_actions, machine_actions} |
|  +-------------------+                            |
|                                                   |
|  +-------------------+                            |
|  |   Env Components  |                            |
|  |  (AGVs, Machines, Jobs...)                    |
|  +-------------------+                            |
|                                                   |
+---------------------------------------------------+



| 模块            | 职责         | 输入                       | 输出                                 |
| ------------- | ---------- | ------------------------ | ---------------------------------- |
| `JobSolver`   | 任务调度、分配机器  | 当前job/machine状态          | machine_actions, transfer_requests |
| `RouteSolver` | 路径规划、AGV移动 | transfer_requests + 环境状态 | agent_actions                      |
| `Coordinator` | 联合两层策略     | obs                      | {machine_actions, agent_actions}   |
| `Env`         | 执行动作并推进时间  | 来自 coordinator 的决策       | 新 obs, reward, done, info          |

