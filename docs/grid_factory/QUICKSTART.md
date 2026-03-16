# Joint Simulation (joint_sim) 快速入门指南

本文档说明如何集成、使用和二次开发 `joint_sim` 联合仿真库。该库提供 JobShop + MAPF 联合仿真系统，支持 JSSP 作业调度与 AGV 路径规划的联合优化。

## 目录

- [安装与集成](#安装与集成)
- [快速开始](#快速开始)
- [核心组件](#核心组件)
- [Gym 环境接口](#gym-环境接口)
- [Proxy 服务层](#proxy-服务层)
- [JSON 配置加载](#json-配置加载)
- [前端集成](#前端集成)
- [二次开发](#二次开发)

---

## 安装与集成

### 从 PyPI 安装

```bash
uv add joint-sim
```

> 如果搜索不到，请清理缓存：uv cache clean

或使用 pip：

```bash
pip install joint-sim
```

### 从源码安装

```bash
git clone https://github.com/skyrimforest/SkyEngine.git
cd SkyEngine
uv sync
```

> 注意，当你安装joint-sim库后，就可以使用grid_factory等相关配置了。
> 另外，推荐单独使用joint-sim强化学习库进行算法复现与研究。

---

## 快速开始

### 基本使用

```python
from joint_sim import (
    create_simulation,
    GreedyScheduler,
    RuleBasedScheduler,
    PrioritizedRouter,
    SimpleRouter,
)

# 方式一：使用便捷函数创建仿真
sim = create_simulation(
    n_machines=6,
    n_agvs=3,
    n_jobs=10,
    grid_size=(20, 20),
    seed=42
)

# 方式二：使用配置类创建
from joint_sim import FactoryConfig, JointSimulation

config = FactoryConfig(
    n_machines=6,
    n_agvs=3,
    n_jobs=10,
    grid_size=(20, 20),
    n_ops_per_job=(3, 6),      # 每个工件 3-6 个操作
    op_duration_range=(3, 10), # 每个操作 3-10 时间单位
    seed=42,
)
sim = JointSimulation(config)

# 设置调度器和路由器
sim.set_scheduler(GreedyScheduler(priority_rule='lrpt'))
sim.set_router(PrioritizedRouter())

# 按轮次运行仿真
for step in range(1000):
    state = sim.observe()
    decisions = sim.decide()
    sim.execute(decisions)

    # 渲染（可选）
    if step % 10 == 0:
        sim.render(f'frames/frame_{step:04d}.svg')

    if sim.is_done():
        break

# 获取指标
metrics = sim.get_metrics()
print(metrics.to_dict())
```

### 使用便捷的 step() 方法

```python
# 一步完成 observe -> decide -> execute
for step in range(1000):
    state, done = sim.step()
    if done:
        break
```

### 运行完整仿真

```python
# 自动运行直到完成
metrics = sim.run(
    max_steps=5000,
    render_interval=10,   # 每 10 步渲染一次
    render_dir='frames',
    verbose=True
)
print(f"完成工件数: {metrics.n_completed_jobs}")
print(f"总 AGV 移动距离: {metrics.total_agv_distance}")
```

---

## 核心组件

### 调度器 (Scheduler)

调度器负责工件加工顺序和机器任务分配：

```python
from joint_sim import GreedyScheduler, RuleBasedScheduler, RandomScheduler

# 贪心调度器 - 基于优先规则
scheduler = GreedyScheduler(priority_rule='lrpt')  # 最长剩余处理时间优先

# 规则调度器 - 基于启发式规则
scheduler = RuleBasedScheduler()

# 随机调度器 - 随机分配（用于基准测试）
scheduler = RandomScheduler()

sim.set_scheduler(scheduler)
```

### 路由器 (Router)

路由器负责 AGV 路径规划和任务分配：

```python
from joint_sim import PrioritizedRouter, SimpleRouter, RandomRouter

# 优先级路由器 - 基于 A* 的优先级路径规划
router = PrioritizedRouter(
    assigner_type='cost',  # 'cost' 或 'greedy'
)

# 简单路由器 - 最短路径规划
router = SimpleRouter()

# 随机路由器 - 随机分配 AGV 任务
router = RandomRouter()

sim.set_router(router)
```

### 状态观察

```python
state = sim.observe()

# 访问状态信息
print(f"当前时间: {state.time}")
print(f"AGV 数量: {len(state.agvs)}")
print(f"机器数量: {len(state.machines)}")
print(f"已完成工件: {state.n_completed_jobs}")
print(f"空闲 AGV 数: {state.n_idle_agvs}")
print(f"空闲机器数: {state.n_idle_machines}")

# AGV 信息
for agv in state.agvs:
    print(f"AGV {agv.agv_id}: 位置={agv.position}, 状态={agv.status}")
    if agv.carrying:
        print(f"  携带工件: {agv.carrying.job_id}")

# 机器信息
for machine in state.machines:
    print(f"Machine {machine.machine_id}: 状态={machine.status}, 队列={machine.queue_length}")
```

### 指标收集

```python
metrics = sim.get_metrics()

# 指标字典
metrics_dict = metrics.to_dict()
# {
#     'n_completed_jobs': 10,
#     'total_agv_distance': 150,
#     'avg_waiting_time': 5.2,
#     ...
# }
```

---

## Gym 环境接口

`joint_sim` 提供符合 OpenAI Gymnasium 标准的强化学习环境：

```python
import gymnasium as gym
from joint_sim import JointSimGymEnv, FactoryConfig

# 创建环境
config = FactoryConfig(
    n_machines=6,
    n_agvs=3,
    n_jobs=10,
    grid_size=(20, 20),
)

env = JointSimGymEnv(
    config=config,
    assigner_type='cost',      # 'cost' 或 'greedy'
    max_episode_steps=10000,
    reward_scale=1.0
)

# 查看空间定义
print(env.observation_space)  # Dict 空间
print(env.action_space)       # Dict 空间

# 重置环境
obs, info = env.reset(seed=42)

# 执行动作
action = {
    'agv_assignments': np.array([0, -1, 1], dtype=np.int32)  # -1 表示不分配
}
obs, reward, terminated, truncated, info = env.step(action)

# 渲染
env.render(mode='svg', filepath='frame.svg')

# 获取可用动作
available_actions = env.get_available_actions()
```

### 观察空间

```python
observation_space = spaces.Dict({
    'grid': spaces.Box(low=0, high=3, shape=(20, 20), dtype=np.int32),
    'agv_positions': spaces.Box(low=0, high=20, shape=(n_agvs, 2), dtype=np.int32),
    'agv_status': spaces.Box(low=0, high=3, shape=(n_agvs,), dtype=np.int32),
    'agv_carrying': spaces.Box(low=-1, high=n_jobs, shape=(n_agvs,), dtype=np.int32),
    'machine_status': spaces.Box(low=0, high=1, shape=(n_machines,), dtype=np.int32),
    'machine_queue_length': spaces.Box(low=0, high=n_jobs, shape=(n_machines,), dtype=np.int32),
    'time': spaces.Box(low=0, high=max_time, shape=(), dtype=np.int32),
})
```

### 动作空间

```python
action_space = spaces.Dict({
    'agv_assignments': spaces.Box(
        low=-1, high=n_jobs-1,
        shape=(n_agvs,), dtype=np.int32
    )
})
# 每个 AGV 分配一个任务 ID，-1 表示不分配任务
```

---

## Proxy 服务层

`GridFactoryProxy` 提供异步执行和 SSE 流式传输的代理服务层，适用于 Web 服务集成：

### 基本使用

```python
from joint_sim.proxy import GridFactoryProxy, ExecutionStatus

# 创建代理
proxy = GridFactoryProxy()

# 设置算法配置
proxy.set_algorithm("greedy:prioritized:standard")
# 格式: "scheduler:router:complexity"

# 或单独设置
proxy.set_scheduler("greedy")
proxy.set_router("prioritized")

# 获取可用算法列表
algorithms = proxy.get_algorithm_list()
# {
#     'schedulers': [...],
#     'routers': [...],
#     'complexity_options': [...],
#     'presets': [...]
# }
```

### 从配置字典初始化

```python
# 准备配置数据（来自 JSON 文件）
config = {
    "id": "factory-001",
    "name": "示例工厂",
    "topology": {
        "gridWidth": 20,
        "gridHeight": 14,
        "zones": [...],
        "machines": {...},
        "waypoints": {...}
    },
    "agvs": [...],
    "jobs": [...],
    "complexity": "standard"
}

# 设置配置并初始化
proxy.set_config(config)
await proxy.initialize()
```

### 生命周期控制

```python
# 启动执行
await proxy.start()

# 暂停执行
await proxy.pause()

# 重置环境
await proxy.reset()

# 停止执行
await proxy.stop()

# 清理资源
await proxy.cleanup()

# 检查状态
if proxy.is_running():
    print("正在运行")
if proxy.is_paused():
    print("已暂停")
```

### 获取状态快照

```python
# 获取状态快照（用于 SSE 推送）
snapshot = await proxy.get_state_snapshot()
# {
#     'timestamp': 'T+100',
#     'env_timeline': '100',
#     'grid_state': {
#         'positions_xy': [[x1, y1], [x2, y2], ...],
#         'is_active': [True, False, ...]
#     },
#     'machines': {
#         'M1': {'id': 'M1', 'status': 'WORKING', 'load': 2},
#         ...
#     },
#     'agvs': [...],
#     'jobs': [...],
#     'summary': {...}
# }

# 获取指标快照
metrics = await proxy.get_metrics_snapshot()

# 获取控制状态
status = await proxy.get_control_status()
```

### SSE 事件格式

```python
# 获取 SSE 事件
state_events = await proxy.get_state_events()      # [('state', snapshot)]
metrics_events = await proxy.get_metrics_events()  # [('metrics', metrics)]
control_events = await proxy.get_control_events()  # [('control', status)]
```

---

## JSON 配置加载

### 配置文件格式

```json
{
    "id": "factory-001",
    "name": "示例工厂",
    "version": "1.0.0",
    "description": "一个简单的工厂示例",
    "topology": {
        "gridWidth": 20,
        "gridHeight": 14,
        "zones": [
            {
                "id": "zone-1",
                "name": "工作区A",
                "type": "workarea",
                "area": {"x": 0, "y": 0, "w": 5, "h": 5},
                "color": "#4CAF50"
            }
        ],
        "machines": {
            "M1": {
                "id": "M1",
                "name": "机器1",
                "location": [2, 3],
                "size": [1, 1],
                "status": "IDLE"
            }
        },
        "waypoints": {
            "dock-1": {
                "id": "dock-1",
                "type": "dock",
                "location": [0, 0],
                "name": "停靠点1"
            }
        }
    },
    "agvs": [
        {
            "id": 0,
            "name": "AGV-0",
            "initialLocation": [1, 1],
            "velocity": 1.0,
            "capacity": 100
        }
    ],
    "jobs": [
        {
            "job_id": 0,
            "operations": [
                {"machine_id": 0, "duration": 5},
                {"machine_id": 1, "duration": 3}
            ],
            "arrival_time": 0
        }
    ]
}
```

### 加载配置

```python
from joint_sim.io import (
    load_factory_config,
    to_factory_config,
    to_factory_layout,
    to_instance,
    load_and_parse,
)

# 方式一：分步加载
json_config = load_factory_config('factory.json')
factory_config = to_factory_config(json_config, n_jobs=10)
factory_layout = to_factory_layout(json_config)
problem_instance = to_instance(json_config, complexity='standard')

# 方式二：一步加载
factory_config, factory_layout, problem_dict = load_and_parse('factory.json', n_jobs=10)

# 根据复杂度创建问题实例
simple_instance = to_instance(json_config, complexity='simple')    # 无 docks, 无 zones
standard_instance = to_instance(json_config, complexity='standard') # 无 docks, 有 zones
full_instance = to_instance(json_config, complexity='full')        # 有 docks, 有 zones
```

### 配置类说明

```python
from joint_sim.io import (
    ZoneConfig,        # 区域配置
    MachineConfig,     # 机器配置
    WaypointConfig,    # 航点配置
    AGVConfig,         # AGV 配置
    OperationConfig,   # 操作配置
    JobConfig,         # 工件配置
    TopologyConfig,    # 拓扑配置
    FactoryJSONConfig, # 完整工厂配置
)
```

---

## 前端集成

### 数据流

```
ConfigPanel 上传配置
    ↓
store.loadConfigFromFile(config)
    ↓
设置 currentConfigId
    ↓
FactoryPlayerSSE 监听 currentConfigId
    ↓
configLoaded = true
    ↓
显示 FactoryVisualization
    ↓
使用 topologyConfig + currentState 渲染
```

### SSE 连接示例

```javascript
// 连接 SSE 端点
const eventSource = new EventSource('/api/factory/stream');

eventSource.addEventListener('state', (event) => {
    const snapshot = JSON.parse(event.data);
    // 更新可视化
    updateVisualization(snapshot);
});

eventSource.addEventListener('metrics', (event) => {
    const metrics = JSON.parse(event.data);
    // 更新指标面板
    updateMetrics(metrics);
});

eventSource.addEventListener('control', (event) => {
    const status = JSON.parse(event.data);
    // 更新控制状态
    updateControlStatus(status);
});
```

### 状态快照格式

```typescript
interface StateSnapshot {
    timestamp: string;       // "T+100"
    env_timeline: string;    // "100"
    grid_state: {
        positions_xy: number[][];  // [[x1, y1], [x2, y2], ...]
        is_active: boolean[];      // [true, false, ...]
    };
    machines: {
        [key: string]: {
            id: string;
            position: number[];
            status: string;
            load: number;
            queue_length: number;
            current_job: number | null;
        };
    };
    agvs: Array<{
        id: number;
        position: number[];
        status: string;
        is_active: boolean;
        carrying: { job_id: number; op_index: number } | null;
    }>;
    active_transfers: Array<{
        agv_id: number;
        job_id: number;
        from_op: number;
        status: string;
    }>;
    jobs: Array<{
        id: number;
        status: string;
        current_op_index: number;
        is_completed: boolean;
    }>;
    summary: {
        n_completed_jobs: number;
        n_active_jobs: number;
        n_idle_agvs: number;
        n_idle_machines: number;
        total_queue_length: number;
    };
}
```

---

## 二次开发

### 自定义调度器

```python
from joint_sim.scheduler import Scheduler
from joint_sim.entities import TransportRequest
from joint_sim.state import FactoryState

class MyScheduler(Scheduler):
    def get_transport_requests(self, state: FactoryState) -> list[TransportRequest]:
        """获取待分配的运输请求"""
        requests = []
        for job in state.jobs:
            if job.status.value == 'waiting' and job.current_operation:
                # 创建运输请求
                requests.append(TransportRequest(
                    job_id=job.job_id,
                    from_machine=job.last_machine_id,
                    to_machine=job.current_operation.machine_id,
                    from_position=(0, 0),
                    to_position=(0, 0),
                    priority=job.total_remaining_time,
                    request_time=state.time
                ))
        return requests

# 使用自定义调度器
sim.set_scheduler(MyScheduler())
```

### 自定义路由器

```python
from joint_sim.router import Router, RoutePlan
from joint_sim.entities import TransportRequest
from joint_sim.state import FactoryState

class MyRouter(Router):
    def plan_routes(
        self,
        state: FactoryState,
        requests: list[TransportRequest]
    ) -> dict[int, RoutePlan]:
        """规划 AGV 路径"""
        plans = {}

        for agv in state.agvs:
            if not agv.is_idle:
                continue

            # 选择最近的请求
            best_request = None
            best_distance = float('inf')

            for req in requests:
                dist = abs(agv.position[0] - req.from_position[0]) + \
                       abs(agv.position[1] - req.from_position[1])
                if dist < best_distance:
                    best_distance = dist
                    best_request = req

            if best_request:
                # 规划路径（使用 A* 等）
                path = self._plan_path(state, agv.position, best_request.to_position)
                if path:
                    plans[agv.agv_id] = RoutePlan(
                        agv_id=agv.agv_id,
                        path=path,
                        request=best_request
                    )

        return plans

    def _plan_path(self, state, start, goal):
        # 实现路径规划算法
        pass

# 使用自定义路由器
sim.set_router(MyRouter())
```

### 从数据集加载地图

```python
from joint_sim import FactoryConfig

# 从 MAPF 数据集加载地图
config = FactoryConfig.from_dataset(
    map_name='random-32-32-10',
    dataset_name='mapf_gpt',  # 或 'mapf_data'
    n_machines=6,
    n_agvs=3,
    n_jobs=10,
)

sim = JointSimulation(config)
```

### 可视化定制

```python
from joint_sim import SVGVisualizer, VisualizerConfig

# 创建可视化器
visualizer = SVGVisualizer(VisualizerConfig(
    show_heatmap=True,
    heatmap_decay=0.95,
    cell_size=30,
    margin=20,
))

# 渲染状态
svg = visualizer.render(state)

# 保存到文件
visualizer.save(state, 'output.svg')
```

---

## 相关链接

- **GitHub 仓库**: <https://github.com/skyrimforest/SkyEngine>
- **问题反馈**: <https://github.com/skyrimforest/SkyEngine/issues>
- **维护者**: @SkyrimForest 吴昊

---

## API 参考

### 主要导出

```python
from joint_sim import (
    # 仿真
    JointSimulation,
    create_simulation,

    # 配置
    FactoryConfig,
    FactoryLayout,
    generate_layout,
    generate_jobs,

    # 实体
    Machine, Job, Operation, AGV,
    TransportRequest, TransportTask,
    MachineStatus, AGVStatus, JobStatus, JobLocationType,

    # 状态
    FactoryState,
    SimulationSnapshot,

    # 调度器
    Scheduler,
    GreedyScheduler,
    RuleBasedScheduler,
    RandomScheduler,

    # 路由器
    Router,
    PrioritizedRouter,
    SimpleRouter,
    RandomRouter,
    RoutePlan,

    # 可视化
    SVGVisualizer,
    VisualizerConfig,

    # 指标
    Metrics,
    MetricsCollector,

    # Gym 环境
    JointSimGymEnv,

    # 问题定义
    ProblemComplexity,
    TransportTimeMode,
    Position,
    OperationSpec,
    JobSpec,
    MachineSpec,
    DockSpec,
    AGVSpec,
    ZoneSpec,
    JointSchedulingInstance,
    SimpleJointInstance,
    StandardJointInstance,
    FullJointInstance,

    # Proxy
    GridFactoryProxy,
    ExecutionStatus,
)
```

### IO 模块

```python
from joint_sim.io import (
    ZoneConfig,
    MachineConfig,
    WaypointConfig,
    AGVConfig,
    OperationConfig,
    JobConfig,
    TopologyConfig,
    FactoryJSONConfig,
    load_factory_config,
    to_factory_config,
    to_factory_layout,
    to_instance,
    load_and_parse,
)
```
