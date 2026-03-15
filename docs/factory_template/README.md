# Factory 系列文档模板

本目录提供工厂仿真系统的文档模板，用于快速创建新的工厂类型文档。

---

## 一、仓库文档结构说明

```
docs/
├── factory_template/              # 文档模板目录（本目录）
│   └── README.md                  # 本文件，包含所有模板内容
├── grid_factory/                  # Grid 工厂文档
│   ├── QUICKSTART.md             # 快速开始
│   └── img/                      # 图片资源
├── packet_factory/                # Packet 工厂文档
│   └── QUICKSTART.md
├── quick_start/                   # 通用快速开始
│   └── README.md
└── pics/                          # 通用图片资源
```

---

## 二、使用方法

### 创建新工厂文档

1. 在 `docs/` 下创建新目录，如 `your_factory/`

2. 创建必要的文档文件：
   ```
   your_factory/
   ├── QUICKSTART.md              # 快速开始（必需）
   └── img/                       # 图片资源目录（可选）
   ```

3. 复制下面的模板内容到对应文件，替换占位符

### 文档命名规范

| 文件类型 | 命名规范 | 说明 |
|---------|---------|------|
| 快速开始 | QUICKSTART.md | 大写，便于识别 |
| 设计文档 | DESIGN.md | 大写，便于识别 |
| API 参考 | API.md | 大写，便于识别 |
| 图片资源 | img/ 目录 | 放置示意图、流程图等 |

---

## 三、快速开始文档模板 (QUICKSTART.md)

```markdown
# [FactoryName] 快速入门指南

本文档说明如何集成、使用和二次开发 `[FactoryName]` 工厂仿真系统。

> 简要描述工厂系统的用途和核心功能

## 目录

- [安装与集成](#安装与集成)
- [快速开始](#快速开始)
- [核心组件](#核心组件)
- [配置说明](#配置说明)
- [API 参考](#api-参考)
- [常见问题](#常见问题)

---

## 安装与集成

### 从 PyPI 安装

```bash
uv add [package-name]
```

或使用 pip：

```bash
pip install [package-name]
```

### 从源码安装

```bash
git clone https://github.com/skyrimforest/SkyEngine.git
cd SkyEngine
uv sync
```

---

## 快速开始

### 基本使用

```python
from [package_name] import create_factory

factory = create_factory(
    config_path="config.json",
)

await factory.start()
```

---

## 核心组件

| 组件 | 说明 |
|------|------|
| Simulator | 仿真引擎 |
| Scheduler | 调度器 |
| Router | 路由器 |
| Proxy | 代理服务层 |

---

## 配置说明

### 配置文件格式

```json
{
    "id": "factory-001",
    "name": "示例工厂",
    "topology": { ... },
    "agvs": [ ... ],
    "jobs": [ ... ]
}
```

### 配置参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| id | string | 是 | - | 工厂ID |
| name | string | 是 | - | 工厂名称 |

---

## API 参考

### 主要类

```python
class Factory:
    def start(self):
        """启动仿真"""
        pass

    def stop(self):
        """停止仿真"""
        pass
```

---

## 相关链接

- **GitHub 仓库**: https://github.com/skyrimforest/SkyEngine
- **问题反馈**: https://github.com/skyrimforest/SkyEngine/issues
```

---

## 四、设计文档模板 (DESIGN.md)

```markdown
# [FactoryName] 设计文档

本文档描述系统的架构设计和实现细节。

## 系统概述

### 设计目标

- 目标一
- 目标二

### 技术栈

| 技术 | 用途 | 版本 |
|------|------|------|
| Python | 后端核心 | 3.10+ |

---

## 架构设计

### 系统架构

```
┌─────────────────────────────────────────┐
│            前端层 (Frontend)            │
│  ConfigPanel | Visualization | Metrics   │
└─────────────────────────────────────────┘
                    │ SSE
                    ▼
┌─────────────────────────────────────────┐
│            服务层 (Service)             │
│     Proxy | Controller | SSE Handler    │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│            核心层 (Core)                │
│  Scheduler | Router | Simulator       │
└─────────────────────────────────────────┘
```

---

## 核心模块

### 仿真引擎

```python
class SimulationEngine:
    def step(self) -> SimulationResult:
        """执行一步仿真"""
        pass
```

### 调度器

```python
class Scheduler:
    def schedule(self, state) -> ScheduleResult:
        """生成调度计划"""
        pass
```

---

## 数据流

### 仿真执行流程

```
1. load_config() - 加载配置
2. initialize() - 初始化仿真
3. observe() → decide() → execute() - 仿真循环
4. push_snapshot() - SSE 推送
5. is_done() - 判断结束
```

### SSE 事件

| 事件类型 | 数据格式 | 说明 |
|----------|----------|------|
| state | StateSnapshot | 状态快照 |
| metrics | MetricsSnapshot | 指标快照 |
| control | ControlStatus | 控制状态 |

---

## 扩展点

### 自定义调度器

```python
class MyScheduler(Scheduler):
    def schedule(self, state) -> ScheduleResult:
        # 自定义调度逻辑
        pass
```

### 自定义路由器

```python
class MyRouter(Router):
    def route(self, state) -> RouteResult:
        # 自定义路由逻辑
        pass
```
```

---

## 五、参考示例

查看现有工厂文档了解完整格式：

- `docs/grid_factory/QUICKSTART.md` - Grid 工厂快速开始
- `docs/grid_factory/QUICKSTART.md` 中的架构图和数据流章节

---

## 六、模板维护

如需更新模板，请同步更新：
1. 本 README.md 文件
2. 现有工厂文档（如需要）
3. 主 README.md 中的相关章节
