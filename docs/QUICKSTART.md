# SkyEngine 快速开始指南

## 📋 目录

- [环境要求](#环境要求)
- [环境安装](#环境安装)
- [Application 使用](#application-使用)
- [算法研究环境使用](#算法研究环境使用)
- [Docker Compose 部署](#docker-compose-部署)
- [常见问题](#常见问题)

---

## 环境要求

### 基础要求

- **Python**: >= 3.11
- **Node.js**: >= 20.19.0 或 >= 22.12.0
- **包管理器**: uv (推荐) 或 pip
- **操作系统**: Windows / Linux / macOS

### 硬件要求

- **内存**: 建议 8GB 以上
- **GPU**: 可选，用于深度学习模型训练（需要 CUDA 12.9 支持）
- **存储**: 至少 5GB 可用空间

---

## 环境安装

### 1. 安装 uv 包管理器

SkyEngine 使用 `uv` 作为包管理器，它比传统的 pip 更快、更可靠。

#### Windows

```powershell
# 使用 PowerShell 安装
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Linux / macOS

```bash
# 使用 curl 安装
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 验证安装

```bash
uv --version
```

### 2. 克隆项目

```bash
git clone https://github.com/nju-dislab/SkyEngine.git
cd SkyEngine
```

### 3. 安装 Python 依赖

使用 uv 同步项目依赖：

```bash
# 同步所有依赖（推荐）
uv sync

# 或者手动安装
uv pip install -e .
```

**注意事项**：

- 项目使用 PyTorch 2.8.0（CUDA 12.9 版本）
- uv 会自动处理依赖冲突和版本约束
- Pydantic 版本被固定在 1.x（<2.0.0），以确保与 pogema 的兼容性

### 4. 安装前端依赖

```bash
cd application/frontend
npm install
cd ../..
```

### 5. 验证安装

```bash
# 验证 Python 环境
uv run python -c "import torch; import pogema; print('Python 环境正常')"

# 验证前端环境
cd application/frontend && npm run build && cd ../..
```

---

## Application 使用

Application 是 SkyEngine 的可视化交互界面，提供实时仿真监控和调度可视化功能。

### 启动后端服务

在项目根目录下运行：

```bash
uv run uvicorn application.backend.server:app --reload --host 0.0.0.0 --port 8000
```

**启动参数说明**：

- `--reload`: 开发模式，代码修改后自动重载
- `--port 8000`: 后端监听端口（默认 8000）

**后端功能**：

- SSE（Server-Sent Events）实时状态推送
- 工厂代理管理（Grid Factory / Packet Factory）
- 性能指标监控
- 配置加载和管理

### 启动前端服务

打开新终端，运行：

```bash
cd application/frontend
npm run dev
```

前端默认运行在 `http://localhost:5173`

### 使用界面

1. **打开浏览器**访问 `http://localhost:5173`
2. **选择工厂类型**：
   - Grid Factory: 栅格化仿真环境
   - Packet Factory: 包裹调度仿真环境
   - Static Factory: 用于测试
3. **加载配置文件**：上传对应的配置文件（YAML 格式）
4. **启动仿真**：点击"开始"按钮
5. **实时监控**：
   - 状态可视化（地图、AGV、工单）
   - 性能指标（吞吐量、完成时间、资源利用率）
   - 热力图分析

注意，当你初次进入，可能会发现Grid Factory与Packet Factory等环境均无法正常显示，不要着急！请参考如下文档：

> [GridFactory](https://github.com/dayu-autostreamer/skyengine/blob/main/docs/grid_factory)
> [PacketFactory](https://github.com/dayu-autostreamer/skyengine/blob/main/docs/packet_factory)

### API 端点

- `GET /stream/state`: 工厂状态流（SSE）
- `GET /stream/metrics`: 性能指标流（SSE）
- `POST /load_config`: 加载配置
- `POST /control/start`: 启动仿真
- `POST /control/step`: 单步执行
- `POST /control/pause`: 暂停仿真
- `POST /control/resume`: 恢复仿真
- `POST /control/stop`: 停止仿真
- `POST /control/reset`: 重置环境

### 基准数据集

项目内置多个标准数据集：

- **FJSP 数据集**: `dataset/fjsp-instances/`
  - Barnes, Behnke, Brandimarte, Dauzere, Fattahi, Hurink, Kacem
- **JSP 数据集**: `dataset/JSPLIB-master/instances/`
- **AGV 数据集**: `dataset/agv-instances/`
- **地图数据集**: `dataset/map_dataset/pogema-benchmark-main/`
- **StarJob 数据集**: `dataset/mideavalwisard_Starjob/starjob_130k.json`

## Docker Compose 部署

使用 Docker Compose 快速部署完整的生产环境。

### 前置要求

- **Docker**: >= 20.10
- **Docker Compose**: >= 2.0

### 部署步骤

#### 1. 构建并启动服务

在项目根目录下：

```bash
cd dockerfiles
docker-compose -f application.yaml up -d
```

**参数说明**：

- `-d`: 后台运行
- `-f application.yaml`: 指定 compose 文件

#### 2. 查看服务状态

```bash
docker-compose -f application.yaml ps
```

#### 3. 查看日志

```bash
# 查看所有服务日志
docker-compose -f application.yaml logs -f

# 查看后端日志
docker-compose -f application.yaml logs -f backend

# 查看前端日志
docker-compose -f application.yaml logs -f frontend
```

#### 4. 停止服务

```bash
docker-compose -f application.yaml down
```

### 服务端口

- **前端**: `http://localhost:80`
- **后端**: `http://localhost:8000`

### Docker Compose 配置详解

```yaml
services:
  backend:
    build:
      context: ..                          # 构建上下文为项目根目录
      dockerfile: dockerfiles/backend.dockerfile
    container_name: skyengine-backend
    ports:
      - "8000:8000"                        # 后端端口映射
    environment:
      - PYTHONUNBUFFERED=1                 # Python 输出不缓冲
    networks:
      - skyengine-network
    restart: unless-stopped                # 自动重启策略

  frontend:
    build:
      context: ..
      dockerfile: dockerfiles/frontend.dockerfile
    container_name: skyengine-frontend
    ports:
      - "80:80"                            # 前端端口映射
    depends_on:
      - backend                            # 依赖后端服务
    networks:
      - skyengine-network
    restart: unless-stopped
```

### 自定义配置

修改端口映射（例如前端改为 8080）：

```yaml
# dockerfiles/application.yaml
services:
  frontend:
    ports:
      - "8080:80"  # 宿主机:容器
```

### 数据持久化

添加卷挂载以保存仿真结果：

```yaml
services:
  backend:
    volumes:
      - ./logs:/app/logs              # 日志持久化
      - ./results:/app/results            # 结果持久化
```

### 生产环境优化

#### 资源限制

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

#### 健康检查

```yaml
services:
  backend:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

---

## 常见问题

### Q1: uv 安装依赖时报错 "Could not find a version that satisfies..."

**解决方案**：

```bash
# 清除缓存并重新安装
uv cache clean
uv sync --refresh
```

### Q2: PyTorch CUDA 版本不匹配

**解决方案**：

```bash
# 检查 CUDA 版本
nvidia-smi

# 如果需要 CPU 版本的 PyTorch，修改 pyproject.toml 中的索引配置
```

### Q3: 前端启动后无法连接后端

**检查清单**：

- 后端是否在 8000 端口运行
- CORS 配置是否正确
- 防火墙是否阻止连接

```bash
# 测试后端连接
curl http://localhost:8000/stream/state
```

### Q4: Docker 构建失败

**解决方案**：

```bash
# 清理 Docker 缓存
docker system prune -a

# 使用 --no-cache 重新构建
docker-compose -f dockerfiles/application.yaml build --no-cache
```

### Q5: 仿真运行缓慢

**优化建议**：

- 减少地图大小或障碍物数量
- 降低 AGV 数量
- 使用更简单的调度算法
- 关闭不必要的日志输出

### Q6: 如何切换不同的调度算法？

在配置文件中修改：

```yaml
scheduler:
  type: "priority"  # 可选: greedy, priority, best, neural
  priority_rule: "FIFO"  # 优先级规则
```

### Q7: 日志文件过大

**解决方案**：

```bash
# 清理旧日志
uv run python -c "from sky_logs.logger import clean_old_logs; clean_old_logs(days=7)"
```

---

## 下一步

### 深入学习

- 📖 阅读 [README.md](README.md) 了解系统架构
- 📖 查看 [docs/](docs/) 目录获取详细文档
- 🔧 研究 [test/](test/) 目录的示例代码
- 📊 分析基准数据集 [dataset/](dataset/)

### 开发指南

- 🏗️ [组件设计文档](https://github.com/dayu-autostreamer/skyengine/blob/17dbd7f8d22523ebc7a304f2886a359b6d8205ea/docs/quick_start/client)
- 🔄 [事件系统设计](https://github.com/dayu-autostreamer/skyengine/blob/17dbd7f8d22523ebc7a304f2886a359b6d8205ea/docs/quick_start/developer)
- 📝 [调度算法实现](docs/implement/)

### 社区与支持

- 🐛 提交 Issue: [GitHub Issues](https://github.com/nju-dislab/SkyEngine/issues)
- 💬 讨论区: [GitHub Discussions](https://github.com/nju-dislab/SkyEngine/discussions)
- 📧 联系我们: [hitskyrim@qq.com]


---

**祝您使用愉快！🚀**
