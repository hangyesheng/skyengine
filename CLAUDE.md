# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SkyEngine (天工)** is a flexible manufacturing system (FMS) simulation-scheduling integration platform. It provides high-fidelity simulation verification for scheduling algorithms, designed for smart manufacturing, digital twin validation, and multi-agent path planning research.

The architecture follows a **two-layer decoupled design**:
- **Service Layer** (`application/`): FastAPI backend + Vue.js frontend, providing REST APIs, SSE streaming, and visualization
- **Algorithm/Executor Layer** (`executor/`): Factory simulation environments, scheduling algorithms, path planning, and event systems

**Key Principle**: Service layer contains no scheduling logic; algorithm layer has no business interface dependencies.

---

## Development Commands

### Backend (Python/FastAPI)
```bash
# Install dependencies (using uv package manager)
uv sync

# Start backend server (with auto-reload)
uv run uvicorn application.backend.server:app --reload --host 0.0.0.0 --port 8000

# Start without reload
uv run uvicorn application.backend.server:app --host 0.0.0.0 --port 8000
```

### Frontend (Vue.js/Vite)
```bash
cd application/frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Format code
npm run format
```

### Testing
```bash
# Check available factory proxies
uv run python -c "from application.backend.core.ProxyFactory import ProxyFactory; import application.backend.core; print(ProxyFactory.available())"

# Test SSE endpoint
curl -N http://localhost:8000/stream/state

# Health check
curl http://localhost:8000/health
```

---

## Architecture Details

### Service Layer Structure

**Backend** (`application/backend/`):
- `server.py` - Main FastAPI application with SSE endpoints and factory lifecycle management
- `core/` - Proxy pattern implementation for factory abstraction
  - `BaseFactoryProxy.py` - Abstract base defining proxy protocol
  - `ProxyFactory.py` - Registry and factory for creating proxy instances
  - `RouteRegistry.py` - Centralized route registration system
  - `StaticFactoryProxy.py` - Demo/testing proxy with predefined scenarios
  - `PacketFactoryProxy.py` - Real production factory proxy
- `packet_factory/` - Backend services for PacketFactory (backend_server, backend_core, service)

**Frontend** (`application/frontend/`):
- `src/stores/factory.js` - Pinia store managing factory configs, animation frames, and playback state
- `src/utils/sse.js` - SSE manager for real-time communication
- `src/utils/factoryConnection.js` - Factory connection manager for SSE
- `src/utils/api.js` - API route definitions and HTTP utilities
- `src/scenarios/` - Test scenarios (fullSystemTest.js, backendSystemTest.js)
- `src/views/` - Main views (FactoryView, HomeView)
- `src/views/factory/` - Factory-specific management views
- `src/components/` - Reusable UI components (FactoryVisualization, ControlPanel, etc.)

### Algorithm/Executor Layer Structure

**executor/packet_factory/`**:
- `registry/` - Component and event registration system
- `lifecycle/` - Factory lifecycle management (bootstrap, context, initializer)
- `packet_factory_env/` - Factory environment implementation
- `event/` - Event system
- `call_back/` - Callback handlers
- `logger/` - Logging utilities

### Configuration & Data

- `config/` - Factory configuration files (JSON format)
  - `example_factory.json` - Static factory example
  - `grid_factory.json` - Grid-based factory configuration
- `dataset/` - Benchmark datasets for testing (JSPLIB, FJSP instances, POGEMA)

---

## Key Design Patterns

### FactoryProxy Pattern

All factory implementations expose a unified interface through the `FactoryProxyProtocol`:

```python
# Lifecycle
async def initialize()
async def cleanup()
async def start()
async def pause()
async def reset()
async def stop()

# Streaming (SSE)
async def get_state_events() -> list
async def get_metrics_events() -> list
async def get_control_events() -> list

# Snapshots
async def get_state_snapshot() -> dict
async def get_metrics_snapshot() -> dict
async def get_control_status() -> dict
```

Proxies are registered via decorator and created through `ProxyFactory.create("factory_id")`.

### SSE Data Flow

1. Frontend connects to `/stream/state` (or `/stream/metrics`, `/stream/control`)
2. Backend yields SSE events with format: `event: {type}\ndata: {json}\n\n`
3. Frontend SSEManager parses events and updates stores
4. Store updates trigger Vue component re-rendering

**State Snapshot Format**:
```javascript
{
  "timestamp": "T+10s",
  "env_timeline": "10",
  "grid_state": {
    "positions_xy": [[5, 2], [3, 4]],    // AGV [x, y] positions
    "is_active": [true, true]              // AGV active states
  },
  "machines": {
    "M1": { "status": "WORKING", "progress": 75 },
    "M2": { "status": "IDLE", "progress": 0 }
  },
  "active_transfers": [
    { "from": "M1", "to": "M3", "agv": "AGV-1", "progress": 50 }
  ]
}
```

### Frontend Store Architecture

**factory.js** store manages:
- `factories` - Available factory types
- `factoryConfigs` - Loaded factory configurations
- `historyBuffer` - Animation frame queue (unified data source)
- `currentIndex` - Current playback position
- `isPlaying` - Playback state

**Key Methods**:
- `pushSnapshot(snapshot)` - Append frame to buffer (for SSE/real-time)
- `loadData(data)` - Load complete history (for offline replay)
- `nextStep()` - Advance animation frame

### Vite Proxy Configuration

Frontend proxies `/api/*` to `http://127.0.0.1:8000`:

```javascript
proxy: {
  "/api": {
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, ""),
  },
}
```

Use `getApiUrl(route)` from `utils/api.js` for consistent URL generation.

---

## Available Factory Types

| ID | Name | Description | Status |
|----|------|-------------|---------|
| `packet_factory` | 翼辉电池装配无人产线 | Production-ready |
| `grid_factory` | 翼辉原料分拣仓 | Requires joint_sim |
| `northeast_center` | 北满钢铁制造中心 | Demo (alias: static_factory) |
| `southwest_logistics` | 西南铝业制造中心 | Coming soon |

---

## Important File Locations

| Purpose | Path |
|---------|------|
| Backend Server | `application/backend/server.py` |
| Base Proxy | `application/backend/core/BaseFactoryProxy.py` |
| Proxy Factory | `application/backend/core/ProxyFactory.py` |
| Route Registry | `application/backend/core/RouteRegistry.py` |
| Frontend Store | `application/frontend/src/stores/factory.js` |
| SSE Manager | `application/frontend/src/utils/sse.js` |
| API Routes | `application/frontend/src/utils/api.js` |
| Factory Configs | `config/*.json` |
| Backend Tests | `application/frontend/src/scenarios/` |

---

## Adding New Factory Proxies

1. Create proxy class inheriting from `BaseFactoryProxy` in `application/backend/core/`
2. Register with `@ProxyFactory.register_proxy("factory_id")` decorator
3. Implement required async methods (initialize, start, pause, reset, etc.)
4. Register custom routes via `@RouteRegistry.register_route(...)` if needed
5. Add factory definition to `stores/factory.js` factories array
6. Create corresponding Vue component in `views/factory/`

See `PacketFactoryProxy.py` for a complete example with custom route registration.

---

## Testing with Static Data

For frontend development without backend, use `StaticFactoryProxy` which provides predefined AGV trajectories, machine states, and metrics. This is accessed via the "northeast_center" factory ID.

The `fullSystemTest.js` scenario provides 55 frames of animation data for testing visualization.
