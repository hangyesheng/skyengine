// 数据层（静态定义）
// 文件里定义了三类独立的数据结构：
// AGV_TRAJECTORIES — 位置序列，每一步记录一个或多个 AGV 的坐标 (x, y) 和激活状态。总共 55 帧，描述了一条完整的巡回路径（出发 → 各停靠点 → 返回原点）。
// MACHINE_STATES — 机器状态序列，按剧本逻辑分四个阶段：预热 → 高负荷 → 故障 → 恢复。每步产出 M1/M2/M3 的 status（IDLE / WORKING / BROKEN）和 load（0-99）。
// METRICS_DATA — 性能指标序列，以 M1 负载为基准，派生出效率、利用率、AGV 耗电、任务堆积等图表数据。
// EVENTS_LOG — 手动精编的关键事件，稀疏分布在特定步骤上（只有 6 条），描述系统里程碑。

// 我现在需要你做的是：
//  1.利用后端数据,完成上述数据的获取,而不再采用静态数据;
//  2.获取完数据后,你可以继续向下做驱动层.


// 驱动层（runBackendSystemTest）
// 核心是一个 setInterval（每 1500ms 一帧），逐步消费上面的数据，向下游推送。推送分三个通道：
// factoryStore.pushSnapshot(snapshot)   ← AGV 位置 + 机器状态 → 工厂地图视图
// monitorStore.pushMetrics(payload)     ← 图表数据 → 性能监控面板
// monitorStore.pushEvent(event)         ← 事件日志 → 告警/日志列表
// 数据流向可以这样理解：
// AGV_TRAJECTORIES  ─┐
// MACHINE_STATES    ─┼─→ setInterval (逐帧) ─→ factoryStore.pushSnapshot
// METRICS_DATA      ─┤                      ─→ monitorStore.pushMetrics
// EVENTS_LOG        ─┘                      ─→ monitorStore.pushEvent

// 我现在需要你做的是：
//  1.利用从后端获取的数据，完成上述数据的推送逻辑，替换掉之前的静态数据驱动;
//  2.确保推送的时序和数据内容与之前静态定义的完全一致，以保持前端展示不变。


// ### 更换数据源时需要保持不变的部分

// 后续替换真实数据源时，**Store 的接口契约**必须保持稳定，你只需要替换"谁来喂数据"，三个 push 方法的入参结构不变：

// | 方法 | 关键字段 |
// |---|---|
// | `pushSnapshot` | `grid_state.positions_xy`, `grid_state.is_active`, `machines` |
// | `pushMetrics` | `machine.data`, `agv.data`, `job.data`, `keyMetrics` |
// | `pushEvent` | `title`, `message`, `type`, `idx` |

// 我说的"更换数据源" — 是打算接入 SSE 形式.
import { sseManager } from '../utils/sse.js';
import { API_ROUTES, getApiUrl } from '../utils/api.js';
import { apiPost } from '../utils/api.js';

/**
 * 创建带缓冲的状态管理器
 * 解决问题：
 * 1. SSE 数据来得快，动画播放慢 - 使用缓冲队列
 * 2. 重复 timestamp - 去重
 * 3. 动画播放控制 - 按固定速度消费缓冲队列
 */
class BufferedStateManager {
    constructor(factoryStore, options = {}) {
        this.factoryStore = factoryStore;
        this.buffer = [];              // 缓冲队列
        this.seenTimestamps = new Set(); // 已处理的 timestamp
        this.isPlaying = false;          // 是否正在播放
        this.playbackInterval = options.playbackInterval || 200; // 播放间隔 (ms)
        this.maxBufferSize = options.maxBufferSize || 500; // 最大缓冲大小
        this.intervalId = null;
        this.lastProcessedIndex = -1;
    }

    /**
     * 添加状态到缓冲队列（带去重）
     */
    addState(data) {
        const timestamp = data.timestamp || data.env_timeline;

        // 去重：如果 timestamp 已存在，跳过
        if (timestamp && this.seenTimestamps.has(timestamp)) {
            console.log(`[BufferedStateManager] 跳过重复 timestamp: ${timestamp}`);
            return false;
        }

        // 记录 timestamp
        if (timestamp) {
            this.seenTimestamps.add(timestamp);
        }

        // 构建快照
        const snapshot = {
            timestamp: timestamp,
            env_timeline: data.env_timeline || timestamp,
            grid_state: data.grid_state || { positions_xy: [], is_active: [] },
            machines: data.machines || {},
            active_transfers: data.active_transfers || []
        };

        // 添加到缓冲队列
        this.buffer.push(snapshot);

        // 限制缓冲大小（FIFO）
        if (this.buffer.length > this.maxBufferSize) {
            const removed = this.buffer.shift();
            console.log(`[BufferedStateManager] 缓冲已满，移除最旧: ${removed.timestamp}`);
        }

        console.log(`[BufferedStateManager] 添加状态: ${timestamp}, 缓冲大小: ${this.buffer.length}`);
        return true;
    }

    /**
     * 开始播放动画
     */
    startPlayback() {
        if (this.isPlaying) return;
        this.isPlaying = true;

        this.intervalId = setInterval(() => {
            this._playNextFrame();
        }, this.playbackInterval);

        console.log(`[BufferedStateManager] 开始播放，间隔: ${this.playbackInterval}ms`);
    }

    /**
     * 停止播放动画
     */
    stopPlayback() {
        this.isPlaying = false;
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        console.log('[BufferedStateManager] 停止播放');
    }

    /**
     * 播放下一帧
     */
    _playNextFrame() {
        if (this.buffer.length === 0) {
            // 缓冲为空，等待数据
            return;
        }

        // 取出下一帧
        const snapshot = this.buffer.shift();
        this.lastProcessedIndex++;

        // 推送到 store
        this.factoryStore.pushSnapshot(snapshot);
    }

    /**
     * 获取缓冲状态
     */
    getStatus() {
        return {
            bufferSize: this.buffer.length,
            isPlaying: this.isPlaying,
            seenTimestamps: this.seenTimestamps.size,
            lastProcessedIndex: this.lastProcessedIndex
        };
    }

    /**
     * 清理
     */
    cleanup() {
        this.stopPlayback();
        this.buffer = [];
        this.seenTimestamps.clear();
        this.lastProcessedIndex = -1;
    }
}

/**
 * 后端系统测试 - 通过 SSE 与后端交互
 * @param {Object} factoryStore - 工厂状态 Store
 * @param {Object} monitorStore - 监控 Store
 * @param {Object} data - 配置数据 { algorithm }
 * @param {Function} onFinish - 完成回调
 * @returns {Function} 清理函数，用于停止测试和断开 SSE 连接
 */
export async function backendSystemTest(factoryStore, monitorStore, data, onFinish) {
    // 保存 SSE 连接 ID
    let connectionId = null;

    // 创建带缓冲的状态管理器
    const stateManager = new BufferedStateManager(factoryStore, {
        playbackInterval: 200,  // 200ms 播放一帧
        maxBufferSize: 500
    });

    // 清理函数 - 断开 SSE 连接
    const cleanup = () => {
        if (connectionId) {
            console.log('[backendSystemTest] 断开 SSE 连接:', connectionId);
            sseManager.disconnect(connectionId);
            connectionId = null;
        }
        stateManager.cleanup();
        factoryStore.isPlaying = false;
    };

    // 后端发送请求,构造数据 - 必须顺序执行，避免竞态条件
    try {
        // 1. 重置环境
        console.log("[backendSystemTest] Step 1: 重置环境...");
        await apiPost(API_ROUTES.FACTORY_CONTROL_RESET, null, { timeout: 15000 });
        console.log("[backendSystemTest] 环境RESET完成");

        // 2. 设定策略
        console.log("[backendSystemTest] Step 2: 设定策略...", data);
        await apiPost(API_ROUTES.FACTORY_ALGORITHM_SET, data, { timeout: 15000 });
        console.log("[backendSystemTest] 策略设定完成");

        // 3. 启动仿真
        console.log("[backendSystemTest] Step 3: 启动仿真...");
        await apiPost(API_ROUTES.FACTORY_CONTROL_PLAY, null, { timeout: 15000 });
        console.log("[backendSystemTest] 仿真启动完成");
    } catch (error) {
        console.error("[backendSystemTest] API调用失败:", error);
        cleanup();
        if (onFinish) onFinish();
        return cleanup;
    }
    // 通过SSE监听数据流，更新store

    // 1. 初始化
    factoryStore.reset();
    monitorStore.clear();
    factoryStore.isPlaying = true; // 开启播放状态，让 Store 知道我们在进行实时更新

    // 启动缓冲播放器
    stateManager.startPlayback();

    // 通过SSE获得agvFrame和machineFrame
    const stateUrl = getApiUrl(API_ROUTES.STREAM_STATE);
    connectionId = sseManager.connect(stateUrl, {
        eventTypes: ['state'],  // 监听的事件类型
        eventHandlers: {
            // 处理快照事件 - 添加到缓冲队列
            state: (data) => {
                console.log('[SSE] Received state:', data.timestamp || data.env_timeline);

                // 检查是否完成
                if (data.status === 'finished' || data.status === 'stopped') {
                    // 运行完成，断开连接
                    console.log('[backendSystemTest] 仿真完成');
                    cleanup();
                    if (onFinish) onFinish();
                    return;
                }

                // 检查是否是有效状态（不是 idle/no_factory/error）
                if (data.status === 'idle' || data.status === 'no_factory' || data.status === 'error') {
                    console.log('[SSE] 跳过无效状态:', data.status);
                    return;
                }

                // 添加到缓冲队列（带去重）
                stateManager.addState(data);

                // 打印缓冲状态
                const status = stateManager.getStatus();
                console.log(`[BufferedStateManager] 缓冲: ${status.bufferSize}, 已处理: ${status.seenTimestamps}`);
            },
        },

        onError: (error) => {
            console.error('SSE error:', error);
            cleanup();
            if (onFinish) onFinish();
        }
    });

    // 返回清理函数，供外部调用
    return cleanup;
}
