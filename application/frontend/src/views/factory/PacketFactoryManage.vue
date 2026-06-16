<template>
  <div class="factory-manage-container">
    <div class="middle-panel">
      <FactoryPlayerSSE
        :hide-control-panel="true"
        :edit-mode="isEditMode"
        :background-theme="backgroundTheme"
        :background-size="backgroundSize"
      />
    </div>

    <FactoryTabsPanel
      :tabs="tabs"
      v-model="activeTab"
      v-model:show-panel="showPanel"
      :is-edit-mode="isEditMode"
      :is-running-test="isRunningTest"
      factory-type="packet_factory"
      @edit-mode-change="isEditMode = $event"
    >
      <!-- ==================== 仿真 Tab ==================== -->
      <template #tab-simulation>
        <div class="simulation-tab">
          <div class="sim-field">
            <label>工厂配置</label>
            <select v-model="selectedFactory" class="plan-select" :disabled="isRunningTest">
              <option value="" disabled>选择工厂配置...</option>
              <option v-for="factory in factoryList" :key="factory.id" :value="factory.id">
                {{ factory.id }}
              </option>
            </select>
          </div>

          <div class="sim-field">
            <label>Agent 策略</label>
            <select v-model="selectedAgent" class="plan-select" :disabled="isRunningTest">
              <option value="">默认 (无 Agent)</option>
              <option v-for="agent in agentList" :key="agent.id" :value="agent.id">
                {{ agent.id.replace('.yaml', '') }}
              </option>
            </select>
          </div>

          <div class="sim-btn-row">
            <button @click="handleFactoryRender" class="sim-action-btn render" :disabled="!selectedFactory || isRunningTest">
              🗺️ 渲染
            </button>
            <button @click="handleFactoryStart" class="sim-action-btn start" :disabled="isRunningTest">
              ▶ 启动
            </button>
            <button @click="handleFactoryPause" class="sim-action-btn pause" :disabled="isRunningTest">
              ⏸ 暂停
            </button>
            <button @click="handleFactoryReset" class="sim-action-btn reset" :disabled="isRunningTest">
              ⏹ 重置
            </button>
          </div>

          <div class="sim-field">
            <label>仿真速度: {{ speedLevel }}</label>
            <input type="range" v-model.number="speedLevel" min="1" max="60" class="sim-slider"
              @change="changeSpeed" :disabled="isRunningTest" />
          </div>

          <!-- 工厂存活状态 -->
          <div v-if="aliveStatus" class="sim-field">
            <div class="alive-badge" :class="{ alive: aliveStatus.is_alive }">
              {{ aliveStatus.is_alive ? '🟢 运行中' : '🔴 已停止' }}
              <span v-if="aliveStatus.training_completed" class="training-done">训练完成 | Makespan: {{ aliveStatus.makespan }}s</span>
            </div>
          </div>

          <div class="sim-divider"></div>

          <div class="sim-field">
            <label>场景风格</label>
            <select v-model="backgroundTheme" class="plan-select">
              <option value="clean">简洁</option>
              <option value="factory">工厂车间</option>
            </select>
          </div>
          <div v-if="backgroundTheme === 'factory'" class="sim-field">
            <label>厂房尺寸</label>
            <select v-model.number="backgroundSize" class="plan-select">
              <option :value="1">紧凑</option>
              <option :value="2">标准</option>
              <option :value="3">宽敞</option>
              <option :value="4">大厅</option>
            </select>
          </div>
        </div>
      </template>

      <!-- ==================== 控制 Tab ==================== -->
      <template #tab-control>
        <div class="control-tab">
          <!-- AGV 控制 -->
          <div class="ctrl-section">
            <div class="ctrl-title">🚛 AGV 控制</div>
            <div class="ctrl-row">
              <select v-model="selectedAgv" class="plan-select ctrl-select">
                <option :value="null" disabled>选择 AGV...</option>
                <option v-for="agv in agvList" :key="agv.id" :value="agv.id">AGV {{ agv.id }}</option>
              </select>
              <button @click="pauseAgv" class="ctrl-btn warn" :disabled="selectedAgv === null">⏸ 暂停</button>
              <button @click="resumeAgv" class="ctrl-btn ok" :disabled="selectedAgv === null">▶ 恢复</button>
            </div>
          </div>

          <!-- Machine 控制 -->
          <div class="ctrl-section">
            <div class="ctrl-title">⚙️ 机器控制</div>
            <div class="ctrl-row">
              <select v-model="selectedMachine" class="plan-select ctrl-select">
                <option :value="null" disabled>选择机器...</option>
                <option v-for="m in machineList" :key="m.id" :value="m.id">Machine {{ m.id }}</option>
              </select>
              <button @click="pauseMachine" class="ctrl-btn warn" :disabled="selectedMachine === null">⏸ 暂停</button>
              <button @click="resumeMachine" class="ctrl-btn ok" :disabled="selectedMachine === null">▶ 恢复</button>
            </div>
          </div>

          <!-- Job 控制 -->
          <div class="ctrl-section">
            <div class="ctrl-title">📦 任务控制</div>
            <div class="ctrl-row">
              <select v-model="selectedJob" class="plan-select ctrl-select">
                <option :value="null" disabled>选择任务...</option>
                <option v-for="job in jobList" :key="job.id" :value="job.id">Job {{ job.id }}</option>
              </select>
              <button @click="addJob" class="ctrl-btn primary" :disabled="selectedJob === null">➕ 添加</button>
            </div>
          </div>

          <!-- Job 进度 -->
          <div class="ctrl-section" v-if="jobProgressList.length > 0">
            <div class="ctrl-title">📋 任务进度</div>
            <div v-for="job in jobProgressList" :key="job.id" class="job-progress-item">
              <div class="job-progress-header">
                <span>Job {{ job.id }}</span>
                <span :class="job.status === 'FINISHED' ? 'status-done' : 'status-running'">
                  {{ job.status === 'FINISHED' ? '✓ 完成' : '● 进行中' }}
                </span>
              </div>
              <div class="job-progress-bar">
                <div class="job-progress-fill" :style="{ width: job.progress + '%' }"
                  :class="job.status === 'FINISHED' ? 'fill-done' : 'fill-running'"></div>
              </div>
              <span class="job-progress-pct">{{ job.progress.toFixed(1) }}%</span>
            </div>
          </div>
        </div>
      </template>

      <!-- ==================== 配置 Tab ==================== -->
      <template #tab-config>
        <div class="config-tab">
          <div class="config-section">
            <div class="ctrl-title">📄 配置上传</div>
            <div class="config-upload-row">
              <input v-model="configName" class="plan-select config-input" placeholder="配置名称" />
              <button @click="getStandardConfig" class="ctrl-btn ok">📥 标准配置</button>
              <button @click="uploadConfigSet" class="ctrl-btn primary" :disabled="!configName">📤 上传</button>
            </div>
            <div class="config-drop-zone" @dragover.prevent @drop.prevent="handleDrop">
              <input type="file" ref="fileInput" accept=".yaml,.yml" @change="handleFileSelect" hidden />
              <div class="drop-zone-content" @click="$refs.fileInput.click()">
                <span>📂 点击或拖拽 YAML 文件到此处</span>
              </div>
            </div>
          </div>

          <div class="config-section">
            <div class="ctrl-title">📝 日志下载</div>
            <div class="config-upload-row">
              <button @click="downloadLog('backend')" class="ctrl-btn ok">📋 后端日志</button>
              <button @click="downloadLog('system')" class="ctrl-btn primary">📋 系统日志</button>
            </div>
          </div>
        </div>
      </template>

      <!-- ==================== 甘特图 Tab ==================== -->
      <template #tab-gantt>
        <div class="gantt-tab">
          <div class="gantt-mode-bar">
            <button v-for="mode in ganttModes" :key="mode.value"
              class="gantt-mode-btn" :class="{ active: ganttViewMode === mode.value }"
              @click="ganttViewMode = mode.value">
              {{ mode.icon }} {{ mode.label }}
            </button>
          </div>
          <div class="gantt-chart-container">
            <GanttChart v-if="ganttViewMode === 'agv'" type="agv" :auto-refresh="true" :refresh-interval="3000" />
            <GanttChart v-else-if="ganttViewMode === 'machine'" type="machine" :auto-refresh="true" :refresh-interval="3000" />
            <div v-else-if="ganttViewMode === 'both'" class="gantt-dual-view">
              <div class="gantt-half"><GanttChart type="agv" :auto-refresh="true" :refresh-interval="3000" /></div>
              <div class="gantt-half"><GanttChart type="machine" :auto-refresh="true" :refresh-interval="3000" /></div>
            </div>
          </div>
        </div>
      </template>

      <!-- ==================== 指标 / 日志 使用默认面板 ==================== -->
    </FactoryTabsPanel>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from "vue";
import { ElMessage } from "element-plus";
import { useFactoryStore } from "@/stores/factory";
import { useMonitorStore } from "@/stores/monitor";
import { apiGet, apiPost } from "@/utils/api";
import FactoryPlayerSSE from "@/components/FactoryPlayerSSE.vue";
import FactoryTabsPanel from "@/components/FactoryTabsPanel.vue";
import GanttChart from "@/components/GanttChart.vue";

const store = useFactoryStore();
const monitorStore = useMonitorStore();

// ==================== UI 状态 ====================
const isEditMode = ref(false);
const isRunningTest = ref(false);
const showPanel = ref(true);
const activeTab = ref("simulation");
const backgroundTheme = ref("factory");
const backgroundSize = ref(2);

const tabs = [
  { key: "simulation", label: "仿真", icon: "🚀" },
  { key: "control",    label: "控制", icon: "⚙️" },
  { key: "config",     label: "配置", icon: "🔧" },
  { key: "gantt",      label: "甘特图", icon: "📊" },
  { key: "metrics",    label: "指标", icon: "📈" },
  { key: "events",     label: "日志", icon: "📋" },
];

// ==================== 仿真 Tab 状态 ====================
const selectedFactory = ref("");
const selectedAgent = ref("");
const factoryList = ref([]);
const agentList = ref([]);
const speedLevel = ref(parseInt(sessionStorage.getItem('speedLevel')) || 3);
const aliveStatus = ref(null);

// ==================== 控制 Tab 状态 ====================
const selectedAgv = ref(null);
const selectedMachine = ref(null);
const selectedJob = ref(null);
const agvList = ref([]);
const machineList = ref([]);
const jobList = ref([]);
const jobProgressList = ref([]);

// ==================== 配置 Tab 状态 ====================
const configName = ref("");
const selectedFile = ref(null);

// ==================== 甘特图 Tab 状态 ====================
const ganttViewMode = ref("agv");
const ganttModes = [
  { value: "agv",     label: "AGV",     icon: "🚛" },
  { value: "machine", label: "机器",    icon: "⚙️" },
  { value: "both",    label: "双视图", icon: "📊" },
];

// ==================== 定时器管理 ====================
let aliveCheckTimer = null;
let entityRefreshTimer = null;
let progressRefreshTimer = null;
let statePollTimer = null;
let topologyPollTimer = null;

// ==================== 图拓扑 → 3D 渲染器配置转换 ====================

/**
 * 将 PacketFactory 图拓扑数据转换为 FactoryPlayerSSE/3D 渲染器期望的拓扑格式。
 *
 * 后端 GET /map/topology 返回:
 *   { points: [{id, x, y}], links: [{id, source, target, weight}],
 *     machines: [{id, x, y, point_id, status}], agvs: [{id, x, y, ...}],
 *     gridWidth, gridHeight, timeline }
 *
 * 3D 渲染器 (store.setCurrentTopologyConfig) 期望:
 *   { zones, machines: { "M0": {x,y,...} }, waypoints: { "P0": {x,y,type,...} },
 *     gridWidth, gridHeight }
 */
function convertTopologyFor3D(topology) {
  const { points, links, machines, agvs, gridWidth, gridHeight } = topology;

  // 1. 机器 → machines 字典 (3D 渲染器要求 location: [x, y] 和 id)
  const machinesDict = {};
  for (const m of machines) {
    machinesDict[`M${m.id}`] = {
      id: `M${m.id}`,
      name: `Machine ${m.id}`,
      location: [m.x, m.y],
      size: [1, 1],
      status: 'IDLE',
    };
  }

  // 2. 图节点 → waypoints (3D 渲染器要求 location: [x, y] 和 id)
  const machinePointIds = new Set(machines.map(m => m.point_id));
  const waypointsDict = {};
  for (const p of points) {
    const type = machinePointIds.has(p.id) ? 'dock' : 'route';
    waypointsDict[`P${p.id}`] = {
      id: `P${p.id}`,
      name: `Point ${p.id}`,
      location: [p.x, p.y],
      type,
    };
  }

  // 3. 边 → edges 数组 (3D 渲染器接受 [fromId, toId] 对)
  const edges = links.map(l => [`P${l.source}`, `P${l.target}`]);

  // 4. AGV 初始位置
  const agvDefs = agvs.map(a => ({
    id: a.id,
    name: `AGV-${a.id}`,
    initialLocation: [a.x, a.y],
    velocity: a.velocity || 1.0,
    capacity: a.capacity || 100,
    status: 'IDLE',
  }));

  return {
    id: `packet_factory_${Date.now()}`,
    name: selectedFactory.value || 'PacketFactory',
    zones: [],
    machines: machinesDict,
    waypoints: waypointsDict,
    edges,
    gridWidth,
    gridHeight,
    agvs: agvDefs,
    baseGridSize: 40,
  };
}

// ==================== API 调用：仿真控制 ====================

async function loadFactoryList() {
  try {
    const data = await apiGet("/factory/list");
    factoryList.value = data.factory_list || [];
  } catch (e) {
    console.error("[PacketFactory] 加载工厂列表失败:", e);
  }
}

async function loadAgentList() {
  try {
    const data = await apiGet("/agent/list");
    agentList.value = data.agent_list || [];
  } catch (e) {
    console.error("[PacketFactory] 加载 Agent 列表失败:", e);
  }
}

async function handleFactoryRender() {
  if (!selectedFactory.value) {
    ElMessage.warning("请先选择工厂配置");
    return;
  }
  try {
    isRunningTest.value = true;
    // 先停止可能残留的旧轮询
    stopTopologyPoll();
    stopStatePoll();

    const body = { target_factory: selectedFactory.value };
    if (selectedAgent.value) body.agent_name = selectedAgent.value;

    await apiPost("/map/render", body, { timeout: 30000 });
    ElMessage.success("渲染请求已发送");

    // 渲染请求发送后, 后端在后台线程启动环境, 需要等待环境就绪
    startAliveCheck();
    startEntityRefresh();

    // 轮询获取拓扑数据并加载到 3D 渲染器
    pollTopologyUntilReady();
  } catch (e) {
    ElMessage.error("渲染失败: " + e.message);
    isRunningTest.value = false;
  }
}

/**
 * 尝试从后端加载拓扑数据并注入 3D 渲染器 Store。
 * 成功时返回 true，拓扑为空或请求失败时返回 false。
 */
async function loadTopologyIfReady() {
  try {
    const topology = await apiGet("/map/topology");
    if (topology.points && topology.points.length > 0) {
      // 转换为 3D 渲染器格式并加载到 Store
      const config3D = convertTopologyFor3D(topology);
      store.setCurrentTopologyConfig(config3D);
      store.initializeAGVs();

      // 加载初始状态快照
      const snapshot = await apiGet("/map/state");
      if (snapshot.grid_state?.positions_xy?.length) {
        store.pushSnapshot(snapshot);
      }

      // 启动状态轮询, 实时更新 AGV 位置和机器状态
      startStatePoll();
      return true;
    }
  } catch (e) {
    // 环境尚未就绪, 忽略
  }
  return false;
}

/**
 * 轮询后端拓扑接口, 直到环境就绪 (返回非空数据), 然后加载到 Store。
 * 使用递归 setTimeout 代替 setInterval，避免上一个请求未返回时发起重叠请求。
 */
function pollTopologyUntilReady() {
  let attempts = 0;
  const maxAttempts = 90; // 最多等 45 秒 (90 * 500ms)

  async function tick() {
    attempts++;
    const loaded = await loadTopologyIfReady();
    if (loaded) {
      ElMessage.success("3D 工厂视图已加载");
      isRunningTest.value = false;
      return;
    }
    if (attempts >= maxAttempts) {
      ElMessage.warning("等待环境就绪超时，请稍后点击「启动」重试");
      isRunningTest.value = false;
      return;
    }
    topologyPollTimer = setTimeout(tick, 500);
  }

  tick();
}

async function handleFactoryStart() {
  try {
    // 如果 3D 视图尚未加载（渲染轮询可能超时），尝试加载拓扑
    if (!store.currentConfigId) {
      const loaded = await loadTopologyIfReady();
      if (loaded) {
        ElMessage.success("3D 工厂视图已加载");
      }
    }

    await apiPost("/factory/start");
    ElMessage.success("工厂已启动");

    // 确保状态轮询在运行，以便 3D 视图实时更新
    if (store.currentConfigId && !statePollTimer) {
      startStatePoll();
    }
  } catch (e) {
    ElMessage.error("启动失败: " + e.message);
  }
}

async function handleFactoryPause() {
  try {
    await apiPost("/factory/pause");
    ElMessage.success("工厂已暂停");
  } catch (e) {
    ElMessage.error("暂停失败: " + e.message);
  }
}

async function handleFactoryReset() {
  try {
    await apiPost("/factory/reset");
    ElMessage.success("工厂已重置");
    aliveStatus.value = null;
    stopStatePoll();
    stopTopologyPoll();
  } catch (e) {
    ElMessage.error("重置失败: " + e.message);
  }
}

async function changeSpeed() {
  try {
    await apiPost("/factory/speed", { speedLevel: speedLevel.value });
    sessionStorage.setItem('speedLevel', String(speedLevel.value));
    ElMessage.success(`速度已调整为 ${speedLevel.value}`);
  } catch (e) {
    ElMessage.error("速度调整失败");
  }
}

// ==================== 状态轮询：实时更新 3D 视图 ====================

function startStatePoll() {
  stopStatePoll();
  statePollTimer = setInterval(pollStateSnapshot, 500);
}

function stopStatePoll() {
  if (statePollTimer) { clearInterval(statePollTimer); statePollTimer = null; }
}

function stopTopologyPoll() {
  if (topologyPollTimer) { clearTimeout(topologyPollTimer); topologyPollTimer = null; }
}

async function pollStateSnapshot() {
  try {
    const snapshot = await apiGet("/map/state");
    if (snapshot.status === 'finished') {
      stopStatePoll();
      store.isPlaying = false;
    }
    store.pushSnapshot(snapshot);
  } catch { /* ignore */ }
}

// ==================== API 调用：实体控制 ====================

async function pauseAgv() {
  if (selectedAgv.value === null) return;
  try {
    await apiPost(`/agv/pause/${selectedAgv.value}`);
    ElMessage.success(`AGV ${selectedAgv.value} 已暂停`);
  } catch (e) {
    ElMessage.error("暂停 AGV 失败");
  }
}

async function resumeAgv() {
  if (selectedAgv.value === null) return;
  try {
    await apiPost(`/agv/resume/${selectedAgv.value}`);
    ElMessage.success(`AGV ${selectedAgv.value} 已恢复`);
  } catch (e) {
    ElMessage.error("恢复 AGV 失败");
  }
}

async function pauseMachine() {
  if (selectedMachine.value === null) return;
  try {
    await apiPost(`/machine/pause/${selectedMachine.value}`);
    ElMessage.success(`Machine ${selectedMachine.value} 已暂停`);
  } catch (e) {
    ElMessage.error("暂停机器失败");
  }
}

async function resumeMachine() {
  if (selectedMachine.value === null) return;
  try {
    await apiPost(`/machine/resume/${selectedMachine.value}`);
    ElMessage.success(`Machine ${selectedMachine.value} 已恢复`);
  } catch (e) {
    ElMessage.error("恢复机器失败");
  }
}

async function addJob() {
  if (selectedJob.value === null) return;
  try {
    await apiPost(`/job/add/${selectedJob.value}`);
    ElMessage.success(`任务 ${selectedJob.value} 已添加`);
  } catch (e) {
    ElMessage.error("添加任务失败");
  }
}

// ==================== API 调用：配置/日志 ====================

async function getStandardConfig() {
  try {
    const response = await fetch("/api/standard/get?t=" + Date.now(), { method: "GET" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    const disposition = response.headers.get('Content-Disposition');
    let filename = 'pipeline_config.yaml';
    if (disposition && disposition.includes('filename=')) {
      filename = disposition.split('filename=')[1].replace(/"/g, '');
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    ElMessage.error("下载失败");
  }
}

function handleFileSelect(e) {
  selectedFile.value = e.target.files[0] || null;
}

function handleDrop(e) {
  const files = e.dataTransfer.files;
  if (files.length > 0) selectedFile.value = files[0];
}

async function uploadConfigSet() {
  if (!configName.value) {
    ElMessage.warning("请输入配置名称");
    return;
  }
  if (!selectedFile.value) {
    ElMessage.warning("请选择 YAML 文件");
    return;
  }
  try {
    const formData = new FormData();
    formData.append('file', selectedFile.value);
    const response = await fetch(
      `/api/yaml/upload?config_name=${encodeURIComponent(configName.value)}&t=${Date.now()}`,
      { method: "POST", body: formData }
    );
    if (!response.ok) throw new Error("Upload failed");
    ElMessage.success("上传成功");
    selectedFile.value = null;
    configName.value = "";
    loadFactoryList();
  } catch (e) {
    ElMessage.error("上传失败");
  }
}

async function downloadLog(fileType) {
  try {
    const response = await fetch("/api/log/download?t=" + Date.now(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_type: fileType }),
    });
    if (!response.ok) throw new Error("Download failed");
    const disposition = response.headers.get('Content-Disposition');
    let filename = 'log.txt';
    if (disposition && disposition.includes('filename=')) {
      filename = disposition.split('filename=')[1].replace(/"/g, '');
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    ElMessage.error("下载失败");
  }
}

// ==================== 定时刷新逻辑 ====================

function startAliveCheck() {
  stopAliveCheck();
  aliveCheckTimer = setInterval(async () => {
    try {
      const data = await apiGet("/factory/alive");
      aliveStatus.value = data;
      if (data.is_alive) {
        // 工厂存活后降低检查频率
        clearInterval(aliveCheckTimer);
        aliveCheckTimer = setInterval(async () => {
          try {
            const d = await apiGet("/factory/alive");
            aliveStatus.value = d;
          } catch { /* ignore */ }
        }, 3000);
      }
    } catch { /* ignore */ }
  }, 500);
}

function stopAliveCheck() {
  if (aliveCheckTimer) { clearInterval(aliveCheckTimer); aliveCheckTimer = null; }
}

function startEntityRefresh() {
  stopEntityRefresh();
  entityRefreshTimer = setInterval(refreshEntities, 2000);
}

function stopEntityRefresh() {
  if (entityRefreshTimer) { clearInterval(entityRefreshTimer); entityRefreshTimer = null; }
}

async function refreshEntities() {
  try {
    const [agvs, machines, jobs] = await Promise.all([
      apiGet("/agvs").catch(() => ({ agvs: [] })),
      apiGet("/machines").catch(() => ({ machines: [] })),
      apiGet("/jobs").catch(() => ({ jobs: [] })),
    ]);
    if (agvs.agvs?.length) agvList.value = agvs.agvs;
    if (machines.machines?.length) machineList.value = machines.machines;
    if (jobs.jobs?.length) jobList.value = jobs.jobs;
  } catch { /* ignore */ }
}

function startProgressRefresh() {
  stopProgressRefresh();
  progressRefreshTimer = setInterval(fetchJobProgress, 2000);
}

function stopProgressRefresh() {
  if (progressRefreshTimer) { clearInterval(progressRefreshTimer); progressRefreshTimer = null; }
}

async function fetchJobProgress() {
  try {
    const data = await apiGet("/jobs/progress");
    jobProgressList.value = data.jobs || [];
  } catch { /* ignore */ }
}

// ==================== 生命周期 ====================

onMounted(() => {
  store.reset();
  loadFactoryList();
  loadAgentList();
  startProgressRefresh();
});

onUnmounted(() => {
  stopAliveCheck();
  stopEntityRefresh();
  stopProgressRefresh();
  stopStatePoll();
  stopTopologyPoll();
});
</script>

<style scoped>
@import "../styles/FactoryManage.css";

/* ====== 仿真 Tab 专属样式 ====== */

.sim-action-btn {
  flex: 1;
  padding: 8px 0;
  border: 1px solid rgba(100, 180, 255, 0.2);
  background: rgba(20, 25, 45, 0.6);
  border-radius: 6px;
  cursor: pointer;
  color: rgba(200, 220, 255, 0.9);
  font-size: 12px;
  font-weight: 500;
  transition: all 0.2s;
}

.sim-action-btn:hover:not(:disabled) {
  background: rgba(40, 50, 80, 0.8);
  border-color: rgba(100, 180, 255, 0.4);
}

.sim-action-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.sim-action-btn.render {
  border-color: rgba(100, 200, 255, 0.3);
  background: rgba(30, 80, 120, 0.5);
}

.sim-action-btn.start {
  border-color: rgba(0, 232, 136, 0.3);
  background: rgba(20, 80, 50, 0.5);
}

.sim-action-btn.pause {
  border-color: rgba(234, 179, 8, 0.3);
  background: rgba(80, 65, 20, 0.5);
}

.sim-action-btn.reset {
  border-color: rgba(234, 102, 102, 0.3);
  background: rgba(80, 30, 30, 0.5);
}

.sim-slider {
  width: 100%;
  height: 4px;
  -webkit-appearance: none;
  appearance: none;
  background: rgba(100, 180, 255, 0.15);
  border-radius: 2px;
  outline: none;
}

.sim-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: rgba(100, 180, 255, 0.8);
  cursor: pointer;
}

.alive-badge {
  font-size: 11px;
  padding: 4px 8px;
  border-radius: 4px;
  background: rgba(80, 30, 30, 0.5);
  border: 1px solid rgba(234, 102, 102, 0.2);
  color: rgba(200, 220, 255, 0.8);
}

.alive-badge.alive {
  background: rgba(20, 80, 50, 0.5);
  border-color: rgba(0, 232, 136, 0.3);
}

.training-done {
  margin-left: 8px;
  color: rgba(0, 232, 136, 0.8);
  font-size: 10px;
}

.sim-divider {
  height: 1px;
  background: rgba(255, 255, 255, 0.08);
  margin: 8px 0;
}

/* ====== 控制 Tab 专属样式 ====== */

.control-tab {
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.ctrl-section {
  background: rgba(15, 20, 35, 0.5);
  border: 1px solid rgba(100, 180, 255, 0.08);
  border-radius: 8px;
  padding: 10px;
}

.ctrl-title {
  font-size: 12px;
  font-weight: 600;
  color: rgba(200, 220, 255, 0.9);
  margin-bottom: 8px;
}

.ctrl-row {
  display: flex;
  gap: 6px;
  align-items: center;
}

.ctrl-select {
  flex: 1;
  min-width: 0;
}

.ctrl-btn {
  padding: 5px 10px;
  border: 1px solid rgba(100, 180, 255, 0.15);
  background: rgba(20, 25, 45, 0.6);
  border-radius: 4px;
  cursor: pointer;
  color: rgba(200, 220, 255, 0.85);
  font-size: 11px;
  transition: all 0.15s;
  white-space: nowrap;
}

.ctrl-btn:hover:not(:disabled) {
  background: rgba(40, 50, 80, 0.8);
  border-color: rgba(100, 180, 255, 0.35);
}

.ctrl-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.ctrl-btn.warn {
  border-color: rgba(234, 179, 8, 0.25);
  background: rgba(80, 65, 20, 0.4);
}

.ctrl-btn.ok {
  border-color: rgba(0, 232, 136, 0.25);
  background: rgba(20, 80, 50, 0.4);
}

.ctrl-btn.primary {
  border-color: rgba(100, 180, 255, 0.25);
  background: rgba(30, 60, 100, 0.4);
}

.job-progress-item {
  margin-bottom: 8px;
}

.job-progress-header {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  margin-bottom: 3px;
  color: rgba(200, 220, 255, 0.8);
}

.status-done { color: rgba(0, 232, 136, 0.8); }
.status-running { color: rgba(100, 180, 255, 0.8); }

.job-progress-bar {
  height: 4px;
  background: rgba(100, 180, 255, 0.1);
  border-radius: 2px;
  overflow: hidden;
}

.job-progress-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.3s;
}

.fill-done { background: rgba(0, 232, 136, 0.6); }
.fill-running { background: rgba(100, 180, 255, 0.6); }

.job-progress-pct {
  font-size: 10px;
  color: rgba(160, 190, 230, 0.5);
}

/* ====== 配置 Tab 专属样式 ====== */

.config-tab {
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.config-section {
  background: rgba(15, 20, 35, 0.5);
  border: 1px solid rgba(100, 180, 255, 0.08);
  border-radius: 8px;
  padding: 10px;
}

.config-upload-row {
  display: flex;
  gap: 6px;
  align-items: center;
  margin-bottom: 8px;
}

.config-input {
  flex: 1;
  min-width: 0;
}

.config-drop-zone {
  border: 1px dashed rgba(100, 180, 255, 0.2);
  border-radius: 6px;
  padding: 16px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
}

.config-drop-zone:hover {
  border-color: rgba(100, 180, 255, 0.4);
  background: rgba(30, 50, 80, 0.3);
}

.drop-zone-content {
  font-size: 12px;
  color: rgba(160, 190, 230, 0.5);
}

/* ====== 甘特图 Tab 专属样式 ====== */

.gantt-tab {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.gantt-mode-bar {
  display: flex;
  gap: 4px;
  padding: 8px 10px;
  border-bottom: 1px solid rgba(100, 180, 255, 0.08);
  flex-shrink: 0;
}

.gantt-mode-btn {
  flex: 1;
  padding: 5px 0;
  border: 1px solid rgba(100, 180, 255, 0.12);
  background: rgba(20, 25, 45, 0.6);
  border-radius: 4px;
  cursor: pointer;
  color: rgba(160, 190, 230, 0.5);
  font-size: 11px;
  transition: all 0.15s;
}

.gantt-mode-btn:hover {
  background: rgba(40, 50, 80, 0.8);
  color: rgba(200, 220, 255, 0.8);
}

.gantt-mode-btn.active {
  background: rgba(30, 60, 100, 0.6);
  border-color: rgba(100, 180, 255, 0.4);
  color: rgba(200, 220, 255, 0.95);
}

.gantt-chart-container {
  flex: 1;
  overflow: hidden;
  min-height: 0;
}

.gantt-dual-view {
  display: flex;
  gap: 4px;
  height: 100%;
}

.gantt-half {
  flex: 1;
  min-width: 0;
  overflow: hidden;
}
</style>
