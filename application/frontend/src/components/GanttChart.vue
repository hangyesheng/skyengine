<template>
  <div class="gantt-chart-container">
    <div class="gantt-header">
      <h3>{{ title }}</h3>
      <el-button size="small" @click="refreshData" :loading="loading">
        <el-icon><Refresh /></el-icon>
        刷新
      </el-button>
    </div>
    
    <div v-if="loading" class="loading-state">
      <el-icon class="is-loading"><Loading /></el-icon>
      <span>加载中...</span>
    </div>
    
    <div v-else-if="!hasData" class="empty-state">
      <el-empty description="暂无数据" />
    </div>
    
    <div v-else class="gantt-content" ref="ganttContentRef">
      <!-- 时间轴 -->
      <div class="timeline-axis">
        <div 
          v-for="(tick, index) in timeTicks" 
          :key="index"
          class="time-tick"
          :style="{ left: tick.position + '%' }"
        >
          {{ tick.label }}
        </div>
      </div>
      
      <!-- 资源行 -->
      <div 
        v-for="resource in resources" 
        :key="resource.id"
        class="resource-row"
      >
        <div class="resource-label">
          {{ type === 'agv' ? 'AGV' : 'Machine' }} {{ resource.id }}
        </div>
        <div class="resource-timeline">
          <!-- 有负载的运输操作条 -->
          <div
            v-for="op in resource.operations"
            :key="'op-' + op.operation_id"
            class="operation-bar"
            :class="{ 'agv': type === 'agv', 'machine': type === 'machine' }"
            :style="{
              left: calculatePosition(op.start_time) + '%',
              width: calculateWidth(op.start_time, op.end_time) + '%'
            }"
            @mouseenter="showTooltip($event, op)"
            @mouseleave="hideTooltip"
          >
            <span class="op-label">Op{{ op.operation_id }}</span>
          </div>
          
          <!-- 空载移动操作条（仅AGV类型显示） -->
          <template v-if="type === 'agv' && resource.empty_moves">
            <div
              v-for="(emptyMove, idx) in resource.empty_moves"
              :key="'empty-' + idx"
              class="operation-bar empty-move"
              :style="{
                left: calculatePosition(emptyMove.start_time) + '%',
                width: calculateWidth(emptyMove.start_time, emptyMove.end_time) + '%'
              }"
              @mouseenter="showTooltip($event, emptyMove)"
              @mouseleave="hideTooltip"
            >
              <span class="op-label">Empty</span>
            </div>
          </template>

        </div>
      </div>
      
      <!-- 悬浮提示框 -->
      <div 
        v-if="tooltip.visible"
        class="tooltip"
        :style="{ top: tooltip.y + 'px', left: tooltip.x + 'px' }"
      >
        <template v-if="type === 'agv'">
          <div v-if="tooltip.data?.is_empty_move" class="tooltip-item"><strong>Type:</strong> Empty Move (空载)</div>
          <div v-else class="tooltip-item"><strong>Type:</strong> Loaded Transport (有负载)</div>
        </template>
        <div v-if="!tooltip.data?.is_empty_move" class="tooltip-item"><strong>Operation:</strong> {{ tooltip.data?.operation_id }}</div>
        <div v-if="!tooltip.data?.is_empty_move" class="tooltip-item"><strong>Job:</strong> {{ tooltip.data?.job_id }}</div>
        <div class="tooltip-item"><strong>Start:</strong> {{ tooltip.data?.start_time?.toFixed(2) }}s</div>
        <div class="tooltip-item"><strong>End:</strong> {{ tooltip.data?.end_time?.toFixed(2) }}s</div>
        <div class="tooltip-item"><strong>Duration:</strong> {{ (tooltip.data?.end_time - tooltip.data?.start_time)?.toFixed(2) }}s</div>
        <div v-if="type === 'agv'" class="tooltip-item">
          <strong>From→To:</strong> {{ tooltip.data?.from_machine }}→{{ tooltip.data?.to_machine }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch, onUnmounted } from 'vue';
import { Refresh, Loading } from '@element-plus/icons-vue';
import axios from 'axios';

const props = defineProps({
  type: {
    type: String,
    required: true,
    validator: (value) => ['agv', 'machine'].includes(value)
  },
  autoRefresh: {
    type: Boolean,
    default: false
  },
  refreshInterval: {
    type: Number,
    default: 3000
  }
});

const title = computed(() => {
  return props.type === 'agv' ? 'AGV Transport Gantt Chart' : 'Machine Processing Gantt Chart';
});

const loading = ref(false);
const rawData = ref([]);
const ganttContentRef = ref(null);
const tooltip = ref({
  visible: false,
  x: 0,
  y: 0,
  data: null
});

let refreshTimer = null;

// 计算资源列表
const resources = computed(() => {
  if (!rawData.value || rawData.value.length === 0) return [];
  
  if (props.type === 'agv') {
    return rawData.value.agvs || [];
  } else {
    return rawData.value.machines || [];
  }
});

// 检查是否有数据
const hasData = computed(() => {
  return resources.value.some(resource => resource.operations && resource.operations.length > 0);
});

// 计算时间范围
const timeRange = computed(() => {
  let minTime = Infinity;
  let maxTime = 0;
  
  resources.value.forEach(resource => {
    if (resource.operations) {
      resource.operations.forEach(op => {
        if (op.start_time !== null && op.start_time !== undefined) {
          minTime = Math.min(minTime, op.start_time);
        }
        if (op.end_time !== null && op.end_time !== undefined) {
          maxTime = Math.max(maxTime, op.end_time);
        }
      });
    }
  });
  
  if (minTime === Infinity || maxTime === 0) {
    return { min: 0, max: 100 };
  }
  
  // 添加5%的边距
  const padding = (maxTime - minTime) * 0.05;
  return {
    min: Math.max(0, minTime - padding),
    max: maxTime + padding
  };
});

// 计算时间刻度
const timeTicks = computed(() => {
  const range = timeRange.value.max - timeRange.value.min;
  const ticks = [];
  const tickCount = 10; // 显示10个刻度
  
  for (let i = 0; i <= tickCount; i++) {
    const time = timeRange.value.min + (range * i / tickCount);
    ticks.push({
      label: time.toFixed(1) + 's',
      position: (i / tickCount) * 100
    });
  }
  
  return ticks;
});

// 计算位置百分比
const calculatePosition = (time) => {
  if (time === null || time === undefined) return 0;
  const range = timeRange.value.max - timeRange.value.min;
  if (range === 0) return 0;
  return ((time - timeRange.value.min) / range) * 100;
};

// 计算宽度百分比
const calculateWidth = (startTime, endTime) => {
  if (startTime === null || startTime === undefined || 
      endTime === null || endTime === undefined) return 0;
  const range = timeRange.value.max - timeRange.value.min;
  if (range === 0) return 0;
  return ((endTime - startTime) / range) * 100;
};

// 显示提示框
const showTooltip = (event, operation) => {
  const rect = event.target.getBoundingClientRect();
  tooltip.value = {
    visible: true,
    x: rect.left + rect.width / 2,
    y: rect.top - 10,
    data: operation
  };
};

// 隐藏提示框
const hideTooltip = () => {
  tooltip.value.visible = false;
};

// 获取数据
const fetchData = async () => {
  loading.value = true;
  try {
    const endpoint = props.type === 'agv' ? '/api/gantt/agv' : '/api/gantt/machine';
    const response = await axios.get(endpoint);
    rawData.value = response.data;
  } catch (error) {
    console.error(`Failed to fetch ${props.type} gantt data:`, error);
  } finally {
    loading.value = false;
  }
};

// 刷新数据
const refreshData = () => {
  fetchData();
};

// 监听类型变化
watch(() => props.type, () => {
  fetchData();
});

// 自动刷新
onMounted(() => {
  fetchData();
  
  if (props.autoRefresh) {
    refreshTimer = setInterval(fetchData, props.refreshInterval);
  }
});

// 清理定时器
onUnmounted(() => {
  if (refreshTimer) {
    clearInterval(refreshTimer);
  }
});
</script>

<style scoped>
.gantt-chart-container {
  background: white;
  border-radius: 8px;
  padding: 15px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  height: 100%;
  display: flex;
  flex-direction: column;
}

.gantt-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 2px solid #e4e7ed;
}

.gantt-header h3 {
  margin: 0;
  font-size: 16px;
  color: #303133;
}

.loading-state, .empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: #909399;
}

.loading-state .el-icon {
  font-size: 32px;
  margin-bottom: 10px;
}

.gantt-content {
  flex: 1;
  overflow: auto;
  position: relative;
  min-height: 200px;
}

.timeline-axis {
  position: sticky;
  top: 0;
  background: white;
  height: 30px;
  border-bottom: 1px solid #dcdfe6;
  margin-left: 100px;
  z-index: 10;
}

.time-tick {
  position: absolute;
  transform: translateX(-50%);
  font-size: 11px;
  color: #606266;
  white-space: nowrap;
}

.resource-row {
  display: flex;
  align-items: center;
  height: 40px;
  border-bottom: 1px solid #f0f0f0;
}

.resource-row:hover {
  background-color: #f5f7fa;
}

.resource-label {
  width: 100px;
  flex-shrink: 0;
  font-size: 13px;
  color: #606266;
  font-weight: 500;
  padding-right: 10px;
  text-align: right;
}

.resource-timeline {
  flex: 1;
  position: relative;
  height: 100%;
}

.operation-bar {
  position: absolute;
  height: 24px;
  top: 8px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.operation-bar:hover {
  transform: translateY(-2px);
  box-shadow: 0 3px 6px rgba(0, 0, 0, 0.15);
  z-index: 5;
}

/* AGV样式 - 紫色渐变 */
.operation-bar.agv {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: 1px solid #667eea;
}

.operation-bar.agv .op-label {
  color: white;
}

/* Machine样式 - 粉色渐变 */
.operation-bar.machine {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  border: 1px solid #f093fb;
}

.operation-bar.machine .op-label {
  color: white;
}

/* 空载移动样式 - 灰色虚线边框 */
.operation-bar.empty-move {
  background: linear-gradient(135deg, #e0e0e0 0%, #bdbdbd 100%);
  border: 2px dashed #9e9e9e;
  opacity: 0.7;
}

.operation-bar.empty-move .op-label {
  color: #616161;
  font-style: italic;
}

.op-label {
  font-size: 10px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding: 0 4px;
}

.tooltip {
  position: fixed;
  background: rgba(0, 0, 0, 0.85);
  color: white;
  padding: 10px 12px;
  border-radius: 6px;
  font-size: 12px;
  z-index: 1000;
  pointer-events: none;
  transform: translate(-50%, -100%);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  min-width: 150px;
}

.tooltip-item {
  margin: 4px 0;
  line-height: 1.5;
}

.tooltip-item strong {
  color: #ffd700;
}
</style>
