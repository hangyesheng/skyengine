<template>
  <div class="outline">
    <div>
      <h3>Factory Management</h3>
    </div>

    <div>
      <div class="new-dag-font-style">Upload Config Set</div>
      <el-upload
          class="upload-demo"
          drag
          action="https://run.mocky.io/v3/9d059bf9-4660-45f2-925d-ce80ad6c4d15"
          multiple
      >
        <el-icon class="el-icon--upload">
          <upload-filled/>
        </el-icon>
        <div class="el-upload__text">
          Drop file here or <em>click to upload</em>
        </div>
        <template #tip>
          <div class="el-upload__tip">
            A config set should contain 3 config files
          </div>
        </template>
      </el-upload>

      <br/>
      <br/>
      <div style="display: inline">
        <div class="new-dag-font-style">
          Factory Info
          <el-tooltip placement="right">
            <template #content>
              Both logical and physical included.
            </template>
            <el-button size="small" circle>i</el-button>
          </el-tooltip>
        </div>
      </div>

      <div>
        <ul style="list-style-type: none" class="svc-container">
          <li
              v-for="(service, index) in services"
              :key="index"
              class="svc-item"
          >
            <el-tooltip
                placement="right"
                :open-delay="500"
                enterable
            >
              <template #content>
                <div class="description">
                  {{ service.description }}
                </div>
              </template>
              <div
                  class="vue-flow__node-input"
                  :draggable="true"
                  @dragstart="onDragStart($event, '', service)"
              >
                {{ service.name }}
              </div>
            </el-tooltip>
          </li>
        </ul>
      </div>
    </div>
    <br/>

    <!--    <div>-->
    <!--      <ElRow>-->
    <!--        <ElCol :span="2">-->
    <!--          <el-button type="warning" @click="draw">Draw</el-button>-->
    <!--        </ElCol>-->
    <!--        <ElCol :span="18"></ElCol>-->
    <!--        <ElCol :span="2">-->
    <!--          <el-button-->
    <!--              type="primary"-->
    <!--              round-->
    <!--              @click="handleNewSubmit"-->
    <!--              v-if="drawing"-->
    <!--          >Add-->
    <!--          </el-button-->
    <!--          >-->
    <!--        </ElCol>-->
    <!--        <ElCol :span="2">-->
    <!--          <el-button type="primary" round @click="clearInput" v-if="drawing"-->
    <!--          >Clear-->
    <!--          </el-button-->
    <!--          >-->
    <!--        </ElCol>-->
    <!--      </ElRow>-->
    <!--    </div>-->

    <!-- 绘制区域,自动绘制 -->
    <div>
      <ElRow>
        <ElCol :span="18">
          <div
              class="draw-container"
              v-if="drawing"
              @drop="onDrop($event, flowNodes, flowNodeMap)"
          >
            <VueFlow
                class="main-flow"
                :nodes="flowNodes"
                :edges="flowEdges"
                :default-viewport="{ zoom: 1.5 }"
                :min-zoom="1"
                :max-zoom="2"
                :fit-view-on-init="false"
                @dragover="onDragOver"
                @dragleave="onDragLeave"
            >
              <div v-if="drawing" class="drag-tip">
                <el-icon class="tip-icon">
                  <MagicStick/>
                </el-icon>
                <span>Drag service nodes to build your workflow</span>
              </div>

              <Background pattern-color="#aaa" :gap="16"/>

              <MiniMap/>
              <Panel class="process-panel" position="top-right">
                <div class="layout-panel">
                  <button title="set horizontal layout" @click="layoutGraph('LR')">
                    <Icon name="horizontal"/>
                  </button>

                  <button title="set vertical layout" @click="layoutGraph('TB')">
                    <Icon name="vertical"/>
                  </button>
                </div>
              </Panel>
            </VueFlow>
          </div>
        </ElCol>

        <ElCol :span="1">

        </ElCol>

        <ElCol :span="5">
          <!-- Start/Pause/Reset 控制按钮 -->
          <el-card style="max-width: 480px">
            <el-button type="warning" @click="handleFactoryStart">开始模拟</el-button>
          </el-card>
          <el-card style="max-width: 480px">
            <el-button type="warning" @click="handleFactoryPause">暂停模拟</el-button>
          </el-card>
          <el-card style="max-width: 480px">
            <el-button type="warning" @click="handleFactoryReset">重置</el-button>
          </el-card>

          <!-- 速度调节 -->
          <el-card style="max-width: 480px; margin-top: 10px">
            <div style="text-align: center; margin-bottom: 10px;">请选择速度 (当前: {{ speedLevel }})</div>
            <el-slider v-model="speedLevel" :min="1" :max="10" show-input @change="changeSpeed"></el-slider>
          </el-card>

          <!-- AGV 控制 -->
          <el-card style="max-width: 480px; margin-top: 10px">
            <el-row :gutter="10">
              <el-col :span="14">
                <el-select v-model="selectedAgv" placeholder="选择 AGV" style="width: 100%">
                  <el-option
                    v-for="agv in agvList"
                    :key="agv.id"
                    :label="'AGV' + agv.id"
                    :value="agv.id"
                  />
                </el-select>
              </el-col>
              <el-col :span="5">
                <el-button @click="pauseAgv" type="primary" style="width: 100%">暂停</el-button>
              </el-col>
              <el-col :span="5">
                <el-button @click="resumeAgv" type="success" style="width: 100%">恢复</el-button>
              </el-col>
            </el-row>
          </el-card>

          <!-- 机器控制 -->
          <el-card style="max-width: 480px; margin-top: 10px">
            <el-row :gutter="10">
              <el-col :span="14">
                <el-select v-model="selectedMachine" placeholder="选择机器" style="width: 100%">
                  <el-option
                    v-for="machine in machineList"
                    :key="machine.id"
                    :label="'机器' + machine.id"
                    :value="machine.id"
                  />
                </el-select>
              </el-col>
              <el-col :span="5">
                <el-button @click="pauseMachine" type="primary" style="width: 100%">暂停</el-button>
              </el-col>
              <el-col :span="5">
                <el-button @click="resumeMachine" type="success" style="width: 100%">恢复</el-button>
              </el-col>
            </el-row>
          </el-card>

          <!-- Job 控制 -->
          <el-card style="max-width: 480px; margin-top: 10px">
            <el-row :gutter="10">
              <el-col :span="14">
                <el-select v-model="selectedJob" placeholder="选择任务" style="width: 100%">
                  <el-option
                    v-for="job in jobList"
                    :key="job.id"
                    :label="'任务' + job.id"
                    :value="job.id"
                  />
                </el-select>
              </el-col>
              <el-col :span="10">
                <el-button @click="addJob" type="primary" style="width: 100%">添加任务</el-button>
              </el-col>
            </el-row>
          </el-card>
        </ElCol>
      </ElRow>


    </div>
    <br/>
    <br/>
  </div>
</template>

<script>
import {ElButton, ElCol, ElInput, ElMessage, ElRow, ElTable, ElTableColumn, ElTag, ElTooltip,} from "element-plus";
import {nextTick, ref} from "vue";
import {MarkerType, Panel, useVueFlow, VueFlow} from "@vue-flow/core";
import {ControlButton, Controls} from "@vue-flow/controls";
import {Background} from "@vue-flow/background";
import {MiniMap} from "@vue-flow/minimap";
import useDragAndDrop from "./useDnD";
import Icon from "./Icon.vue";
import {useLayout} from "./useLayout";
import {Connection, Link, MagicStick, Right} from '@element-plus/icons-vue';
import {UploadFilled} from '@element-plus/icons-vue'
import axios from "axios";

export default {
  name: "DagManage",
  components: {
    nextTick,
    ElTable,
    ElTableColumn,
    ElTooltip,
    ElTag,
    ElInput,
    ElButton,
    ElCol,
    ElRow,
    ElMessage,
    VueFlow,
    Background,
    MarkerType,
    MiniMap,
    ControlButton,
    Controls,
    Icon,
    Panel,

    Connection,
    Link,
    Right,
    MagicStick,
    UploadFilled
  },
  setup() {
    const {onInit, onNodeDragStop, onConnect, fitView} = useVueFlow();

    const {onDragOver, onDrop, onDragLeave, isDragOver, onDragStart} =
        useDragAndDrop();

    const layoutMethods = useLayout();


    const flowNodes = ref([])
    const flowEdges = ref([])
    const flowNodeMap = ref({})

    const layoutGraph = async (direction) => {
      try {
        const layoutNodes = layout(
            [...flowNodes.value],
            [...flowEdges.value],
            direction
        )
        flowNodes.value = layoutNodes
        await nextTick()
        fitView()
      } catch (e) {
        console.error("Layout failed:", e)
        ElMessage.error("DAG layout error")
      }
    }

    // life cycle callback
    onInit((vueFlowInstance) => {
      vueFlowInstance.fitView();
    });
    onNodeDragStop(({event, nodes, node}) => {
    });
    onConnect((connection) => {
      const line = {
        id: connection.source + "-" + connection.target,
        source: connection.source,
        target: connection.target,
        label: "",
        markerEnd: MarkerType.ArrowClosed,
      };
      flowEdges.value.push(line);
      flowNodeMap.value[connection.source].data.succ.push(connection.target);
      flowNodeMap.value[connection.target].data.prev.push(connection.source);
    });

    const factoryStart = async () => {
      console.log("FactoryStart")
      try {
        const response = await axios.post('/api/factory/start');
        console.log('Response: ', response.data);
      } catch (error) {
        console.error('API request failed:', error.response ? error.response.data : error.message);
        throw error;
      }
    }

    const factoryPause = async () => {
      console.log("FactoryPause")
      try {
        const response = await axios.post('/api/factory/pause');
        console.log('Response: ', response.data);
      } catch (error) {
        console.error('API request failed:', error.response ? error.response.data : error.message);
        throw error;
      }
    }

    const factoryReset = async () => { 
      console.log("FactoryReset")
      try {
        const response = await axios.post('/api/factory/reset');
        console.log('Response: ', response.data);
      } catch (error) {
        console.error('API request failed:', error.response ? error.response.data : error.message);
        throw error;
      }
    }

    return {
      onDragOver,
      onDrop,
      onDragLeave,
      isDragOver,
      onDragStart,
      layoutGraph,
      flowNodes,
      flowEdges,
      flowNodeMap,
      factoryStart,
      factoryPause,
      factoryReset,
      ...layoutMethods,
    };
  },

  data() {
    return {
      services: [],
      editInput: "",
      newInputName: "",
      newInputDag: "",
      newInputDagId: "",
      editDisabled: true,
      editingIndex: -1,
      editingRow: null,

      // drawing flag
      drawing: false,

      speedLevel: 3,
      selectedAgv: null,
      agvList: [],
      selectedMachine: null,
      machineList: [],
      selectedJob: null,
      jobList: [],

      activeDag: null,
      hoverPosition: {x: 0, y: 0},
    };
  },

  methods: {
    flushDrawData() {
      this.flowNodes = [];
      this.flowEdges = [];
      this.flowNodeMap = {};
    },
    draw() {
      this.drawing = !this.drawing;
    },
    clearInput() {
      this.newInputName = "";
      this.newInputDag = "";
      this.newInputDagId = "";
      this.flushDrawData();
    },
    
    handleFactoryStart() {
      this.factoryStart();
    },

    handleFactoryPause() {
      this.factoryPause();
    },

    handleFactoryReset() {
      this.factoryReset();
    },

    async changeSpeed(value) {
      try {
        const res = await axios.post('/api/factory/speed', {
          speedLevel: value
        });
        this.$message.success(`速度已调整为 ${value}`);
        console.log('速度调整成功:', res.data);
      } catch (error) {
        console.error('速度调整失败:', error);
        this.$message.error('速度调整失败');
      }
    },

    async loadAgvs() {
      try {
        const res = await axios.get('/api/agvs');
        this.agvList = res.data.agvs;
      } catch (error) {
        this.$message.error('加载 AGV 列表失败');
        console.error('加载 AGV 失败:', error);
      }
    },

    async pauseAgv() {
      if (this.selectedAgv === null) {
        this.$message.warning('请先选择一个 AGV');
        return;
      }
      try {
        await axios.post(`/api/agv/pause/${this.selectedAgv}`, {
          agvId: this.selectedAgv
        });
        this.$message.success(`AGV ${this.selectedAgv} 已暂停`);
      } catch (error) {
        console.error('暂停 AGV 失败:', error);
        this.$message.error('暂停 AGV 失败');
      }
    },

    async resumeAgv() {
      if (this.selectedAgv === null) {
        this.$message.warning('请先选择一个 AGV');
        return;
      }
      try {
        await axios.post(`/api/agv/resume/${this.selectedAgv}`, {
          agvId: this.selectedAgv
        });
        this.$message.success(`AGV ${this.selectedAgv} 已恢复`);
      } catch (error) {
        console.error('恢复 AGV 失败:', error);
        this.$message.error('恢复 AGV 失败');
      }
    },

    async loadMachines() {
      try {
        const res = await axios.get('/api/machines');
        this.machineList = res.data.machines;
      } catch (error) {
        console.error('加载机器失败:', error);
        this.$message.error('加载机器列表失败');
      }
    },

    async pauseMachine() {
      if (this.selectedMachine === null) {
        this.$message.warning('请先选择一个机器');
        return;
      }
      try {
        await axios.post(`/api/machine/pause/${this.selectedMachine}`, {
          machineId: this.selectedMachine
        });
        this.$message.success(`机器 ${this.selectedMachine} 已暂停`);
      } catch (error) {
        console.error('暂停机器失败:', error);
        this.$message.error('暂停机器失败');
      }
    },

    async resumeMachine() {
      if (this.selectedMachine === null) {
        this.$message.warning('请先选择一个机器');
        return;
      }
      try {
        await axios.post(`/api/machine/resume/${this.selectedMachine}`, {
          machineId: this.selectedMachine
        });
        this.$message.success(`机器 ${this.selectedMachine} 已恢复`);
      } catch (error) {
        console.error('恢复机器失败:', error);
        this.$message.error('恢复机器失败');
      }
    },

    async loadJobs() {
      try {
        const res = await axios.get('/api/jobs');
        this.jobList = res.data.jobs;
      } catch (error) {
        console.error('加载任务失败:', error);
        this.$message.error('加载任务列表失败');
      }
    },

    async addJob() {
      if (this.selectedJob === null) {
        this.$message.warning('请先选择一个任务');
        return;
      }
      try {
        await axios.post(`/api/job/add/${this.selectedJob}`, {
          jobId: this.selectedJob
        });
        this.$message.success(`任务 ${this.selectedJob} 已添加`);
      } catch (error) {
        console.error('添加任务失败:', error);
        this.$message.error('添加任务失败');
      }
    },
  }
  ,
  mounted() {
    // init agv, machine, job data list
    this.loadAgvs();
    this.loadMachines();
    this.loadJobs();
  }
  ,
}
;
</script>

<style scoped>
body {
  font-family: Arial, sans-serif;
  background-color: #f9f9f9;
  margin: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

form {
  max-width: 600px;
  padding: 20px;
  background-color: #fff;
  border-radius: 8px;
  /* box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); */
}

h3 {
  font-size: 24px;
  color: #333;
  margin-bottom: 20px;
}

input[type="text"],
input[type="file"] {
  width: calc(100% - 20px);
  padding: 10px;
  margin-bottom: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 16px;
}

input[type="file"] {
  cursor: pointer;
}

.draw-container {
  width: 100%;
  height: 500px;

  display: flex;
  align-items: flex-start;
  justify-content: center;
  margin-top: 15px;
  border: 1px solid #ccc;
}

.compact-container {
  width: 100%;
  height: 100px;
}

.el-button {
  font-size: 16px;
  margin-right: 10px;
}

.el-button:first-child {
  margin-left: 0;
}

.outline {
  /* max-width: 600px; */
  padding: 20px;
  background-color: #fff;
  border-radius: 8px;
  /* box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); */
}

.new-dag-font-style {
  font-size: 16px;
  margin-bottom: 15px;
  font-weight: bold;
}

.svc-container {
  margin-top: -20px;
  display: flex;
  flex-wrap: wrap;
  padding: 10px;
  list-style-type: none;
}


.svc-item {
  display: inline-block;
  margin: 2px;
  padding: 2px;
  border-radius: 12px;
}

.vue-flow__node-input {
  border: 1px solid #e2e8f0; /* 柔和的蓝灰色 */
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  width: auto;
  padding: 10px;
  font-size: 18px;
}

.vue-flow__node-input:hover {
  border-color: #94a3b8; /* 悬停时稍深的颜色 */
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.vue-flow__node-input2 {
  width: auto;
  padding: 5px;
  font-size: 20px;
}

.description {
  white-space: pre-wrap;
  word-wrap: break-word;
}

.el-button {
  font-size: 16px;
  margin-right: 10px;
}

.el-button:first-child {
  margin-left: 0;
}

.process-panel,
.layout-panel {
  display: flex;
  gap: 10px;
}

.process-panel {
  background-color: #2d3748;
  padding: 10px;
  border-radius: 8px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
  display: flex;
  flex-direction: column;
}

.process-panel button {
  border: none;
  cursor: pointer;
  background-color: #4a5568;
  border-radius: 8px;
  color: white;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
}

.process-panel button {
  font-size: 16px;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.checkbox-panel {
  display: flex;
  align-items: center;
  gap: 10px;
}

.process-panel button:hover,
.layout-panel button:hover {
  background-color: #2563eb;
  transition: background-color 0.2s;
}

.process-panel label {
  color: white;
  font-size: 12px;
}

.drag-tip {
  position: absolute;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 10;
  background: rgba(255, 255, 255, 0.9);
  padding: 8px 16px;
  border-radius: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
  gap: 8px;
  backdrop-filter: blur(4px);
  border: 1px solid #ebeef5;
  animation: float 3s ease-in-out infinite;
}

.tip-icon {
  color: #409eff;
  font-size: 18px;
}

.drag-tip span {
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 14px;
  color: #606266;
  font-weight: 500;
  letter-spacing: 0.3px;
}

@keyframes float {
  0%, 100% {
    transform: translateX(-50%) translateY(0);
  }
  50% {
    transform: translateX(-50%) translateY(-4px);
  }
}


.dag-preview {
  position: relative;
  padding: 8px;
  border-radius: 8px;
  background: #f8fafc;
  transition: all 0.3s;
}

.mini-dag {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-bottom: 8px;

  .node {
    padding: 2px 8px;
    background: #e2e8f0;
    border-radius: 4px;
    font-size: 12px;
  }

  .edges {
    color: #94a3b8;
    display: flex;
    gap: 2px;
  }
}

.stats {
  display: flex;
  gap: 8px;

  .el-tag {
    display: flex;
    align-items: center;
    gap: 4px;
  }
}


.dag-detail-card {
  position: fixed;
  z-index: 99999 !important;
  pointer-events: none;
  width: 400px;
  height: 300px;
  right: 120px;
  left: auto !important;
  top: 40% !important;
  transform: translate(-20px, -50%) !important;
  background: white;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
  padding: 16px;
  animation: slide-in 0.3s ease;
  max-width: 90vw;
  max-height: 80vh;
  overflow: auto;
  transition: all 0.3s ease;

  .dag-title {
    font-weight: 600;
    margin-bottom: 12px;
    color: #2d3748;
  }


}

@keyframes slide-in {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.dag-detail-card .vue-flow__node {
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 14px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}

.dag-detail-card .vue-flow__edge-path {
  stroke: #64748b;
  stroke-width: 2;
}

.mini-dag .node {
  max-width: 80px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.edges {
  .no-edges {
    color: #999;
    font-size: 12px;
  }

  .el-icon {
    &:nth-child(n+4) {
      display: none;
    }
  }
}

.el-table__body tr.hover-row > td {
  z-index: auto !important;
}

.first-service {
  background: #e2e8f0;
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 12px;
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.main-flow .vue-flow__node {
}

.preview-flow {
  z-index: 100000;
  pointer-events: none;
}

.preview-flow .vue-flow__node {
  transform: scale(0.8);
  cursor: default !important;
}

</style>
