<template>
  <div class="outline">
    <div>
      <h3>GridFactory Management</h3>
    </div>
    <!--  上传区域,动态配置  -->
    <div>
      <div class="new-dag-font-style">Upload Config Set</div>
      <div class="upload-config-box">
        <ElRow>
          <ElCol :span="8">
            <el-card style="max-width: 550px;height: 180px">
              <template #header>
                <div class="card-header">
                  <el-tag @click="test">Config Set Operation</el-tag>
                </div>
              </template>
              <ElRow>
                <ElCol :span="11">
                  <ElButton type="success" plain @click="getStandardConfig">
                    Get Standard
                  </ElButton>
                </ElCol>
                <ElCol :span="5">
                  <ElButton type="primary" plain @click="uploadConfigSet">
                    Upload
                  </ElButton>
                </ElCol>
                <ElCol :span="8">
                  <ElInput placeholder="Set Config Name!" v-model="config_name">
                  </ElInput>
                </ElCol>
              </ElRow>
            </el-card>
          </ElCol>
          <ElCol :span="1">

          </ElCol>
          <ElCol :span="15">
            <el-upload
                ref="uploadRef"
                drag
                multiple
                :auto-upload="false"
                :on-change="handleChange"
                :file-list="fileList"
                :action="target_url"
            >
              <el-icon class="el-icon--upload">
                <upload-filled/>
              </el-icon>
              <div class="el-upload__text">
                Drop file here or <em>click to add !</em>
              </div>
              <template #tip>
                <div class="el-upload__tip">
                  A config set should contain 3 config files
                </div>
              </template>
            </el-upload>
          </ElCol>


        </ElRow>

      </div>


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

    <!-- 展示控制区域,展示运行系统 -->
    <div class="factory-container">
      <el-row :gutter="20">
        <!-- 左侧控制面板 -->
        <el-col :span="6">
          <!-- Control Panel -->
          <el-card class="panel-card" shadow="hover">
            <div class="panel-header">
              <span>Control Panel</span>
            </div>

            <el-form label-position="top" class="control-form">
              <el-form-item label="选择工厂">
                <el-select v-model="selectedFactory" placeholder="Select Factory">
                  <el-option
                      v-for="factory in stores.factoryList"
                      :key="factory.id"
                      :label="'Factory: ' + factory.id"
                      :value="factory.id"
                  />
                </el-select>
              </el-form-item>
              <el-form-item label="选择任务集">
                <el-select v-model="selectedJob" placeholder="Select Job">
                  <el-option
                      v-for="job in stores.jobList"
                      :key="job.id"
                      :label="'Job: ' + job.id"
                      :value="job.id"
                  />
                </el-select>
              </el-form-item>
              <el-form-item label="操作">
                <el-row :gutter="8">
                  <el-col :span="6">
                    <el-button type="primary" @click="handleFactoryRender" style="width: 100%">Render</el-button>
                  </el-col>
                  <el-col :span="6">
                    <el-button type="success" @click="handleFactoryStart" style="width: 100%">Start</el-button>
                  </el-col>
                  <el-col :span="6">
                    <el-button type="warning" @click="handleFactoryPause" style="width: 100%">Pause</el-button>
                  </el-col>
                  <el-col :span="6">
                    <el-button type="danger" @click="handleFactoryReset" style="width: 100%">Reset</el-button>
                  </el-col>

                </el-row>
              </el-form-item>

              <el-form-item label="Speed Level">
                <el-slider v-model="speedLevel" :min="1" :max="10" show-input @change="changeSpeed"/>
              </el-form-item>
            </el-form>
          </el-card>

          <!-- Agent Control -->
          <el-card class="panel-card mt-3" shadow="hover">
            <div class="panel-header">
              <span>Agent Control</span>
            </div>
            <el-form label-position="top" class="control-form">
              <el-form-item label="AGV Agent">
                <el-col :span="16">
                  <el-select v-model="selectedAgvAgent" placeholder="Select AGV Agent" style="width: 100%">
                    <el-option
                        v-for="agv_agent in agvAgentList"
                        :key="agv_agent.id"
                        :label="' AgvAgent' + job.id"
                        :value="agv_agent.id"
                    />
                  </el-select>
                </el-col>
                <el-col :span="8">
                  <el-button type="primary" @click="addJob" style="width: 100%">Add</el-button>
                </el-col>
              </el-form-item>
              <el-form-item label="System Agent">
                <el-col :span="16">
                  <el-select v-model="selectedSystemAgent" placeholder="Select System Agent" style="width: 100%">
                    <el-option
                        v-for="system_agent in jobList"
                        :key="system_agent.id"
                        :label="'System Agent' + system_agent.id"
                        :value="system_agent.id"
                    />
                  </el-select>
                </el-col>
                <el-col :span="8">
                  <el-button type="primary" @click="addJob" style="width: 100%">Add</el-button>
                </el-col>
              </el-form-item>

            </el-form>


          </el-card>
          <!-- Job Control -->
          <el-card class="panel-card mt-3" shadow="hover">
            <div class="panel-header">
              <span>Job Control</span>
            </div>
            <el-row :gutter="8" align="middle">
              <el-col :span="16">
                <el-select v-model="selectedJob" placeholder="Select Job" style="width: 100%">
                  <el-option
                      v-for="job in jobList"
                      :key="job.id"
                      :label="'Job ' + job.id"
                      :value="job.id"
                  />
                </el-select>
              </el-col>
              <el-col :span="8">
                <el-button type="primary" @click="addJob" style="width: 100%">Add</el-button>
              </el-col>
            </el-row>
          </el-card>

          <!-- Job Pools -->
          <el-card class="panel-card mt-3" shadow="hover">
            <div class="panel-header">
              <span>Job Pools</span>
            </div>
            <el-collapse>
              <el-collapse-item
                  v-for="(jobs, jobId) in jobDict"
                  :key="jobId"
                  :name="jobId"
              >
                <template #title>
                  <span>Job Pool {{ jobId }}</span>
                </template>
                <div v-for="job in jobs" :key="job.id" class="job-item">
                  <div class="job-info">
                    <span>Job {{ job.id }}</span>
                    <span :class="job.status === 'FINISHED' ? 'status-finished' : 'status-processing'">
                    {{ job.status === 'FINISHED' ? 'Finished' : 'Processing...' }}
                  </span>
                  </div>
                  <el-progress
                      :percentage="job.progress"
                      :status="job.status === 'FINISHED' ? 'success' : undefined"
                  />
                </div>
              </el-collapse-item>
            </el-collapse>
          </el-card>
        </el-col>

        <!-- 右侧工厂地图 -->
        <el-col :span="18">
          <el-card class="map-card" shadow="hover">
            <div class="map-header">
              <span>Current Factory Map</span>
              <div class="map-controls">
              </div>
            </div>
            <div class="map-container">
              <el-image :src="map_src" style="width: 100%; height: 100%;" fit="cover"/>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script setup>
import {
  ElButton, ElCol, ElInput, ElMessage, ElRow, ElTable, ElTableColumn, ElTag, ElTooltip, ElImage,
} from "element-plus";

import {ref, onMounted, onUnmounted} from "vue";
import {UploadFilled} from '@element-plus/icons-vue'
import axios from "axios";
import {useFactoryState} from "/@/stores/factoryState.ts"

const uploadRef = ref();
const config_name = ref('');
const fileList = ref([]);
const target_url = ref('');
const map_src = ref("");
const selectedFactory = ref('');
let agvIntervalId = null;
let machineIntervalId = null;
let jobIntervalId = null;
let intervalId = null;
const stores = useFactoryState()
const speedLevel = ref(1);

// ---------- 智能体相关流程 ----------
const selectedAgent = ref("") //系统开始时会返回当前init Agent
const agentList = ref([
  {value: "agent1", label: "A*算法 Agent"},
  {value: "agent2", label: "智能Agent 2"},
  {value: "agent3", label: "智能Agent 3"},
])

const updateAgent = () => {
  const data = {
    'agent': selectedAgent.value
  }
  fetch("/api/agent/update", {
    method: "POST",
    body: JSON.stringify(data),
  }).then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      }
  )
      .then((data) => {
        console.log(data)
      })
      .catch((error) => {
      });
}

const jobList = ref([]);

let checkAliveInterval = null; // 用于持续检查 factory/alive 的定时器

function updateMapSrcBuffered() {
  const preloadImg = new Image();
  const newSrc = `/api/map/update?_t=${Date.now()}`;

  preloadImg.onload = () => {
    map_src.value = newSrc; // 图片加载完后再替换
  };
  preloadImg.src = newSrc;
}

onMounted(() => {
  updateCurrentFactoryMapList()
});
onUnmounted(() => {
  clearInterval(agvIntervalId);
  clearInterval(machineIntervalId);
  clearInterval(jobIntervalId);
  clearInterval(intervalId);
  clearInterval(checkAliveInterval);
});

const handleChange = (uploadFile, uploadFiles) => {
  fileList.value.push(uploadFile)
  updateCurrentFactoryMapList()
}
const getStandardConfig = () => {
  fetch(`/api/standard/get?t=${Date.now()}`, {
    method: "GET"
  })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status} ${response.statusText}`);
        }

        // 从响应头获取 filename
        const disposition = response.headers.get('Content-Disposition');
        let filename = 'template_config_set.zip';  // 默认

        if (disposition && disposition.includes('filename=')) {
          filename = disposition.split('filename=')[1].replace(/"/g, '');
        }

        return response.blob().then(blob => ({blob, filename}));
      })
      .then(({blob, filename}) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;  // 动态指定 filename
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      })
      .catch(error => {
        ElMessage.error("Download failed");
        console.error(error);
      });
}

const uploadConfigSet = () => {
  target_url.value = '/api/' + config_name.value + `/yaml/upload?t=${Date.now()}`
  uploadRef.value.submit()
  fileList.value = []
  ElMessage.success("Upload success!");
}

const updateCurrentFactoryMapList = () => {
  fetch("/api/factory/list", {
    method: "GET",
  }).then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      }
  )
      .then((data) => {
        stores.factoryList = data.factory_list;
      })
      .catch((error) => {
      });
}


const handleFactoryRender = () => {
  fetch("/api/map/render", {
    method: "POST",
    body: JSON.stringify({target_factory: selectedFactory.value}),
  })
      .then((response) => {
        agvIntervalId = setInterval(loadAgvs(), 1000);
        machineIntervalId = setInterval(loadMachines(), 1000);
        jobIntervalId = setInterval(loadJobs(), 1000);
        console.log(response);
      })
      .catch((error) => {
      });
}

// 测试factory是否alive，若alive则更新下拉菜单和图片
const checkFactoryAlive = async () => {
  fetch("/api/factory/alive", {
    method: "GET",
  })
      .then((response) => {
        console.log(response)
        return response.json();
      })
      .then((data) => {
        console.log(data);
        if (data.is_alive) {
          agvIntervalId = setInterval(loadAgvs(), 1000);
          machineIntervalId = setInterval(loadMachines(), 1000);
          jobIntervalId = setInterval(loadJobs(), 1000);

          // 启动成功,则每隔 0.2 秒更新一次图片
          intervalId = setInterval(updateMapSrcBuffered, 250);

          clearInterval(checkAliveInterval);
        }
      })
      .catch((error) => {
        console.log(error);
      });
};

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

</script>

<style scoped>
h3 {
  font-size: 24px;
  color: #333;
  margin-bottom: 20px;
}

.factory-container {
  padding: 20px;
}

body {
  font-family: Arial, sans-serif;
  background-color: #f9f9f9;
  margin: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

.panel-card {
  border-radius: 12px;
}

.upload-config-box {
  padding: 20px;
  background-color: #f0f8ff; /* 浅蓝背景 */
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  margin: 0 auto;
}

.new-dag-font-style {
  font-size: 16px;
  margin-bottom: 15px;
  font-weight: bold;
}

.panel-card .panel-header {
  font-weight: bold;
  font-size: 16px;
  margin-bottom: 12px;
}

.mt-3 {
  margin-top: 16px;
}

.job-item {
  margin-bottom: 12px;
}

.job-info {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
}

.status-finished {
  color: green;
  font-weight: bold;
}

.status-processing {
  color: #409EFF;
  font-weight: bold;
}

.map-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.map-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  font-weight: bold;
  font-size: 18px;
}

.map-controls {
  display: flex;
  gap: 12px;
  align-items: center;
}

.map-container {
  width: 100%;
  height: 600px;
  background: #fafafa;
  border-radius: 12px;
  overflow: hidden;
}
</style>