<template>
  <div class="train-page">
    <!-- 标题 -->
    <h2 class="page-title">Train Management</h2>

    <!-- 1. 基础信息 -->
    <el-card class="section-card">
      <template #header>
        <div class="card-header">📌 Basic Info</div>
      </template>
      <el-form label-width="120px" :model="basicInfo">
        <el-form-item label="Task Name">
          <el-input v-model="basicInfo.taskName" placeholder="Enter task name" />
        </el-form-item>
        <el-form-item label="Description">
          <el-input
            v-model="basicInfo.description"
            type="textarea"
            placeholder="Brief description"
          />
        </el-form-item>
        <el-form-item label="Model Selection">
          <el-select v-model="basicInfo.model" placeholder="Choose model">
            <el-option label="Scheduling Optimizer" value="scheduler" />
            <el-option label="Failure Predictor" value="predictor" />
            <el-option label="Energy Optimizer" value="energy" />
          </el-select>
        </el-form-item>
        <el-form-item label="Dataset">
          <el-upload
            drag
            multiple
            :auto-upload="false"
            :on-change="handleDatasetUpload"
            :file-list="fileList"
          >
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              Drop dataset here or <em>click to add</em>
            </div>
          </el-upload>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 2. 参数配置 -->
    <el-card class="section-card">
      <template #header>
        <div class="card-header">⚙️ Training Parameters</div>
      </template>
      <el-form label-width="120px" :model="params">
        <el-form-item label="Batch Size">
          <el-input-number v-model="params.batchSize" :min="1" />
        </el-form-item>
        <el-form-item label="Learning Rate">
          <el-input-number v-model="params.learningRate" :step="0.0001" />
        </el-form-item>
        <el-form-item label="Epochs">
          <el-input-number v-model="params.epochs" :min="1" />
        </el-form-item>
        <el-form-item label="Device">
          <el-select v-model="params.device" placeholder="Select device">
            <el-option label="CPU" value="cpu" />
            <el-option label="GPU (1)" value="gpu1" />
            <el-option label="GPU (Multi)" value="gpu-multi" />
          </el-select>
        </el-form-item>
        <el-button type="success" @click="startTraining">🚀 Start Training</el-button>
      </el-form>
    </el-card>

    <!-- 3. 运行监控 -->
    <el-card class="section-card">
      <template #header>
        <div class="card-header">📊 Training Monitor</div>
      </template>
      <div class="chart-container">
        <x-vue-echarts class="echart" :option="chartOption" autoresize />
      </div>
    </el-card>

    <!-- 4. 结果管理 -->
    <el-card class="section-card">
      <template #header>
        <div class="card-header">📂 Training History</div>
      </template>
      <el-table :data="history" stripe style="width: 100%">
        <el-table-column prop="time" label="Time" width="180" />
        <el-table-column prop="task" label="Task Name" />
        <el-table-column prop="model" label="Model" />
        <el-table-column prop="status" label="Status" />
        <el-table-column prop="metrics" label="Metrics" />
      </el-table>
    </el-card>
  </div>
</template>

<script>
import { ref } from "vue";
import { UploadFilled } from "@element-plus/icons-vue";

export default {
  name: "TrainManage",
  components: { UploadFilled },
  setup() {
    // 基础信息
    const basicInfo = ref({
      taskName: "",
      description: "",
      model: "",
    });

    // 上传数据集
    const fileList = ref([]);
    const handleDatasetUpload = (file, fileListRef) => {
      fileList.value = fileListRef;
    };

    // 参数
    const params = ref({
      batchSize: 32,
      learningRate: 0.001,
      epochs: 10,
      device: "cpu",
    });

    // 训练历史
    const history = ref([
      { time: "2025-09-18 14:00", task: "Job Scheduler", model: "scheduler", status: "Success", metrics: "Acc: 92%" },
      { time: "2025-09-17 10:20", task: "Failure Prediction", model: "predictor", status: "Failed", metrics: "-" },
    ]);

    // ECharts 示例
    const chartOption = ref({
      title: { text: "Training Loss Curve" },
      xAxis: { type: "category", data: ["1", "2", "3", "4", "5"] },
      yAxis: { type: "value" },
      series: [
        {
          data: [10, 8, 6, 4, 3],
          type: "line",
          smooth: true,
        },
      ],
    });

    // 开始训练
    const startTraining = () => {
      console.log("Start training with params:", params.value);
    };

    return {
      basicInfo,
      params,
      fileList,
      handleDatasetUpload,
      history,
      chartOption,
      startTraining,
    };
  },
};
</script>

<style scoped>
.train-page {
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
}

.page-title {
  font-size: 26px;
  font-weight: bold;
  margin-bottom: 20px;
}

.section-card {
  margin-bottom: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}

.card-header {
  font-weight: bold;
  font-size: 16px;
}

.chart-container {
  height: 300px;
}

.echart {
  width: 100%;
  height: 100%;
}
</style>
