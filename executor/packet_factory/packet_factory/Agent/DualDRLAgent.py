"""
双层 DRL Agent - Routing Agent + Sequencing Agent

Routing Agent: 负责为作业分配机器（路由决策）
Sequencing Agent: 负责从队列中选择作业执行（排序决策）

网络结构:
- Routing Agent: Input -> InstanceNorm -> FC(16,tanh) x3 -> FC(8,tanh) x2 -> Output
- Sequencing Agent: 双路径输入 -> InstanceNorm -> Concat -> FC(48,36,36,24,24,12,tanh) -> Output
"""

from .BaseAgent import BaseAgent, DEFAULT_STEP_TIME, TRAINING, EVALUATION, INFERENCE
from executor.packet_factory.packet_factory.packet_factory_env.Job.Operation import Operation
from executor.packet_factory.packet_factory.packet_factory_env.Machine.Machine import Machine
from executor.packet_factory.packet_factory.packet_factory_env.Agv.AGV import AGV
from executor.packet_factory.packet_factory.packet_factory_env.Job.Job import Job
from executor.packet_factory.logger.logger import LOGGER
from executor.packet_factory.packet_factory.packet_factory_env.Utils.util import OperationStatus, MachineStatus, AGVStatus
from executor.packet_factory.registry import register_component

import numpy as np
import json
import os
from typing import List, Tuple, Any, Dict, Optional
import random
import time
import math


class InstanceNormalization:
    """实例归一化层"""
    
    def __init__(self, eps=1e-5):
        self.eps = eps
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """对输入进行实例归一化"""
        if x.ndim == 1:
            mean = np.mean(x)
            std = np.std(x)
            return (x - mean) / (std + self.eps)
        else:
            # 对每个样本独立归一化
            mean = np.mean(x, axis=-1, keepdims=True)
            std = np.std(x, axis=-1, keepdims=True)
            return (x - mean) / (std + self.eps)


class RoutingNetwork:
    """
    Routing Agent 神经网络
    
    结构: Input -> InstanceNorm -> FC(16,tanh) x3 -> FC(8,tanh) x2 -> Output
    """
    
    def __init__(self, input_dim: int, output_dim: int, learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # 实例归一化层
        self.instance_norm = InstanceNormalization()
        
        # 网络权重（简化版，实际应使用 PyTorch/TensorFlow）
        # FC(16) x3
        self.w1 = np.random.randn(input_dim, 16) * 0.01
        self.b1 = np.zeros(16)
        self.w2 = np.random.randn(16, 16) * 0.01
        self.b2 = np.zeros(16)
        self.w3 = np.random.randn(16, 16) * 0.01
        self.b3 = np.zeros(16)
        
        # FC(8) x2
        self.w4 = np.random.randn(16, 8) * 0.01
        self.b4 = np.zeros(8)
        self.w5 = np.random.randn(8, 8) * 0.01
        self.b5 = np.zeros(8)
        
        # 输出层
        self.w_out = np.random.randn(8, output_dim) * 0.01
        self.b_out = np.zeros(output_dim)
        
        # Q-table 用于离散动作空间
        self.q_table: Dict[str, float] = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        # 实例归一化
        x = self.instance_norm.normalize(x)
        
        # FC(16, tanh) x3
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        h3 = np.tanh(h2 @ self.w3 + self.b3)
        
        # FC(8, tanh) x2
        h4 = np.tanh(h3 @ self.w4 + self.b4)
        h5 = np.tanh(h4 @ self.w5 + self.b5)
        
        # 输出层
        out = h5 @ self.w_out + self.b_out
        
        return out
    
    def get_q_value(self, state_key: str, action_idx: int) -> float:
        """获取 Q 值"""
        key = f"{state_key}_{action_idx}"
        return self.q_table.get(key, 0.0)
    
    def update_q_value(self, state_key: str, action_idx: int, value: float):
        """更新 Q 值"""
        key = f"{state_key}_{action_idx}"
        self.q_table[key] = value
    
    def save(self, filepath: str):
        """保存网络参数"""
        params = {
            'w1': self.w1.tolist(),
            'b1': self.b1.tolist(),
            'w2': self.w2.tolist(),
            'b2': self.b2.tolist(),
            'w3': self.w3.tolist(),
            'b3': self.b3.tolist(),
            'w4': self.w4.tolist(),
            'b4': self.b4.tolist(),
            'w5': self.w5.tolist(),
            'b5': self.b5.tolist(),
            'w_out': self.w_out.tolist(),
            'b_out': self.b_out.tolist(),
            'q_table': self.q_table
        }
        with open(filepath, 'w') as f:
            json.dump(params, f)
        LOGGER.info(f"[RoutingNetwork] 模型已保存至: {filepath}")
    
    def load(self, filepath: str):
        """加载网络参数"""
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        self.w1 = np.array(params['w1'])
        self.b1 = np.array(params['b1'])
        self.w2 = np.array(params['w2'])
        self.b2 = np.array(params['b2'])
        self.w3 = np.array(params['w3'])
        self.b3 = np.array(params['b3'])
        self.w4 = np.array(params['w4'])
        self.b4 = np.array(params['b4'])
        self.w5 = np.array(params['w5'])
        self.b5 = np.array(params['b5'])
        self.w_out = np.array(params['w_out'])
        self.b_out = np.array(params['b_out'])
        self.q_table = params.get('q_table', {})
        LOGGER.info(f"[RoutingNetwork] 模型已从 {filepath} 加载")


class SequencingNetwork:
    """
    Sequencing Agent 神经网络
    
    结构: 双路径输入 -> InstanceNorm -> Concat -> FC(48,36,36,24,24,12,tanh) -> Output
    """
    
    def __init__(self, input_dim_path1: int, input_dim_path2: int, output_dim: int, 
                 learning_rate: float = 0.001):
        self.input_dim_path1 = input_dim_path1  # 路径1维度（子输入1-4）
        self.input_dim_path2 = input_dim_path2  # 路径2维度（子输入5-6）
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # 实例归一化层（6个）
        self.instance_norms = [InstanceNormalization() for _ in range(6)]
        
        # 拼接后的总维度
        concat_dim = input_dim_path1 + input_dim_path2
        
        # 隐藏层: FC(48, 36, 36, 24, 24, 12)
        self.w1 = np.random.randn(concat_dim, 48) * 0.01
        self.b1 = np.zeros(48)
        self.w2 = np.random.randn(48, 36) * 0.01
        self.b2 = np.zeros(36)
        self.w3 = np.random.randn(36, 36) * 0.01
        self.b3 = np.zeros(36)
        self.w4 = np.random.randn(36, 24) * 0.01
        self.b4 = np.zeros(24)
        self.w5 = np.random.randn(24, 24) * 0.01
        self.b5 = np.zeros(24)
        self.w6 = np.random.randn(24, 12) * 0.01
        self.b6 = np.zeros(12)
        
        # 输出层
        self.w_out = np.random.randn(12, output_dim) * 0.01
        self.b_out = np.zeros(output_dim)
        
        # Q-table
        self.q_table: Dict[str, float] = {}
    
    def forward(self, path1_input: np.ndarray, path2_input: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        :param path1_input: 路径1输入（子输入1-4展平后）
        :param path2_input: 路径2输入（子输入5-6展平后）
        :return: 输出向量
        """
        # 对6个子输入分别进行实例归一化（这里简化为对两条路径分别归一化）
        path1_norm = self.instance_norms[0].normalize(path1_input)
        path2_norm = self.instance_norms[1].normalize(path2_input)
        
        # 拼接
        x = np.concatenate([path1_norm, path2_norm])
        
        # 隐藏层
        h1 = np.tanh(x @ self.w1 + self.b1)
        h2 = np.tanh(h1 @ self.w2 + self.b2)
        h3 = np.tanh(h2 @ self.w3 + self.b3)
        h4 = np.tanh(h3 @ self.w4 + self.b4)
        h5 = np.tanh(h4 @ self.w5 + self.b5)
        h6 = np.tanh(h5 @ self.w6 + self.b6)
        
        # 输出层
        out = h6 @ self.w_out + self.b_out
        
        return out
    
    def get_q_value(self, state_key: str, action_idx: int) -> float:
        """获取 Q 值"""
        key = f"{state_key}_{action_idx}"
        return self.q_table.get(key, 0.0)
    
    def update_q_value(self, state_key: str, action_idx: int, value: float):
        """更新 Q 值"""
        key = f"{state_key}_{action_idx}"
        self.q_table[key] = value
    
    def save(self, filepath: str):
        """保存网络参数"""
        params = {
            'w1': self.w1.tolist(), 'b1': self.b1.tolist(),
            'w2': self.w2.tolist(), 'b2': self.b2.tolist(),
            'w3': self.w3.tolist(), 'b3': self.b3.tolist(),
            'w4': self.w4.tolist(), 'b4': self.b4.tolist(),
            'w5': self.w5.tolist(), 'b5': self.b5.tolist(),
            'w6': self.w6.tolist(), 'b6': self.b6.tolist(),
            'w_out': self.w_out.tolist(), 'b_out': self.b_out.tolist(),
            'q_table': self.q_table
        }
        with open(filepath, 'w') as f:
            json.dump(params, f)
        LOGGER.info(f"[SequencingNetwork] 模型已保存至: {filepath}")
    
    def load(self, filepath: str):
        """加载网络参数"""
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        self.w1 = np.array(params['w1'])
        self.b1 = np.array(params['b1'])
        self.w2 = np.array(params['w2'])
        self.b2 = np.array(params['b2'])
        self.w3 = np.array(params['w3'])
        self.b3 = np.array(params['b3'])
        self.w4 = np.array(params['w4'])
        self.b4 = np.array(params['b4'])
        self.w5 = np.array(params['w5'])
        self.b5 = np.array(params['b5'])
        self.w6 = np.array(params['w6'])
        self.b6 = np.array(params['b6'])
        self.w_out = np.array(params['w_out'])
        self.b_out = np.array(params['b_out'])
        self.q_table = params.get('q_table', {})
        LOGGER.info(f"[SequencingNetwork] 模型已从 {filepath} 加载")


@register_component("packet_factory.DualDRLAgent")
class DualDRLAgent(BaseAgent):
    """
    双层 DRL Agent
    
    包含 Routing Agent（路由决策）和 Sequencing Agent（排序决策）
    """
    
    def __init__(self, name=None, agent_id=None, context=None, mode: str = TRAINING, 
                 model_path: Optional[str] = None):
        """
        初始化双层 DRL Agent
        
        :param name: 智能体名称
        :param agent_id: 智能体 ID
        :param context: 环境上下文
        :param mode: 运行模式 training | evaluation | inference
        :param model_path: 模型文件路径
        """
        super().__init__(name, agent_id, context, mode, model_path)
        
        # ========== 网络配置 ==========
        # Routing Agent 配置
        self.routing_input_dim = 20  # 动态调整
        self.routing_output_dim = 50  # 最大机器数
        self.routing_network = None
        
        # Sequencing Agent 配置
        self.seq_path1_dim = 16  # 路径1维度
        self.seq_path2_dim = 8   # 路径2维度
        self.seq_output_dim = 100  # 最大作业数
        self.sequencing_network = None
        
        # ========== 强化学习参数 ==========
        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01
        
        # ========== 经验回放缓冲区 ==========
        self.replay_buffer: List[Dict] = []
        self.buffer_size = 10000
        self.batch_size = 64
        
        # ========== 训练统计 ==========
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0
        self.training_history: Dict[str, List] = {
            'episodes': [],
            'routing_rewards': [],
            'sequencing_rewards': [],
            'total_rewards': [],
            'makespans': [],
            'epsilon': []
        }
        
        # 最近的状态和动作
        self.last_routing_state: Optional[np.ndarray] = None
        self.last_routing_action: Optional[int] = None
        self.last_sequencing_state_path1: Optional[np.ndarray] = None
        self.last_sequencing_state_path2: Optional[np.ndarray] = None
        self.last_sequencing_action: Optional[int] = None
        
        # 初始化网络
        self._initialize_networks()
        
        # 加载模型（如果有）
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            LOGGER.info(f"[DualDRLAgent] 加载模型成功：{model_path}")
    
    def _initialize_networks(self):
        """初始化 Routing 和 Sequencing 网络"""
        self.routing_network = RoutingNetwork(
            input_dim=self.routing_input_dim,
            output_dim=self.routing_output_dim,
            learning_rate=self.learning_rate
        )
        
        self.sequencing_network = SequencingNetwork(
            input_dim_path1=self.seq_path1_dim,
            input_dim_path2=self.seq_path2_dim,
            output_dim=self.seq_output_dim,
            learning_rate=self.learning_rate
        )
        LOGGER.info("[DualDRLAgent] 网络初始化完成")
    
    def _extract_routing_state(self, machines: List[Machine], jobs: List[Job], 
                               current_time: float) -> np.ndarray:
        """
        提取 Routing Agent 状态
        
        状态组成:
        1. 机器信息: Available time, Sum of processing times
        2. 待调度作业信息: Processing time
        3. 新到达作业信息: Time till imminent arrival
        
        :param machines: 机器列表
        :param jobs: 作业列表
        :param current_time: 当前时间
        :return: 状态向量
        """
        state_features = []
        
        # 1. 机器信息
        for machine in machines[:10]:  # 限制最多10台机器
            # Available time
            am_k = machine.timer if machine.status == MachineStatus.WORKING else current_time
            state_features.append(am_k / 1000.0)  # 归一化
            
            # Sum of processing times (负载量)
            load_sum = sum(
                op.get_duration(machine.id) - op.process_time 
                for op in machine.input_queue 
                if op.get_status() in [OperationStatus.READY, OperationStatus.WAITING]
            )
            state_features.append(load_sum / 100.0)  # 归一化
        
        # 填充到固定长度
        while len(state_features) < 20:
            state_features.extend([0.0, 0.0])
        
        # 2. 待调度作业的处理时间（取第一个就绪作业）
        ready_op = None
        for job in jobs:
            if job.is_finished():
                continue
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() == OperationStatus.READY:
                    ready_op = op
                    break
            if ready_op:
                break
        
        if ready_op and machines:
            # 在各可选机器上的处理时间
            for machine in machines[:5]:  # 限制最多5台
                if ready_op.is_machine_capable(machine.id):
                    proc_time = ready_op.get_duration(machine.id)
                    state_features.append(proc_time / 100.0)
                else:
                    state_features.append(0.0)
        else:
            state_features.extend([0.0] * 5)
        
        # 3. 新到达作业信息（简化处理，假设所有作业都已到达）
        # 实际应用中需要根据具体的 Job 类结构调整
        state_features.append(1.0)  # 默认无未来作业
        
        # 填充到固定维度
        state_array = np.array(state_features[:self.routing_input_dim], dtype=np.float32)
        if len(state_array) < self.routing_input_dim:
            state_array = np.pad(state_array, (0, self.routing_input_dim - len(state_array)))
        
        return state_array
    
    def _extract_sequencing_state(self, machines: List[Machine], jobs: List[Job],
                                  current_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取 Sequencing Agent 状态（双路径）
        
        路径1 (16维): 作业数量、紧邻操作处理时间统计
        路径2 (8维): 剩余处理时间、系统级信息、异质性指标
        
        :param machines: 机器列表
        :param jobs: 作业列表
        :param current_time: 当前时间
        :return: (path1_state, path2_state)
        """
        # ===== 路径1: 作业数量和紧邻操作处理时间 =====
        path1_features = []
        
        # 1. Jobs in system
        jobs_in_system = sum(1 for j in jobs if not j.is_finished())
        path1_features.append(jobs_in_system / 100.0)
        
        # 2. Jobs in queue (平均队列长度)
        if machines:
            avg_queue_len = np.mean([len(m.input_queue) for m in machines])
            path1_features.append(avg_queue_len / 20.0)
        else:
            path1_features.append(0.0)
        
        # 3. Expected arrivals（简化为0）
        path1_features.append(0.0)
        
        # 4-7. 紧邻操作处理时间统计 (Sum, Mean, Min, Total)
        imminent_times = []
        for machine in machines:
            if machine.input_queue:
                next_op = machine.input_queue[0]
                time_left = next_op.get_duration(machine.id) - next_op.process_time
                imminent_times.append(time_left)
        
        if imminent_times:
            path1_features.append(sum(imminent_times) / 1000.0)  # Sum
            path1_features.append(float(np.mean(imminent_times)) / 100.0)  # Mean
            path1_features.append(min(imminent_times) / 100.0)  # Min
            path1_features.append(sum(imminent_times) / 1000.0)  # Total (重复)
        else:
            path1_features.extend([0.0] * 4)
        
        # 填充到16维
        while len(path1_features) < 16:
            path1_features.append(0.0)
        
        path1_state = np.array(path1_features[:16], dtype=np.float32)
        
        # ===== 路径2: 剩余处理时间、系统信息、异质性 =====
        path2_features = []
        
        # 1-3. 剩余处理时间统计 (Avg, Max, Total)
        remaining_times = []
        for job in jobs:
            if job.is_finished():
                continue
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() in [OperationStatus.READY, OperationStatus.WAITING]:
                    # 简化的剩余时间计算（使用第一个可选机器的处理时间）
                    if op.durations:
                        rpt = op.durations[0][1] - op.process_time
                        remaining_times.append(rpt)
        
        if remaining_times:
            path2_features.append(float(np.mean(remaining_times)) / 100.0)  # Avg RPT
            path2_features.append(max(remaining_times) / 100.0)  # Max RPT
            path2_features.append(sum(remaining_times) / 1000.0)  # Total RPT
        else:
            path2_features.extend([0.0] * 3)
        
        # 4-5. 后续工位可用时间 (Avg, Min ASW)
        asw_times = [m.timer for m in machines if m.status == MachineStatus.WORKING]
        if asw_times:
            path2_features.append(float(np.mean(asw_times)) / 1000.0)  # Avg ASW
            path2_features.append(min(asw_times) / 1000.0)  # Min ASW
        else:
            path2_features.extend([0.0] * 2)
        
        # 6. Completion rate
        completed_jobs = sum(1 for j in jobs if j.is_finished())
        total_jobs = len(jobs) if jobs else 1
        completion_rate = completed_jobs / total_jobs
        path2_features.append(completion_rate)
        
        # 7-8. 异质性指标 (CV of processing time, CV of RPT)
        if remaining_times and len(remaining_times) > 1:
            cv_rpt = float(np.std(remaining_times)) / (float(np.mean(remaining_times)) + 1e-5)
            path2_features.append(cv_rpt)
        else:
            path2_features.append(0.0)
        
        if asw_times and len(asw_times) > 1:
            cv_asw = float(np.std(asw_times)) / (float(np.mean(asw_times)) + 1e-5)
            path2_features.append(cv_asw)
        else:
            path2_features.append(0.0)
        
        # 填充到8维
        while len(path2_features) < 8:
            path2_features.append(0.0)
        
        path2_state = np.array(path2_features[:8], dtype=np.float32)
        
        return path1_state, path2_state
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """将状态向量转换为字符串键（用于 Q-table）"""
        # 离散化状态
        discretized = np.round(state * 10).astype(int)
        return '_'.join(map(str, discretized))
    
    def _select_routing_action(self, state: np.ndarray, valid_machines: List[Machine]) -> int:
        """
        Routing Agent 选择动作（机器分配）
        
        :param state: 状态向量
        :param valid_machines: 可选机器列表
        :return: 选中的机器索引
        """
        if not valid_machines:
            return -1
        
        # ε-greedy 策略
        if random.random() < self.epsilon:
            # 探索：随机选择
            action_idx = random.randint(0, len(valid_machines) - 1)
        else:
            # 利用：选择 Q 值最大的动作
            q_values = []
            if self.routing_network is not None:
                for i, machine in enumerate(valid_machines):
                    state_key = self._state_to_key(state)
                    q_val = self.routing_network.get_q_value(state_key, i)
                    q_values.append(q_val)
            
            action_idx = int(np.argmax(q_values)) if q_values else 0
        
        return action_idx
    
    def _select_sequencing_action(self, path1_state: np.ndarray, path2_state: np.ndarray,
                                  ready_operations: List[Tuple[int, int]]) -> int:
        """
        Sequencing Agent 选择动作（作业选择）
        
        :param path1_state: 路径1状态
        :param path2_state: 路径2状态
        :param ready_operations: 就绪操作列表 [(job_idx, op_idx), ...]
        :return: 选中的操作索引
        """
        if not ready_operations:
            return -1
        
        # ε-greedy 策略
        if random.random() < self.epsilon:
            # 探索：随机选择
            action_idx = random.randint(0, len(ready_operations) - 1)
        else:
            # 利用：选择 Q 值最大的动作
            q_values = []
            if self.sequencing_network is not None:
                for i in range(len(ready_operations)):
                    state_key = self._state_to_key(np.concatenate([path1_state, path2_state]))
                    q_val = self.sequencing_network.get_q_value(state_key, i)
                    q_values.append(q_val)
            
            action_idx = int(np.argmax(q_values)) if q_values else 0
        
        return action_idx
    
    def _calculate_routing_reward(self, machines: List[Machine], makespan: float) -> float:
        """
        计算 Routing Agent 奖励
        
        奖励组成:
        - 负载均衡奖励: -std(machine_loads)
        - 完工时间惩罚: -makespan
        - 机器利用率奖励: +avg_utilization
        
        :param machines: 机器列表
        :param makespan: 完工时间
        :return: 奖励值
        """
        if not machines:
            return 0.0
        
        # 计算机器负载
        loads = [len(m.input_queue) for m in machines]
        load_std = float(np.std(loads)) if loads else 0
        
        # 计算机器利用率
        utilization = sum(1 for m in machines if m.status == MachineStatus.WORKING) / len(machines)
        
        # 组合奖励
        reward = -load_std * 0.3 - makespan * 0.001 + utilization * 0.5
        
        return float(reward)
    
    def _calculate_sequencing_reward(self, prev_queue_len: int, curr_queue_len: int,
                                     urgency_score: float = 0.0) -> float:
        """
        计算 Sequencing Agent 奖励
        
        奖励组成:
        - 队列长度减少: -(curr - prev)
        - 紧急度奖励: +urgency_score
        - 瓶颈缓解: +bottleneck_relief
        
        :param prev_queue_len: 之前的队列长度
        :param curr_queue_len: 当前队列长度
        :param urgency_score: 紧急度分数
        :return: 奖励值
        """
        queue_reduction = -(curr_queue_len - prev_queue_len)
        bottleneck_relief = 0.1 if curr_queue_len < prev_queue_len else 0.0
        
        reward = queue_reduction * 0.5 + urgency_score * 0.3 + bottleneck_relief
        
        return float(reward)
    
    def _store_experience(self, experience: Dict):
        """存储经验"""
        self.replay_buffer.append(experience)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def _train_from_replay(self):
        """从回放缓冲区训练"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 随机采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # 简化版训练：直接更新 Q 值
        for exp in batch:
            state_key = exp['state_key']
            action = exp['action']
            reward = exp['reward']
            next_state_key = exp['next_state_key']
            done = exp['done']
            
            if self.routing_network is None:
                continue
            
            # Q-learning 更新
            current_q = self.routing_network.get_q_value(state_key, action)
            
            if done:
                target_q = reward
            else:
                # 简化：假设下一个状态的最大 Q 值为 0
                max_next_q = 0
                target_q = reward + self.discount_factor * max_next_q
            
            # 更新 Q 值
            new_q = current_q + self.learning_rate * (target_q - current_q)
            self.routing_network.update_q_value(state_key, action, new_q)
    
    def reward(self, *args, **kwargs) -> float:
        """
        计算奖励
        
        :param args: (env_info,)
        :param kwargs: 额外参数
        :return: 奖励值
        """
        if not args:
            return 0.0
        
        env_info = args[0]
        makespan = env_info.get('makespan', 0)
        machines = env_info.get('machines', [])
        
        routing_reward = self._calculate_routing_reward(machines, makespan)
        self.current_episode_reward += routing_reward
        
        return routing_reward
    
    def train(self, *args, **kwargs):
        """
        训练 Agent
        
        :param args: (observations, rewards, terminations, truncations, infos)
        :param kwargs: 额外参数
        """
        if self.mode != TRAINING:
            return
        
        observations, rewards, terminations, truncations, infos = args
        
        # 存储经验
        if self.last_routing_state is not None and self.last_routing_action is not None:
            experience = {
                'state_key': self._state_to_key(self.last_routing_state),
                'action': self.last_routing_action,
                'reward': rewards if isinstance(rewards, (int, float)) else 0,
                'next_state_key': self._state_to_key(observations) if observations is not None else '',
                'done': terminations.get(self.agent_id, False) if isinstance(terminations, dict) else False
            }
            self._store_experience(experience)
        
        # 从回放缓冲区训练
        self._train_from_replay()
        
        # 衰减 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def sample(self, agvs: List[AGV], machines: List[Machine], jobs: List[Job]) -> Tuple[List, float]:
        """
        决策采样
        
        :param agvs: AGV 列表
        :param machines: 机器列表
        :param jobs: 作业列表
        :return: (decisions, step_time)
        """
        time_start = time.time()
        decisions = []
        
        # 获取当前时间
        current_time = 0  # 简化，实际应从环境获取
        
        # 提取 Routing Agent 状态
        routing_state = self._extract_routing_state(machines, jobs, current_time)
        self.last_routing_state = routing_state
        
        # 提取 Sequencing Agent 状态
        seq_path1, seq_path2 = self._extract_sequencing_state(machines, jobs, current_time)
        self.last_sequencing_state_path1 = seq_path1
        self.last_sequencing_state_path2 = seq_path2
        
        # 收集所有就绪操作
        ready_operations = []
        for job_idx, job in enumerate(jobs):
            if job.is_finished():
                continue
            for op_idx in range(job.get_operation_count()):
                op = job.get_operation(op_idx)
                if op.get_status() == OperationStatus.READY:
                    ready_operations.append((job_idx, op_idx, op))
        
        if not ready_operations or not machines:
            return [], DEFAULT_STEP_TIME
        
        # Routing Agent: 为每个就绪操作分配机器
        for job_idx, op_idx, op in ready_operations:
            # 找到可选机器
            valid_machines = [m for m in machines if op.is_machine_capable(m.id)]
            if not valid_machines:
                continue
            
            # Routing Agent 选择机器
            machine_action = self._select_routing_action(routing_state, valid_machines)
            selected_machine = valid_machines[machine_action] if machine_action >= 0 else None
            
            if not selected_machine:
                continue
            
            self.last_routing_action = machine_action
            
            # Sequencing Agent: 从队列中选择（这里简化为直接执行）
            # 实际应用中，Sequencing Agent 会在机器队列有多个作业时决定执行顺序
            
            # 选择 AGV（使用 AVAILABLE 或 READY 状态）
            available_agvs = [agv for agv in agvs if hasattr(agv, 'status') and 
                             agv.status in [AGVStatus.READY, AGVStatus.ASSIGNED]]
            selected_agv = random.choice(available_agvs) if available_agvs else None
            
            if selected_agv:
                decisions.append((op, selected_agv, selected_machine))
        
        time_end = time.time()
        step_time = max(time_end - time_start, DEFAULT_STEP_TIME)
        
        return decisions, step_time
    
    def is_finish(self) -> bool:
        """判断任务是否完成"""
        return not self.alive
    
    def save_model(self, path: Optional[str] = None):
        """
        保存模型
        
        :param path: 保存路径
        """
        if path is None:
            path = self.model_path
        
        if path is None:
            LOGGER.warning("[DualDRLAgent] 未指定模型保存路径")
            return
        
        # 创建目录
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # 保存 Routing Network
        if self.routing_network is not None:
            routing_path = path.replace('.json', '_routing.json')
            self.routing_network.save(routing_path)
        
        # 保存 Sequencing Network
        if self.sequencing_network is not None:
            sequencing_path = path.replace('.json', '_sequencing.json')
            self.sequencing_network.save(sequencing_path)
        
        # 保存元数据
        metadata = {
            'mode': self.mode,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'episode_rewards': self.episode_rewards,
            'training_history': self.training_history,
            'routing_input_dim': self.routing_input_dim,
            'routing_output_dim': self.routing_output_dim,
            'seq_path1_dim': self.seq_path1_dim,
            'seq_path2_dim': self.seq_path2_dim,
            'seq_output_dim': self.seq_output_dim
        }
        metadata_path = path.replace('.json', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        LOGGER.info(f"[DualDRLAgent] 模型已保存至: {path}")
    
    def load_model(self, path: str):
        """
        加载模型
        
        :param path: 模型路径
        """
        if not os.path.exists(path):
            LOGGER.warning(f"[DualDRLAgent] 模型文件不存在: {path}")
            return
        
        # 加载 Routing Network
        routing_path = path.replace('.json', '_routing.json')
        if os.path.exists(routing_path) and self.routing_network is not None:
            self.routing_network.load(routing_path)
        
        # 加载 Sequencing Network
        sequencing_path = path.replace('.json', '_sequencing.json')
        if os.path.exists(sequencing_path) and self.sequencing_network is not None:
            self.sequencing_network.load(sequencing_path)
        
        # 加载元数据
        metadata_path = path.replace('.json', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.epsilon = metadata.get('epsilon', self.epsilon)
            self.learning_rate = metadata.get('learning_rate', self.learning_rate)
            self.discount_factor = metadata.get('discount_factor', self.discount_factor)
            self.episode_rewards = metadata.get('episode_rewards', [])
            self.training_history = metadata.get('training_history', self.training_history)
        
        LOGGER.info(f"[DualDRLAgent] 模型已从 {path} 加载")
