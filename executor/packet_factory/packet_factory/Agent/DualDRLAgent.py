"""
双层 DRL Agent - Routing Agent + Sequencing Agent (PyTorch 实现)

Routing Agent: 负责为作业分配机器（路由决策）
Sequencing Agent: 负责从队列中选择作业执行（排序决策）

网络结构:
- Routing Agent: Input -> InstanceNorm -> FC(16,tanh) x3 -> FC(8,tanh) x2 -> Output
- Sequencing Agent: 双路径输入 -> InstanceNorm -> Concat -> FC(48,36,36,24,24,12,tanh) -> Output
"""

from .BaseAgent import BaseAgent, DEFAULT_STEP_TIME, FRONTEND, BACKEND, TRAINING, INFERENCE
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

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# PyTorch 神经网络模块
# ===========================================================================

class RoutingNetwork(nn.Module):
    """
    Routing Agent 神经网络 (DQN)

    结构: Input -> InstanceNorm -> FC(16,tanh) x3 -> FC(8,tanh) x2 -> Output
    输出为各动作的 Q 值
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 实例归一化 (处理 1D 单样本输入)
        self.instance_norm = nn.InstanceNorm1d(1, affine=False)

        # FC(16, tanh) x3
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)

        # FC(8, tanh) x2
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 8)

        # 输出层
        self.fc_out = nn.Linear(8, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回各动作的 Q 值"""
        # x: (batch, input_dim) -> InstanceNorm 需要 (batch, channels, length)
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        x = self.instance_norm(x)
        x = x.squeeze(1)  # (batch, input_dim)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))

        return self.fc_out(x)


class SequencingNetwork(nn.Module):
    """
    Sequencing Agent 神经网络 (DQN)

    结构: 双路径输入 -> InstanceNorm -> Concat -> FC(48,36,36,24,24,12,tanh) -> Output
    输出为各动作的 Q 值
    """

    def __init__(self, input_dim_path1: int, input_dim_path2: int, output_dim: int):
        super().__init__()
        self.input_dim_path1 = input_dim_path1
        self.input_dim_path2 = input_dim_path2
        self.output_dim = output_dim

        # 双路径实例归一化
        self.norm1 = nn.InstanceNorm1d(1, affine=False)
        self.norm2 = nn.InstanceNorm1d(1, affine=False)

        concat_dim = input_dim_path1 + input_dim_path2

        # 隐藏层: FC(48, 36, 36, 24, 24, 12)
        self.fc1 = nn.Linear(concat_dim, 48)
        self.fc2 = nn.Linear(48, 36)
        self.fc3 = nn.Linear(36, 36)
        self.fc4 = nn.Linear(36, 24)
        self.fc5 = nn.Linear(24, 24)
        self.fc6 = nn.Linear(24, 12)

        # 输出层
        self.fc_out = nn.Linear(12, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, path1: torch.Tensor, path2: torch.Tensor) -> torch.Tensor:
        """前向传播，返回各动作的 Q 值"""
        # InstanceNorm
        p1 = path1.unsqueeze(1)  # (batch, 1, dim)
        p1 = self.norm1(p1).squeeze(1)

        p2 = path2.unsqueeze(1)
        p2 = self.norm2(p2).squeeze(1)

        # 拼接
        x = torch.cat([p1, p2], dim=-1)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))

        return self.fc_out(x)


# ===========================================================================
# 经验回放缓冲区
# ===========================================================================

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer: List[Dict[str, np.ndarray]] = []

    def push(self, experience: Dict[str, np.ndarray]):
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        return {
            key: torch.tensor(np.array([exp[key] for exp in batch]),
                              dtype=torch.float32, device=self.device)
            for key in batch[0]
        }

    def __len__(self):
        return len(self.buffer)


# ===========================================================================
# DualDRLAgent
# ===========================================================================

@register_component("packet_factory.DualDRLAgent")
class DualDRLAgent(BaseAgent):
    """
    双层 DRL Agent (PyTorch DQN 实现)

    包含 Routing Agent（路由决策）和 Sequencing Agent（排序决策）
    """

    def __init__(self, name=None, agent_id=None, context=None,
                 ui_mode: str = BACKEND, task_mode: str = TRAINING,
                 model_path: Optional[str] = None, **kwargs):
        super().__init__(name, agent_id, context, ui_mode, task_mode, model_path)

        # ========== 设备 ==========
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ========== 网络配置 ==========
        self.routing_input_dim = 20
        self.routing_output_dim = 50
        self.seq_path1_dim = 16
        self.seq_path2_dim = 8
        self.seq_output_dim = 100

        # ========== 强化学习参数 ==========
        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01
        self.target_update_freq = 100  # 目标网络更新频率

        # ========== 初始化网络 ==========
        self._initialize_networks()

        # ========== 经验回放 ==========
        self.buffer_size = 10000
        self.batch_size = 64
        self.routing_replay = ReplayBuffer(self.buffer_size, self.device)
        self.sequencing_replay = ReplayBuffer(self.buffer_size, self.device)

        # ========== 训练统计 ==========
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0
        self.training_step = 0
        self.training_history: Dict[str, List] = {
            'episodes': [],
            'routing_rewards': [],
            'sequencing_rewards': [],
            'total_rewards': [],
            'makespans': [],
            'epsilon': [],
            'routing_loss': [],
            'sequencing_loss': [],
        }

        # 最近的状态和动作
        self.last_routing_state: Optional[np.ndarray] = None
        self.last_routing_action: Optional[int] = None
        self.last_sequencing_state_path1: Optional[np.ndarray] = None
        self.last_sequencing_state_path2: Optional[np.ndarray] = None
        self.last_sequencing_action: Optional[int] = None

        # 加载模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            LOGGER.info(f"[DualDRLAgent] 加载模型成功：{model_path}")
        else:
            LOGGER.info(f"[DualDRLAgent] 模型{model_path}未找到，将使用随机初始化网络")

    def _initialize_networks(self):
        """初始化 DQN 网络和目标网络"""
        # Routing 网络
        self.routing_net = RoutingNetwork(
            self.routing_input_dim, self.routing_output_dim
        ).to(self.device)
        self.routing_target = RoutingNetwork(
            self.routing_input_dim, self.routing_output_dim
        ).to(self.device)
        self.routing_target.load_state_dict(self.routing_net.state_dict())
        self.routing_target.eval()
        self.routing_optimizer = torch.optim.Adam(
            self.routing_net.parameters(), lr=self.learning_rate
        )

        # Sequencing 网络
        self.seq_net = SequencingNetwork(
            self.seq_path1_dim, self.seq_path2_dim, self.seq_output_dim
        ).to(self.device)
        self.seq_target = SequencingNetwork(
            self.seq_path1_dim, self.seq_path2_dim, self.seq_output_dim
        ).to(self.device)
        self.seq_target.load_state_dict(self.seq_net.state_dict())
        self.seq_target.eval()
        self.seq_optimizer = torch.optim.Adam(
            self.seq_net.parameters(), lr=self.learning_rate
        )

        LOGGER.info(f"[DualDRLAgent] 网络初始化完成 (device={self.device})")

    # ------------------------------------------------------------------
    # 状态提取
    # ------------------------------------------------------------------

    def _extract_routing_state(self, machines: List[Machine], jobs: List[Job],
                               current_time: float) -> np.ndarray:
        """
        提取 Routing Agent 状态

        状态组成:
        1. 机器信息: Available time, Sum of processing times
        2. 待调度作业信息: Processing time
        3. 新到达作业信息: Time till imminent arrival
        """
        state_features = []

        # 1. 机器信息
        for machine in machines[:10]:
            am_k = machine.timer if machine.status == MachineStatus.WORKING else current_time
            state_features.append(am_k / 1000.0)

            load_sum = sum(
                op.get_duration(machine.id) - op.process_time
                for op in machine.input_queue
                if op.get_status() in [OperationStatus.READY, OperationStatus.WAITING]
            )
            state_features.append(load_sum / 100.0)

        while len(state_features) < 20:
            state_features.extend([0.0, 0.0])

        # 2. 待调度作业处理时间
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
            for machine in machines[:5]:
                if ready_op.is_machine_capable(machine.id):
                    proc_time = ready_op.get_duration(machine.id)
                    state_features.append(proc_time / 100.0)
                else:
                    state_features.append(0.0)
        else:
            state_features.extend([0.0] * 5)

        state_features.append(1.0)

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
        """
        # ===== 路径1 =====
        path1_features = []

        jobs_in_system = sum(1 for j in jobs if not j.is_finished())
        path1_features.append(jobs_in_system / 100.0)

        if machines:
            avg_queue_len = np.mean([len(m.input_queue) for m in machines])
            path1_features.append(avg_queue_len / 20.0)
        else:
            path1_features.append(0.0)

        path1_features.append(0.0)  # Expected arrivals

        imminent_times = []
        for machine in machines:
            if machine.input_queue:
                next_op = machine.input_queue[0]
                time_left = next_op.get_duration(machine.id) - next_op.process_time
                imminent_times.append(time_left)

        if imminent_times:
            path1_features.append(sum(imminent_times) / 1000.0)
            path1_features.append(float(np.mean(imminent_times)) / 100.0)
            path1_features.append(min(imminent_times) / 100.0)
            path1_features.append(sum(imminent_times) / 1000.0)
        else:
            path1_features.extend([0.0] * 4)

        while len(path1_features) < 16:
            path1_features.append(0.0)

        path1_state = np.array(path1_features[:16], dtype=np.float32)

        # ===== 路径2 =====
        path2_features = []

        remaining_times = []
        for job in jobs:
            if job.is_finished():
                continue
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() in [OperationStatus.READY, OperationStatus.WAITING]:
                    if op.durations:
                        rpt = op.durations[0][1] - op.process_time
                        remaining_times.append(rpt)

        if remaining_times:
            path2_features.append(float(np.mean(remaining_times)) / 100.0)
            path2_features.append(max(remaining_times) / 100.0)
            path2_features.append(sum(remaining_times) / 1000.0)
        else:
            path2_features.extend([0.0] * 3)

        asw_times = [m.timer for m in machines if m.status == MachineStatus.WORKING]
        if asw_times:
            path2_features.append(float(np.mean(asw_times)) / 1000.0)
            path2_features.append(min(asw_times) / 1000.0)
        else:
            path2_features.extend([0.0] * 2)

        completed_jobs = sum(1 for j in jobs if j.is_finished())
        total_jobs = len(jobs) if jobs else 1
        path2_features.append(completed_jobs / total_jobs)

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

        while len(path2_features) < 8:
            path2_features.append(0.0)

        path2_state = np.array(path2_features[:8], dtype=np.float32)

        return path1_state, path2_state

    # ------------------------------------------------------------------
    # 动作选择 (ε-greedy)
    # ------------------------------------------------------------------

    def _select_routing_action(self, state: np.ndarray, valid_machines: List[Machine]) -> int:
        """Routing Agent 选择动作（机器分配），ε-greedy"""
        if not valid_machines:
            return -1

        if random.random() < self.epsilon:
            return random.randint(0, len(valid_machines) - 1)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.routing_net(state_t).squeeze(0).cpu().numpy()

        # 只在可选机器范围内选取最大 Q 值
        valid_q = q_values[:len(valid_machines)]
        return int(np.argmax(valid_q))

    def _select_sequencing_action(self, path1_state: np.ndarray, path2_state: np.ndarray,
                                  ready_operations: List[Tuple[int, int]]) -> int:
        """Sequencing Agent 选择动作（作业选择），ε-greedy"""
        if not ready_operations:
            return -1

        if random.random() < self.epsilon:
            return random.randint(0, len(ready_operations) - 1)

        with torch.no_grad():
            p1_t = torch.tensor(path1_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            p2_t = torch.tensor(path2_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.seq_net(p1_t, p2_t).squeeze(0).cpu().numpy()

        valid_q = q_values[:len(ready_operations)]
        return int(np.argmax(valid_q))

    # ------------------------------------------------------------------
    # 奖励计算
    # ------------------------------------------------------------------

    def _calculate_routing_reward(self, machines: List[Machine], makespan: float) -> float:
        """计算 Routing Agent 奖励"""
        if not machines:
            return 0.0

        loads = [len(m.input_queue) for m in machines]
        load_std = float(np.std(loads)) if loads else 0
        utilization = sum(1 for m in machines if m.status == MachineStatus.WORKING) / len(machines)

        reward = -load_std * 0.3 - makespan * 0.001 + utilization * 0.5
        return float(reward)

    def _calculate_sequencing_reward(self, prev_queue_len: int, curr_queue_len: int,
                                     urgency_score: float = 0.0) -> float:
        """计算 Sequencing Agent 奖励"""
        queue_reduction = -(curr_queue_len - prev_queue_len)
        bottleneck_relief = 0.1 if curr_queue_len < prev_queue_len else 0.0
        reward = queue_reduction * 0.5 + urgency_score * 0.3 + bottleneck_relief
        return float(reward)

    # ------------------------------------------------------------------
    # DQN 训练
    # ------------------------------------------------------------------

    def _train_routing_dqn(self):
        """从回放缓冲区训练 Routing DQN"""
        batch = self.routing_replay.sample(self.batch_size)
        if batch is None:
            return

        state = batch['state']
        action = batch['action'].long()  # (batch, 1)
        reward = batch['reward'].squeeze(-1)  # (batch,)
        next_state = batch['next_state']
        done = batch['done'].squeeze(-1)  # (batch,)

        # 当前 Q 值: Q(s, a)
        q_values = self.routing_net(state)  # (batch, output_dim)
        q_value = q_values.gather(1, action).squeeze(1)  # (batch,)

        # 目标 Q 值: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.routing_target(next_state)
            max_next_q = next_q.max(1)[0]
            target_q = reward + self.discount_factor * max_next_q * (1 - done)

        loss = F.mse_loss(q_value, target_q)

        self.routing_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.routing_net.parameters(), max_norm=1.0)
        self.routing_optimizer.step()

        self.training_history.setdefault('routing_loss', []).append(loss.item())

    def _train_sequencing_dqn(self):
        """从回放缓冲区训练 Sequencing DQN"""
        batch = self.sequencing_replay.sample(self.batch_size)
        if batch is None:
            return

        path1 = batch['path1']
        path2 = batch['path2']
        action = batch['action'].long()  # (batch, 1)
        reward = batch['reward'].squeeze(-1)  # (batch,)
        next_path1 = batch['next_path1']
        next_path2 = batch['next_path2']
        done = batch['done'].squeeze(-1)  # (batch,)

        # 当前 Q 值
        q_values = self.seq_net(path1, path2)  # (batch, output_dim)
        q_value = q_values.gather(1, action).squeeze(1)  # (batch,)

        # 目标 Q 值
        with torch.no_grad():
            next_q = self.seq_target(next_path1, next_path2)
            max_next_q = next_q.max(1)[0]
            target_q = reward + self.discount_factor * max_next_q * (1 - done)

        loss = F.mse_loss(q_value, target_q)

        self.seq_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.seq_net.parameters(), max_norm=1.0)
        self.seq_optimizer.step()

        self.training_history.setdefault('sequencing_loss', []).append(loss.item())

    def _update_target_networks(self):
        """硬更新目标网络"""
        self.routing_target.load_state_dict(self.routing_net.state_dict())
        self.seq_target.load_state_dict(self.seq_net.state_dict())

    # ------------------------------------------------------------------
    # Agent 接口实现
    # ------------------------------------------------------------------

    def reward(self, *args, **kwargs) -> float:
        """计算奖励"""
        if not args:
            return 0.0

        env_info = args[0]
        makespan = env_info.get('makespan', 0)
        machines = env_info.get('machines', [])
        jobs = env_info.get('jobs', [])

        routing_reward = self._calculate_routing_reward(machines, makespan)

        # 计算 Sequencing 奖励
        total_queue_len = sum(len(m.input_queue) for m in machines) if machines else 0
        if not hasattr(self, '_prev_queue_len'):
            self._prev_queue_len = total_queue_len
        sequencing_reward = self._calculate_sequencing_reward(
            self._prev_queue_len, total_queue_len
        )
        self._prev_queue_len = total_queue_len

        total_reward = routing_reward + sequencing_reward
        self.current_episode_reward += total_reward

        return total_reward

    def new_episode(self):
        """Reset episode-level state. Call at the start of each episode."""
        self.current_episode_reward = 0.0
        self._prev_queue_len = 0

        if self.training_history['episodes']:
            last_ep = self.training_history['episodes'][-1] + 1
        else:
            last_ep = 1
        self.training_history['episodes'].append(last_ep)
        self.training_history['epsilon'].append(self.epsilon)

        if self.episode_rewards:
            self.training_history['total_rewards'].append(self.episode_rewards[-1])
        self.episode_rewards = []

    def _extract_reward_from_env(self, rewards) -> float:
        """从 env.step() 返回的 rewards 中提取标量奖励值"""
        if isinstance(rewards, (int, float)):
            return float(rewards)
        if isinstance(rewards, dict):
            # 取第一个值
            for v in rewards.values():
                if isinstance(v, (int, float)):
                    return float(v)
        return 0.0

    def train(self, *args, **kwargs):
        """训练 Agent"""
        if self.mode != TRAINING:
            return

        observations, rewards, terminations, truncations, infos = args

        # 提取标量奖励
        reward_scalar = self._extract_reward_from_env(rewards)

        # 存储 Routing 经验
        if self.last_routing_state is not None and self.last_routing_action is not None:
            next_state = observations if isinstance(observations, np.ndarray) else np.zeros(self.routing_input_dim, dtype=np.float32)
            done = terminations.get(self.agent_id, False) if isinstance(terminations, dict) else False
            self.routing_replay.push({
                'state': self.last_routing_state,
                'action': np.array([self.last_routing_action], dtype=np.float32),
                'reward': np.array([reward_scalar], dtype=np.float32),
                'next_state': next_state[:self.routing_input_dim] if len(next_state) >= self.routing_input_dim else np.pad(next_state, (0, self.routing_input_dim - len(next_state))),
                'done': np.array([float(done)], dtype=np.float32),
            })

        # 存储 Sequencing 经验
        if (self.last_sequencing_state_path1 is not None
                and self.last_sequencing_state_path2 is not None
                and self.last_sequencing_action is not None):
            done = terminations.get(self.agent_id, False) if isinstance(terminations, dict) else False
            self.sequencing_replay.push({
                'path1': self.last_sequencing_state_path1,
                'path2': self.last_sequencing_state_path2,
                'action': np.array([self.last_sequencing_action], dtype=np.float32),
                'reward': np.array([reward_scalar], dtype=np.float32),
                'next_path1': self.last_sequencing_state_path1,
                'next_path2': self.last_sequencing_state_path2,
                'done': np.array([float(done)], dtype=np.float32),
            })

        # 训练
        self._train_routing_dqn()
        self._train_sequencing_dqn()

        # 主动释放 GIL，防止 CPU 密集训练阻塞 asyncio 事件循环
        import time
        time.sleep(0)

        # 更新目标网络
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self._update_target_networks()

        # 衰减 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def sample(self, agvs: List[AGV], machines: List[Machine], jobs: List[Job]) -> Tuple[List, float]:
        """决策采样"""
        time_start = time.time()
        decisions = []

        current_time = 0

        # 提取状态
        routing_state = self._extract_routing_state(machines, jobs, current_time)
        self.last_routing_state = routing_state

        seq_path1, seq_path2 = self._extract_sequencing_state(machines, jobs, current_time)
        self.last_sequencing_state_path1 = seq_path1
        self.last_sequencing_state_path2 = seq_path2

        # 收集就绪操作
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
            valid_machines = [m for m in machines if op.is_machine_capable(m.id)]
            if not valid_machines:
                continue

            machine_action = self._select_routing_action(routing_state, valid_machines)
            selected_machine = valid_machines[machine_action] if machine_action >= 0 else None

            if not selected_machine:
                continue

            self.last_routing_action = machine_action

            selected_agv = random.choice(agvs) if agvs else None

            if selected_agv:
                decisions.append((op, selected_agv, selected_machine))

        time_end = time.time()
        step_time = max(time_end - time_start, DEFAULT_STEP_TIME)

        return decisions, step_time

    def get_training_metrics(self) -> Dict[str, Any]:
        """Return training metrics for convergence detection."""
        metrics = {
            'episode_reward': self.current_episode_reward,
            'epsilon': self.epsilon,
            'training_history': self.training_history,
        }
        # 最近 loss
        if 'routing_loss' in self.training_history and self.training_history['routing_loss']:
            metrics['routing_loss'] = self.training_history['routing_loss'][-1]
        if 'sequencing_loss' in self.training_history and self.training_history['sequencing_loss']:
            metrics['sequencing_loss'] = self.training_history['sequencing_loss'][-1]
        return metrics

    def is_finish(self) -> bool:
        """判断任务是否完成"""
        return not self.alive

    # ------------------------------------------------------------------
    # 模型保存 / 加载
    # ------------------------------------------------------------------

    def save_model(self, path: Optional[str] = None):
        """保存模型"""
        if path is None:
            agent_name = self.name or "DualDRLAgent"
            agent_dir = f"training_logs/models/{agent_name}"
            default_path = f"{agent_dir}/agent_model.pt"

            os.makedirs(agent_dir, exist_ok=True)

            if os.path.exists(default_path):
                file_size = os.path.getsize(default_path)
                if file_size == 0:
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    path = f"{agent_dir}/agent_model_{timestamp}.pt"
                    LOGGER.info(f"[DualDRLAgent] 检测到默认模型文件为空，创建新文件：{path}")
                else:
                    path = default_path
                    LOGGER.info(f"[DualDRLAgent] 覆盖现有模型文件：{path}")
            else:
                path = default_path
                LOGGER.info(f"[DualDRLAgent] 创建新模型文件：{path}")
        else:
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

        checkpoint = {
            'routing_net': self.routing_net.state_dict(),
            'routing_target': self.routing_target.state_dict(),
            'seq_net': self.seq_net.state_dict(),
            'seq_target': self.seq_target.state_dict(),
            'routing_optimizer': self.routing_optimizer.state_dict(),
            'seq_optimizer': self.seq_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'training_step': self.training_step,
            'training_history': self.training_history,
            'episode_rewards': self.episode_rewards,
            'routing_input_dim': self.routing_input_dim,
            'routing_output_dim': self.routing_output_dim,
            'seq_path1_dim': self.seq_path1_dim,
            'seq_path2_dim': self.seq_path2_dim,
            'seq_output_dim': self.seq_output_dim,
        }
        torch.save(checkpoint, path)
        LOGGER.info(f"[DualDRLAgent] 模型已保存至: {path}")

    def load_model(self, path: str):
        """加载模型"""
        if not os.path.exists(path):
            LOGGER.warning(f"[DualDRLAgent] 模型文件不存在: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.routing_net.load_state_dict(checkpoint['routing_net'])
        self.routing_target.load_state_dict(checkpoint['routing_target'])
        self.seq_net.load_state_dict(checkpoint['seq_net'])
        self.seq_target.load_state_dict(checkpoint['seq_target'])
        self.routing_optimizer.load_state_dict(checkpoint['routing_optimizer'])
        self.seq_optimizer.load_state_dict(checkpoint['seq_optimizer'])

        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        self.discount_factor = checkpoint.get('discount_factor', self.discount_factor)
        self.training_step = checkpoint.get('training_step', 0)
        self.training_history = checkpoint.get('training_history', self.training_history)
        self.episode_rewards = checkpoint.get('episode_rewards', [])

        LOGGER.info(f"[DualDRLAgent] 模型已从 {path} 加载")
