from .BaseAgent import BaseAgent, DEFAULT_STEP_TIME, TRAINING, EVALUATION, INFERENCE
from executor.packet_factory.packet_factory.packet_factory_env.Job.Operation import Operation
from executor.packet_factory.packet_factory.packet_factory_env.Machine.Machine import Machine
from executor.packet_factory.packet_factory.packet_factory_env.Agv.AGV import AGV
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

@register_component("packet_factory.AdvancedRLAgent")
class AdvancedRLAgent(BaseAgent):
    """
    高级深度强化学习调度代理
    
    基于柔性制造系统（FJSP）的复杂状态空间设计，包含：
    1. 机器信息（可用时间、负载量）
    2. 待调度作业信息（处理时间）
    3. 新到达作业信息（到达时间）
    4. 作业数量统计
    5. 排队作业的紧邻操作处理时间
    6. 剩余处理时间与后续工位可用时间
    7. 系统级信息（完成率、资源利用率）
    8. 异质性指标（变异系数 CV）
    """
    
    def __init__(self, name=None, agent_id=None, context=None, mode: str = TRAINING, model_path: Optional[str] = None):
        """
        初始化高级 DRL Agent
        
        :param name: 智能体名称
        :param agent_id: 智能体 ID
        :param context: 环境上下文
        :param mode: 运行模式 training | evaluation | inference
        :param model_path: 模型文件路径
        """
        super().__init__(name, agent_id, context, mode, model_path)
        
        # ========== Q-learning / DQN 参数 ==========
        self.q_table: Dict[str, Dict[Tuple[int, int, int], float]] = {}
        self.learning_rate = 0.05
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01
        
        # ========== 状态空间配置 ==========
        # 状态向量维度（动态计算，约 30-40 维）
        self.state_dim = 35
        self.action_space_size = 200
        
        # ========== 训练统计 ==========
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0
        self.training_history: Dict[str, List] = {
            'episodes': [],
            'rewards': [],
            'makespans': [],
            'epsilon': [],
            'avg_machine_utilization': [],
            'avg_agv_utilization': [],
            'completion_rate': []
        }
        
        # 最近的状态和动作（用于训练更新）
        self.last_state_key: Optional[str] = None
        self.last_action: Optional[Tuple[Operation, AGV, Machine]] = None
        self.last_state_vector: Optional[np.ndarray] = None
        
        # 加载模型（如果有）
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            LOGGER.info(f"[AdvancedRLAgent] 加载模型成功：{model_path}")
    
    def _extract_machine_info(self, machines: List[Machine], current_time: float) -> Dict[str, np.ndarray]:
        """
        提取机器信息特征
        
        :param machines: 机器列表
        :param current_time: 当前时间
        :return: 机器特征字典
        """
        if not machines:
            return {
                'available_times': np.array([0.0]),
                'load_sums': np.array([0.0])
            }
        
        available_times = []
        load_sums = []
        
        for machine in machines:
            # Available time: 机器最早可用时间
            am_k = machine.timer if machine.status == MachineStatus.WORKING else current_time
            available_times.append(am_k)
            
            # Sum of processing times: 当前负载量
            load_sum = sum(op.get_duration(machine.id) - op.process_time 
                          for op in machine.input_queue 
                          if op.get_status() in [OperationStatus.READY, OperationStatus.WAITING])
            load_sums.append(load_sum)
        
        return {
            'available_times': np.array(available_times),
            'load_sums': np.array(load_sums)
        }
    
    def _extract_job_info(self, jobs: List[Any], machines: List[Machine]) -> Dict[str, np.ndarray]:
        """
        提取待调度作业信息
        
        :param jobs: Job 列表
        :param machines: 机器列表
        :return: 作业特征字典
        """
        ready_operations = []
        processing_times = []
        
        for job in jobs:
            if job.is_finished():
                continue
            
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() == OperationStatus.READY:
                    ready_operations.append(op)
                    
                    # 获取该操作在各可选机器上的处理时间
                    op_times = [op.get_duration(m.id) for m in machines if op.is_machine_capable(m.id)]
                    if op_times:
                        processing_times.extend(op_times)
        
        return {
            'ready_op_count': np.array([len(ready_operations)]),
            'processing_times': np.array(processing_times) if processing_times else np.array([0.0])
        }
    
    def _extract_arrival_info(self, jobs: List[Any], current_time: float) -> Dict[str, np.ndarray]:
        """
        提取新到达作业信息
        
        :param jobs: Job 列表
        :param current_time: 当前时间
        :return: 到达信息字典
        """
        # 这里假设 jobs 中包含未来到达的作业信息
        # 实际实现需要根据具体的 Job 类结构调整
        future_jobs = [j for j in jobs if hasattr(j, 'arrival_time') and j.arrival_time > current_time]
        
        if future_jobs:
            min_arrival_time = min(j.arrival_time - current_time for j in future_jobs)
            expected_count = len(future_jobs)
        else:
            min_arrival_time = 999.0  # 无未来作业
            expected_count = 0
        
        return {
            'time_till_arrival': np.array([min_arrival_time]),
            'expected_arrivals': np.array([expected_count])
        }
    
    def _extract_queue_info(self, machines: List[Machine]) -> Dict[str, np.ndarray]:
        """
        提取队列信息和紧邻操作处理时间
        
        :param machines: 机器列表
        :return: 队列特征字典
        """
        total_jobs_in_system = 0
        queue_lengths = []
        imminent_processing_times = []
        remaining_times = []
        
        for machine in machines:
            queue_len = len(machine.input_queue)
            total_jobs_in_system += queue_len
            queue_lengths.append(queue_len)
            
            # 紧邻操作处理时间
            if machine.input_queue:
                next_op = machine.input_queue[0]
                imminent_time = next_op.get_duration(machine.id) - next_op.process_time
                imminent_processing_times.append(imminent_time)
                
                # 计算该作业的总剩余处理时间
                rpt = self._calculate_remaining_processing_time(next_op)
                remaining_times.append(rpt)
        
        queue_lengths_arr = np.array(queue_lengths) if queue_lengths else np.array([0])
        imminent_times_arr = np.array(imminent_processing_times) if imminent_processing_times else np.array([0.0])
        remaining_times_arr = np.array(remaining_times) if remaining_times else np.array([0.0])
        
        # 统计特征
        stats = {
            'jobs_in_system': np.array([total_jobs_in_system]),
            'avg_queue_length': np.array([np.mean(queue_lengths_arr)]),
            'sum_imminent_time': np.array([np.sum(imminent_times_arr)]),
            'mean_imminent_time': np.array([np.mean(imminent_times_arr)]),
            'min_imminent_time': np.array([np.min(imminent_times_arr)]),
            'total_remaining_time': np.array([np.sum(remaining_times_arr)]),
            'avg_remaining_time': np.array([np.mean(remaining_times_arr)]),
            'max_remaining_time': np.array([np.max(remaining_times_arr)])
        }
        
        return stats
    
    def _calculate_remaining_processing_time(self, operation: Operation) -> float:
        """
        计算作业的剩余处理时间（包括后续工序）
        
        :param operation: 当前操作
        :return: 剩余处理时间
        """
        remaining_time = operation.get_process_time()
        next_op = operation.get_next_operation()
        
        while next_op is not None:
            # 取平均处理时间作为估计
            if next_op.durations:
                avg_time = sum(dur for _, dur in next_op.durations) / len(next_op.durations)
                remaining_time += avg_time
            next_op = next_op.get_next_operation()
        
        return remaining_time
    
    def _extract_successor_info(self, operations: List[Operation], machines: List[Machine]) -> Dict[str, np.ndarray]:
        """
        提取后续工位可用时间信息
        
        :param operations: 操作列表
        :param machines: 机器列表
        :return: 后续工位特征字典
        """
        successor_available_times = []
        
        for op in operations:
            next_op = op.get_next_operation()
            if next_op is not None:
                # 找到下一工序的可选机器
                for m_id, _ in next_op.durations:
                    target_machine = next((m for m in machines if m.id == m_id), None)
                    if target_machine:
                        successor_available_times.append(target_machine.timer)
        
        if successor_available_times:
            asw_arr = np.array(successor_available_times)
            return {
                'avg_asw': np.array([np.mean(asw_arr)]),
                'min_asw': np.array([np.min(asw_arr)])
            }
        else:
            return {
                'avg_asw': np.array([0.0]),
                'min_asw': np.array([0.0])
            }
    
    def _extract_system_info(self, jobs: List[Any], machines: List[Machine], agvs: List[AGV], current_time: float) -> Dict[str, np.ndarray]:
        """
        提取系统级信息
        
        :param jobs: Job 列表
        :param machines: 机器列表
        :param agvs: AGV 列表
        :param current_time: 当前时间
        :return: 系统特征字典
        """
        # 完成率
        total_jobs = len(jobs)
        completed_jobs = sum(1 for j in jobs if j.is_finished())
        completion_rate = completed_jobs / total_jobs if total_jobs > 0 else 0.0
        
        # 可用时间占比
        machine_available_times = [m.timer for m in machines]
        total_available_time = sum(machine_available_times) if machine_available_times else 1.0
        available_time_shares = [t / total_available_time for t in machine_available_times] if total_available_time > 0 else [0.0] * len(machines)
        
        # AGV 利用率
        active_agvs = sum(1 for agv in agvs if agv.get_status() != AGVStatus.READY)
        agv_utilization = active_agvs / len(agvs) if agvs else 0.0
        
        # 机器利用率
        working_machines = sum(1 for m in machines if m.status == MachineStatus.WORKING)
        machine_utilization = working_machines / len(machines) if machines else 0.0
        
        return {
            'completion_rate': np.array([completion_rate]),
            'avg_available_time_share': np.array([np.mean(available_time_shares)]),
            'agv_utilization': np.array([agv_utilization]),
            'machine_utilization': np.array([machine_utilization])
        }
    
    def _calculate_cv(self, data: np.ndarray) -> float:
        """
        计算变异系数（Coefficient of Variation）
        
        :param data: 数据数组
        :return: 变异系数
        """
        if len(data) < 2 or np.mean(data) == 0:
            return 0.0
        std_dev = float(np.std(data))
        mean_val = float(np.mean(data))
        return std_dev / mean_val if mean_val != 0 else 0.0
    
    def _extract_heterogeneity_info(self, machines: List[Machine], operations: List[Operation]) -> Dict[str, np.ndarray]:
        """
        提取异质性信息（变异系数）
        
        :param machines: 机器列表
        :param operations: 操作列表
        :return: 异质性特征字典
        """
        # CV of processing time
        processing_times = []
        for op in operations:
            if op.durations:
                times = [dur for _, dur in op.durations]
                processing_times.extend(times)
        
        cv_processing_time = self._calculate_cv(np.array(processing_times)) if processing_times else 0.0
        
        # CV of remaining processing time
        remaining_times = [self._calculate_remaining_processing_time(op) for op in operations]
        cv_rpt = self._calculate_cv(np.array(remaining_times)) if remaining_times else 0.0
        
        # CV of available time
        available_times = np.array([m.timer for m in machines])
        cv_available_time = self._calculate_cv(available_times) if len(available_times) > 0 else 0.0
        
        return {
            'cv_processing_time': np.array([cv_processing_time]),
            'cv_remaining_time': np.array([cv_rpt]),
            'cv_available_time': np.array([cv_available_time])
        }
    
    def _get_state_vector(self, agvs: List[AGV], machines: List[Machine], jobs: List[Any]) -> np.ndarray:
        """
        构建完整的状态向量
        
        :param agvs: AGV 列表
        :param machines: 机器列表
        :param jobs: Job 列表
        :return: 状态向量 (numpy array)
        """
        current_time = self.context.env_timeline if self.context and hasattr(self.context, 'env_timeline') else 0.0
        
        # 1. 机器信息
        machine_info = self._extract_machine_info(machines, current_time)
        
        # 2. 作业信息
        job_info = self._extract_job_info(jobs, machines)
        
        # 3. 到达信息
        arrival_info = self._extract_arrival_info(jobs, current_time)
        
        # 4. 队列信息
        queue_info = self._extract_queue_info(machines)
        
        # 5. 收集所有 READY 操作
        ready_operations = []
        for job in jobs:
            if not job.is_finished():
                for i in range(job.get_operation_count()):
                    op = job.get_operation(i)
                    if op.get_status() == OperationStatus.READY:
                        ready_operations.append(op)
        
        # 6. 后续工位信息
        successor_info = self._extract_successor_info(ready_operations, machines)
        
        # 7. 系统信息
        system_info = self._extract_system_info(jobs, machines, agvs, current_time)
        
        # 8. 异质性信息
        heterogeneity_info = self._extract_heterogeneity_info(machines, ready_operations)
        
        # 拼接所有特征
        state_features = np.concatenate([
            # 机器特征（取统计值）
            np.array([np.mean(machine_info['available_times'])]),
            np.array([np.std(machine_info['available_times'])]),
            np.array([np.mean(machine_info['load_sums'])]),
            np.array([np.max(machine_info['load_sums'])]),
            
            # 作业特征
            job_info['ready_op_count'],
            np.array([np.mean(job_info['processing_times'])]),
            np.array([np.min(job_info['processing_times'])]),
            np.array([np.max(job_info['processing_times'])]),
            
            # 到达特征
            arrival_info['time_till_arrival'],
            arrival_info['expected_arrivals'],
            
            # 队列特征
            queue_info['jobs_in_system'],
            queue_info['avg_queue_length'],
            queue_info['sum_imminent_time'],
            queue_info['mean_imminent_time'],
            queue_info['min_imminent_time'],
            queue_info['total_remaining_time'],
            queue_info['avg_remaining_time'],
            queue_info['max_remaining_time'],
            
            # 后续工位特征
            successor_info['avg_asw'],
            successor_info['min_asw'],
            
            # 系统特征
            system_info['completion_rate'],
            system_info['avg_available_time_share'],
            system_info['agv_utilization'],
            system_info['machine_utilization'],
            
            # 异质性特征
            heterogeneity_info['cv_processing_time'],
            heterogeneity_info['cv_remaining_time'],
            heterogeneity_info['cv_available_time']
        ])
        
        # 归一化（防止数值过大）
        state_features = self._normalize_state(state_features)
        
        return state_features
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        归一化状态向量
        
        :param state: 原始状态向量
        :return: 归一化后的状态向量
        """
        # 使用 tanh 进行软归一化到 [-1, 1]
        normalized = np.tanh(state / 100.0)
        return normalized
    
    def _get_state_key(self, agvs: List[AGV], machines: List[Machine], jobs: List[Any]) -> str:
        """
        将环境状态编码为字符串 key（用于 Q-table 查找）
        
        :param agvs: AGV 列表
        :param machines: 机器列表
        :param jobs: Job 列表
        :return: 状态字符串
        """
        state_vector = self._get_state_vector(agvs, machines, jobs)
        
        # 离散化为整数元组
        discretized = tuple(np.round(state_vector * 10).astype(int))
        
        return str(discretized)
    
    def _get_valid_actions(self, agvs: List[AGV], machines: List[Machine], jobs: List[Any]) -> List[Tuple[Operation, AGV, Machine]]:
        """
        获取所有合法的动作
        
        :param agvs: AGV 列表
        :param machines: 机器列表
        :param jobs: Job 列表
        :return: 合法动作列表 [(Operation, AGV, Machine), ...]
        """
        valid_actions = []
        
        for job in jobs:
            if job.is_finished():
                continue
            
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                
                # 只分配 READY 状态的 operation
                if op.get_status() != OperationStatus.READY:
                    continue
                
                # 找到能执行该操作的机器
                capable_machines = [m for m in machines if op.is_machine_capable(m.id) and m.is_available()]
                
                # 找到可用的 AGV
                available_agvs = [agv for agv in agvs if agv.get_status() == AGVStatus.READY]
                
                # 生成所有合法组合
                for agv in available_agvs:
                    for machine in capable_machines:
                        valid_actions.append((op, agv, machine))
        
        return valid_actions
    
    def _action_to_key(self, action: Tuple[Operation, AGV, Machine]) -> Tuple[int, int, int]:
        """将动作转换为可哈希的 key"""
        op, agv, machine = action
        return (op.id, agv.id, machine.id)
    
    def _choose_action(self, state_key: str, valid_actions: List[Tuple[Operation, AGV, Machine]], 
                      state_vector: np.ndarray) -> Optional[Tuple[Operation, AGV, Machine]]:
        """
        根据 epsilon-greedy 策略选择动作
        
        :param state_key: 状态 key
        :param valid_actions: 合法动作列表
        :param state_vector: 状态向量
        :return: 选中的动作
        """
        if not valid_actions:
            return None
        
        # 初始化 Q 值
        if state_key not in self.q_table:
            self.q_table[state_key] = {
                self._action_to_key(a): random.uniform(-0.01, 0.01) for a in valid_actions
            }
        
        # epsilon-greedy 策略
        if random.random() < self.epsilon:
            # 探索：随机选择
            return random.choice(valid_actions)
        else:
            # 利用：选择 Q 值最大的动作
            q_values = {self._action_to_key(a): self.q_table[state_key].get(self._action_to_key(a), 0.0) 
                       for a in valid_actions}
            best_action_key = max(q_values.items(), key=lambda x: x[1])[0]
            
            # 找到对应的动作
            for action in valid_actions:
                if self._action_to_key(action) == best_action_key:
                    return action
            
            return random.choice(valid_actions)
    
    def reward(self, *args, **kwargs) -> float:
        """
        计算奖励值（综合考虑多个因素）
        
        :param args: 位置参数
        :param kwargs: 关键字参数
        :return: 奖励值
        """
        env_info = kwargs.get('env', {})
        if not env_info and args:
            env_info = args[0] if isinstance(args[0], dict) else {}
        
        reward = 0.0
        
        # 1. Makespan 惩罚（时间越长惩罚越大）
        if self.context and hasattr(self.context, 'env_timeline'):
            time_penalty = -0.005 * self.context.env_timeline
            reward += time_penalty
        
        # 2. 完成任务奖励
        if self.context and hasattr(self.context, 'jobs'):
            completed_jobs = sum(1 for job in self.context.jobs if job.is_finished())
            total_jobs = len(self.context.jobs)
            completion_bonus = 50.0 * (completed_jobs / total_jobs) if total_jobs > 0 else 0
            reward += completion_bonus
        
        # 3. 机器利用率奖励
        if self.context and hasattr(self.context, 'machines'):
            working_machines = sum(1 for m in self.context.machines if m.status == MachineStatus.WORKING)
            total_machines = len(self.context.machines)
            utilization_bonus = 2.0 * (working_machines / total_machines) if total_machines > 0 else 0
            reward += utilization_bonus
        
        # 4. AGV 利用率奖励
        if self.context and hasattr(self.context, 'agvs'):
            working_agvs = sum(1 for agv in self.context.agvs if agv.get_status() != AGVStatus.READY)
            total_agvs = len(self.context.agvs)
            agv_bonus = 1.5 * (working_agvs / total_agvs) if total_agvs > 0 else 0
            reward += agv_bonus
        
        # 5. 等待惩罚
        if self.context and hasattr(self.context, 'jobs'):
            waiting_ops = sum(
                1 for job in self.context.jobs
                for op in [job.get_operation(i) for i in range(job.get_operation_count())]
                if op.get_status() == OperationStatus.WAITING
            )
            waiting_penalty = -0.5 * waiting_ops
            reward += waiting_penalty
        
        # 6. 负载均衡奖励（鼓励均匀分配）
        if self.context and hasattr(self.context, 'machines'):
            machine_loads = [len(m.input_queue) for m in self.context.machines]
            if machine_loads:
                load_cv = self._calculate_cv(np.array(machine_loads))
                balance_bonus = -3.0 * load_cv  # CV 越小越好
                reward += balance_bonus
        
        self.current_episode_reward += reward
        return reward
    
    def sample(self, agvs: List[AGV], machines: List[Machine], jobs: List[Any]) -> Tuple[List[Tuple[Operation, AGV, Machine]], float]:
        """
        Agent 推理采样核心逻辑
        
        :param agvs: AGV 列表
        :param machines: 机器列表
        :param jobs: Job 列表
        :return: (decisions, step_time) 决策列表和步长时间
        """
        current_sample = []
        
        # 获取当前状态
        state_vector = self._get_state_vector(agvs, machines, jobs)
        state_key = self._get_state_key(agvs, machines, jobs)
        
        # 获取合法动作
        valid_actions = self._get_valid_actions(agvs, machines, jobs)
        
        if not valid_actions:
            # 检查是否所有任务完成
            finished_count = sum(1 for job in jobs if job.is_finished())
            if finished_count == len(jobs):
                self.alive = False
                return [], 0
            return [], DEFAULT_STEP_TIME
        
        # 根据模式选择动作
        if self.mode == TRAINING:
            # 训练模式：使用 epsilon-greedy 策略
            action = self._choose_action(state_key, valid_actions, state_vector)
            if action:
                current_sample.append(action)
                # 保存状态和动作用于后续训练
                self.last_state_key = state_key
                self.last_action = action
                self.last_state_vector = state_vector.copy()
        elif self.mode in [EVALUATION, INFERENCE]:
            # 评估和推理模式：使用贪婪策略
            if state_key not in self.q_table:
                self.q_table[state_key] = {self._action_to_key(a): 0.0 for a in valid_actions}
            
            # 选择 Q 值最大的动作
            q_values = {self._action_to_key(a): self.q_table[state_key].get(self._action_to_key(a), 0.0) 
                       for a in valid_actions}
            best_action_key = max(q_values.items(), key=lambda x: x[1])[0]
            
            for action in valid_actions:
                if self._action_to_key(action) == best_action_key:
                    current_sample.append(action)
                    break
        else:
            # 默认随机策略
            action = random.choice(valid_actions) if valid_actions else None
            if action:
                current_sample.append(action)
        
        LOGGER.info(f"Finished jobs: {sum(1 for job in jobs if job.is_finished())}/{len(jobs)}")
        
        return current_sample, DEFAULT_STEP_TIME
    
    def after_sample(self, *args, **kwargs):
        """
        采样后钩子函数，在训练模式下执行 Q-learning 更新
        """
        if self.mode != TRAINING:
            return
        
        # 如果有上一次的状态和动作，立即进行 Q 值更新
        if self.last_state_key and self.last_action:
            # 计算即时奖励
            reward = self.reward({})
            
            # 获取下一个状态
            agvs = kwargs.get('agvs', [])
            machines = kwargs.get('machines', [])
            jobs = kwargs.get('jobs', [])
            next_state_key = self._get_state_key(agvs, machines, jobs)
            
            # Q-learning 更新
            action_key = self._action_to_key(self.last_action)
            
            # 确保 Q 表中有当前状态
            if self.last_state_key not in self.q_table:
                self.q_table[self.last_state_key] = {}
            
            if action_key not in self.q_table[self.last_state_key]:
                self.q_table[self.last_state_key][action_key] = 0.0
            
            # 计算下一个状态的最大 Q 值
            max_next_q = 0.0
            if next_state_key in self.q_table and self.q_table[next_state_key]:
                max_next_q = max(self.q_table[next_state_key].values())
            
            # Q-learning 更新公式
            current_q = self.q_table[self.last_state_key][action_key]
            target_q = reward + self.discount_factor * max_next_q
            self.q_table[self.last_state_key][action_key] += self.learning_rate * (target_q - current_q)
            
            # 重置 last_state 和 last_action
            self.last_state_key = None
            self.last_action = None
            self.last_state_vector = None
    
    def train(self, episodes: int = 100, max_steps: int = 1000) -> None:
        """
        训练 Agent
        
        :param episodes: 训练回合数
        :param max_steps: 每回合最大步数
        """
        LOGGER.info(f"[AdvancedRLAgent] 开始训练，episodes={episodes}, max_steps={max_steps}")
        
        for episode in range(episodes):
            episode_reward = 0.0
            step = 0
            done = False
            
            LOGGER.info(f"Episode {episode + 1}/{episodes}")
            
            # 重置 episode 奖励
            self.current_episode_reward = 0.0
            
            # 运行一个完整的 episode
            while not done and step < max_steps:
                step += 1
                
                # 检查是否应该终止（由外部环境控制）
                if not self.alive:
                    done = True
            
            # 记录 episode 结果
            episode_makespan = self.context.env_timeline if self.context and hasattr(self.context, 'env_timeline') else step
            
            # 计算统计信息
            avg_machine_util = 0.0
            avg_agv_util = 0.0
            completion_rate = 0.0
            
            if self.context:
                if hasattr(self.context, 'machines'):
                    working = sum(1 for m in self.context.machines if m.status == MachineStatus.WORKING)
                    avg_machine_util = working / len(self.context.machines) if self.context.machines else 0
                
                if hasattr(self.context, 'agvs'):
                    active = sum(1 for a in self.context.agvs if a.get_status() != AGVStatus.READY)
                    avg_agv_util = active / len(self.context.agvs) if self.context.agvs else 0
                
                if hasattr(self.context, 'jobs'):
                    completed = sum(1 for j in self.context.jobs if j.is_finished())
                    completion_rate = completed / len(self.context.jobs) if self.context.jobs else 0
            
            self.training_history['episodes'].append(episode + 1)
            self.training_history['rewards'].append(self.current_episode_reward)
            self.training_history['makespans'].append(episode_makespan)
            self.training_history['epsilon'].append(self.epsilon)
            self.training_history['avg_machine_utilization'].append(avg_machine_util)
            self.training_history['avg_agv_utilization'].append(avg_agv_util)
            self.training_history['completion_rate'].append(completion_rate)
            
            # epsilon 衰减
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            LOGGER.info(f"Episode {episode + 1}: Reward={self.current_episode_reward:.2f}, "
                       f"Makespan={episode_makespan:.2f}, Epsilon={self.epsilon:.4f}, "
                       f"Completion={completion_rate:.2%}")
        
        LOGGER.info(f"[AdvancedRLAgent] 训练完成")
    
    def update(self, observations: Dict[str, Any], rewards: Dict[str, float], done: bool = False):
        """
        更新模型参数（训练时使用）
        
        :param observations: 观察信息
        :param rewards: 奖励字典
        :param done: 是否 episode 结束
        """
        if self.mode != TRAINING:
            return
        
        # 获取当前 agent 的奖励
        agent_id = self.agent_id if self.agent_id else 'agent'
        reward = rewards.get(agent_id, 0.0)
        
        # Q-learning 更新
        if self.last_state_key and self.last_action:
            # 获取下一个状态
            next_state_key = self._get_state_key(
                observations.get('agvs', []),
                observations.get('machines', []),
                observations.get('jobs', [])
            )
            
            # 计算目标 Q 值
            action_key = self._action_to_key(self.last_action)
            
            # 确保 Q 表中有当前状态
            if self.last_state_key not in self.q_table:
                self.q_table[self.last_state_key] = {}
            
            if action_key not in self.q_table[self.last_state_key]:
                self.q_table[self.last_state_key][action_key] = 0.0
            
            # 计算下一个状态的最大 Q 值
            max_next_q = 0.0
            if not done and next_state_key in self.q_table and self.q_table[next_state_key]:
                max_next_q = max(self.q_table[next_state_key].values())
            
            # Q-learning 更新公式
            current_q = self.q_table[self.last_state_key][action_key]
            target_q = reward + self.discount_factor * max_next_q
            self.q_table[self.last_state_key][action_key] += self.learning_rate * (target_q - current_q)
            
            LOGGER.debug(f"Q-update: state={self.last_state_key[:50]}..., action={action_key}, "
                        f"reward={reward:.2f}, new_q={self.q_table[self.last_state_key][action_key]:.4f}")
        
        # 重置 last_state 和 last_action
        self.last_state_key = None
        self.last_action = None
        self.last_state_vector = None
    
    def save_model(self, path: Optional[str] = None) -> bool:
        """
        保存模型到文件
        
        :param path: 文件路径，如果为 None 则使用默认路径
        :return: 是否成功
        """
        try:
            if path is None:
                agent_name = self.name or "AdvancedRLAgent"
                agent_dir = f"training_logs/models/{agent_name}"
                path = f"{agent_dir}/agent_model.json"
            
            # 确保目录存在（仅当 path 包含目录时）
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            
            # 保存 Q 表和超参数
            model_data = {
                'q_table': {k: {str(k2): v2 for k2, v2 in v.items()} 
                           for k, v in self.q_table.items()},
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'mode': self.mode,
                'training_history': self.training_history,
                'state_dim': self.state_dim,
                'action_space_size': self.action_space_size,
                'metadata': {
                    'agent_name': self.name,
                    'agent_id': self.agent_id,
                    'save_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'description': 'Advanced DRL Agent with comprehensive FJSP state features'
                }
            }
            
            with open(path, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            LOGGER.info(f"[AdvancedRLAgent] 模型已保存到：{path}")
            LOGGER.info(f"  - Q-table size: {len(self.q_table)}")
            LOGGER.info(f"  - Epsilon: {self.epsilon:.4f}")
            LOGGER.info(f"  - Episodes trained: {len(self.training_history.get('episodes', []))}")
            return True
        except Exception as e:
            LOGGER.error(f"[AdvancedRLAgent] 保存模型失败：{e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        从文件加载模型
        
        :param path: 文件路径
        :return: 是否成功
        """
        try:
            with open(path, 'r') as f:
                model_data = json.load(f)
            
            # 恢复 Q 表
            loaded_q_table = {}
            for k, v in model_data.get('q_table', {}).items():
                inner_dict = {}
                for k2, v2 in v.items():
                    key_parts = k2.strip('()').split(',')
                    if len(key_parts) == 3:
                        tuple_key = tuple(map(int, key_parts))
                        inner_dict[tuple_key] = float(v2)
                loaded_q_table[k] = inner_dict
            self.q_table = loaded_q_table
            
            # 恢复超参数
            self.learning_rate = model_data.get('learning_rate', self.learning_rate)
            self.discount_factor = model_data.get('discount_factor', self.discount_factor)
            self.epsilon = model_data.get('epsilon', self.epsilon)
            
            # 恢复训练历史
            if 'training_history' in model_data:
                self.training_history = model_data['training_history']
            
            LOGGER.info(f"[AdvancedRLAgent] 模型加载成功：{path}, Q-table size: {len(self.q_table)}")
            return True
        except Exception as e:
            LOGGER.error(f"[AdvancedRLAgent] 加载模型失败：{e}")
            return False
    
    @staticmethod
    def get_default_model_dir(agent_name: Optional[str] = None) -> str:
        """获取默认模型保存目录"""
        if agent_name:
            return f"training_logs/models/{agent_name}"
        return "training_logs/models"
    
    @staticmethod
    def get_default_result_dir(agent_name: Optional[str] = None) -> str:
        """获取默认训练结果保存目录"""
        if agent_name:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            return f"training_logs/results/{agent_name}_{timestamp}"
        return "training_logs/results"
    
    @staticmethod
    def get_default_evaluation_dir(agent_name: Optional[str] = None) -> str:
        """获取默认评估结果保存目录"""
        if agent_name:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            return f"training_logs/evaluations/{agent_name}_{timestamp}"
        return "training_logs/evaluations"
    
    def list_available_models(self, agent_name: Optional[str] = None) -> List[str]:
        """列出所有可用的模型文件"""
        model_dir = self.get_default_model_dir(agent_name)
        
        if not os.path.exists(model_dir):
            LOGGER.warning(f"模型目录不存在：{model_dir}")
            return []
        
        model_files = []
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.json'):
                    model_files.append(os.path.join(root, file))
        
        return sorted(model_files)
    
    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.agent_id} name={self.name} mode={self.mode}>"
