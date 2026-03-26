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

@register_component("packet_factory.SimpleRLAgent")
class SimpleRLAgent(BaseAgent):
    def __init__(self, name=None, agent_id=None, context=None, mode: str = TRAINING, model_path: Optional[str] = None):
        """
        简单强化学习 Agent，使用 Q-learning 方法
        :param name: 智能体名称
        :param agent_id: 智能体 ID
        :param context: 环境上下文
        :param mode: 运行模式 training | evaluation | inference
        :param model_path: 模型文件路径
        """
        super().__init__(name, agent_id, context, mode, model_path)
        
        # Q-learning 参数
        self.q_table: Dict[str, Dict[Tuple[int, int, int], float]] = {}  # {state_key: {(op_id, agv_id, machine_id): q_value}}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # 训练统计
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0
        self.training_history: Dict[str, List] = {
            'episodes': [],
            'rewards': [],
            'makespans': [],
            'epsilon': []
        }
        
        # 状态空间维度
        self.state_dim = 10  # 状态特征维度
        self.action_space_size = 100  # 最大动作数量
        
        # 最近的状态和动作（用于训练更新）
        self.last_state_key: Optional[str] = None
        self.last_action: Optional[Tuple[Operation, AGV, Machine]] = None
        
        # 加载模型（如果有）
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            LOGGER.info(f"[SimpleRLAgent] 加载模型成功：{model_path}")
    
    def _get_state_key(self, agvs: List[AGV], machines: List[Machine], jobs: List[Any]) -> str:
        """
        将环境状态编码为字符串 key
        :param agvs: AGV 列表
        :param machines: 机器列表
        :param jobs: Job 列表
        :return: 状态字符串
        """
        # 提取关键特征
        job_progress = sum(job.get_progress() for job in jobs) / len(jobs) if jobs else 0
        available_agvs = sum(1 for agv in agvs if agv.get_status() == AGVStatus.READY)
        available_machines = sum(1 for m in machines if m.is_available())
        ready_operations = sum(
            1 for job in jobs 
            for op in [job.get_operation(i) for i in range(job.get_operation_count())]
            if op.get_status() == OperationStatus.READY
        )
        
        # 离散化特征
        state_features = (
            int(job_progress * 10),  # 进度 0-10
            min(available_agvs, 5),  # 可用 AGV 数 0-5
            min(available_machines, 5),  # 可用机器数 0-5
            min(ready_operations, 10),  # 就绪工序数 0-10
        )
        
        return str(state_features)
    
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
                
                # 只分配 READY 或 WAITING 状态的 operation
                if op.get_status() not in [OperationStatus.READY]:
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
    
    def _choose_action(self, state_key: str, valid_actions: List[Tuple[Operation, AGV, Machine]]) -> Optional[Tuple[Operation, AGV, Machine]]:
        """
        根据 epsilon-greedy 策略选择动作
        :param state_key: 状态 key
        :param valid_actions: 合法动作列表
        :return: 选中的动作
        """
        if not valid_actions:
            return None
        
        # 初始化 Q 值（使用小的随机值打破对称性）
        if state_key not in self.q_table:
            self.q_table[state_key] = {
                self._action_to_key(a): random.uniform(-0.1, 0.1) for a in valid_actions
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
        计算奖励值
        :param args: 位置参数
        :param kwargs: 关键字参数，可能包含 env、observations 等
        :return: 奖励值
        """
        # 从参数中提取环境信息
        env_info = kwargs.get('env', {})
        if not env_info and args:
            env_info = args[0] if isinstance(args[0], dict) else {}
        
        reward = 0.0
        
        # 基于 makespan 的奖励（时间越长奖励越低）
        if self.context and hasattr(self.context, 'env_timeline'):
            time_penalty = -0.01 * self.context.env_timeline
            reward += time_penalty
        
        # 完成任务的奖励
        if self.context and hasattr(self.context, 'jobs'):
            completed_jobs = sum(1 for job in self.context.jobs if job.is_finished())
            total_jobs = len(self.context.jobs)
            completion_bonus = 10.0 * (completed_jobs / total_jobs) if total_jobs > 0 else 0
            reward += completion_bonus
        
        # 机器利用率奖励
        if self.context and hasattr(self.context, 'machines'):
            working_machines = sum(1 for m in self.context.machines if m.status == MachineStatus.WORKING)
            total_machines = len(self.context.machines)
            utilization_bonus = 0.5 * (working_machines / total_machines) if total_machines > 0 else 0
            reward += utilization_bonus
        
        # AGV 利用率奖励
        if self.context and hasattr(self.context, 'agvs'):
            working_agvs = sum(1 for agv in self.context.agvs if agv.get_status() != AGVStatus.READY)
            total_agvs = len(self.context.agvs)
            agv_bonus = 0.3 * (working_agvs / total_agvs) if total_agvs > 0 else 0
            reward += agv_bonus
        
        # 等待惩罚
        if self.context and hasattr(self.context, 'jobs'):
            waiting_ops = sum(
                1 for job in self.context.jobs
                for op in [job.get_operation(i) for i in range(job.get_operation_count())]
                if op.get_status() == OperationStatus.WAITING
            )
            waiting_penalty = -0.1 * waiting_ops
            reward += waiting_penalty
        
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
            action = self._choose_action(state_key, valid_actions)
            if action:
                current_sample.append(action)
                # 保存状态和动作用于后续训练
                self.last_state_key = state_key
                self.last_action = action
        elif self.mode in [EVALUATION, INFERENCE]:
            # 评估和推理模式：使用贪婪策略（总是选择最优）
            if state_key not in self.q_table:
                # 评估模式下初始化 Q 值为 0
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
            
            # Q-learning 更新公式：Q(s,a) += α[r + γ*max(Q(s')) - Q(s,a)]
            current_q = self.q_table[self.last_state_key][action_key]
            target_q = reward + self.discount_factor * max_next_q
            self.q_table[self.last_state_key][action_key] += self.learning_rate * (target_q - current_q)
            
            # 重置 last_state 和 last_action
            self.last_state_key = None
            self.last_action = None
    
    def train(self, episodes: int = 100, max_steps: int = 1000) -> None:
        """
        训练 Agent
        :param episodes: 训练回合数
        :param max_steps: 每回合最大步数
        """
        LOGGER.info(f"[SimpleRLAgent] 开始训练，episodes={episodes}, max_steps={max_steps}")
        
        for episode in range(episodes):
            episode_reward = 0.0
            step = 0
            done = False
            
            LOGGER.info(f"Episode {episode + 1}/{episodes}")
            
            # 重置 episode 奖励
            self.current_episode_reward = 0.0
            
            # 运行一个完整的 episode
            while not done and step < max_steps:
                # 这里需要环境支持 reset 和 step
                # 由于训练需要在外部环境中进行，这里只做标记
                step += 1
                
                # 检查是否应该终止（由外部环境控制）
                if not self.alive:
                    done = True
            
            # 记录 episode 结果
            episode_makespan = self.context.env_timeline if self.context and hasattr(self.context, 'env_timeline') else step
            
            self.training_history['episodes'].append(episode + 1)
            self.training_history['rewards'].append(self.current_episode_reward)
            self.training_history['makespans'].append(episode_makespan)
            self.training_history['epsilon'].append(self.epsilon)
            
            # epsilon 衰减
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            LOGGER.info(f"Episode {episode + 1}: Reward={self.current_episode_reward:.2f}, "
                       f"Makespan={episode_makespan:.2f}, Epsilon={self.epsilon:.4f}")
        
        LOGGER.info(f"[SimpleRLAgent] 训练完成")
    
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
            
            # 计算下一个状态的最大 Q 值（如果是终止状态则为 0）
            max_next_q = 0.0
            if not done and next_state_key in self.q_table and self.q_table[next_state_key]:
                max_next_q = max(self.q_table[next_state_key].values())
            
            # Q-learning 更新公式
            current_q = self.q_table[self.last_state_key][action_key]
            target_q = reward + self.discount_factor * max_next_q
            self.q_table[self.last_state_key][action_key] += self.learning_rate * (target_q - current_q)
            
            LOGGER.debug(f"Q-update: state={self.last_state_key}, action={action_key}, "
                        f"reward={reward:.2f}, new_q={self.q_table[self.last_state_key][action_key]:.4f}")
        
        # 重置 last_state 和 last_action
        self.last_state_key = None
        self.last_action = None
    
    def save_model(self, path: Optional[str] = None) -> bool:
        """
        保存模型到文件
        :param path: 文件路径，如果为 None 则使用默认路径 training_logs/models/agent_model.json
        :return: 是否成功
        """
        try:
            # 如果未指定路径，使用默认路径
            if path is None:
                # 获取 Agent 名称作为目录名
                agent_name = self.name or "SimpleRLAgent"
                agent_dir = f"training_logs/models/{agent_name}"
                path = f"{agent_dir}/agent_model.json"
            
            # 确保目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 保存 Q 表和超参数
            model_data = {
                'q_table': {k: {str(k2): v2 for k2, v2 in v.items()} 
                           for k, v in self.q_table.items()},
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'mode': self.mode,
                'training_history': self.training_history,
                'metadata': {
                    'agent_name': self.name,
                    'agent_id': self.agent_id,
                    'save_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'state_dim': self.state_dim,
                    'action_space_size': self.action_space_size
                }
            }
            
            with open(path, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            LOGGER.info(f"[SimpleRLAgent] 模型已保存到：{path}")
            LOGGER.info(f"  - Q-table size: {len(self.q_table)}")
            LOGGER.info(f"  - Epsilon: {self.epsilon:.4f}")
            return True
        except Exception as e:
            LOGGER.error(f"[SimpleRLAgent] 保存模型失败：{e}")
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
                    # 将字符串 key 转换回 tuple
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
            
            LOGGER.info(f"[SimpleRLAgent] 模型加载成功：{path}, Q-table size: {len(self.q_table)}")
            return True
        except Exception as e:
            LOGGER.error(f"[SimpleRLAgent] 加载模型失败：{e}")
            return False
    
    @staticmethod
    def get_default_model_dir(agent_name: Optional[str] = None) -> str:
        """
        获取默认模型保存目录
        :param agent_name: Agent 名称，如果为 None 则返回所有模型的根目录
        :return: 模型目录路径
        """
        if agent_name:
            return f"training_logs/models/{agent_name}"
        return "training_logs/models"
    
    @staticmethod
    def get_default_result_dir(agent_name: Optional[str] = None) -> str:
        """
        获取默认训练结果保存目录
        :param agent_name: Agent 名称，如果为 None 则返回所有结果的根目录
        :return: 结果目录路径
        """
        if agent_name:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            return f"training_logs/results/{agent_name}_{timestamp}"
        return "training_logs/results"
    
    @staticmethod
    def get_default_evaluation_dir(agent_name: Optional[str] = None) -> str:
        """
        获取默认评估结果保存目录
        :param agent_name: Agent 名称，如果为 None 则返回所有评估的根目录
        :return: 评估目录路径
        """
        if agent_name:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            return f"training_logs/evaluations/{agent_name}_{timestamp}"
        return "training_logs/evaluations"
    
    def list_available_models(self, agent_name: Optional[str] = None) -> List[str]:
        """
        列出所有可用的模型文件
        :param agent_name: Agent 名称，如果为 None 则列出所有模型
        :return: 模型文件路径列表
        """
        model_dir = self.get_default_model_dir(agent_name)
        
        if not os.path.exists(model_dir):
            LOGGER.warning(f"模型目录不存在：{model_dir}")
            return []
        
        model_files = []
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.json'):
                    model_files.append(os.path.join(root, file))
        
        if not model_files:
            LOGGER.info(f"在 {model_dir} 中未找到模型文件")
        
        return sorted(model_files)
    
    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.agent_id} name={self.name} mode={self.mode}>"

