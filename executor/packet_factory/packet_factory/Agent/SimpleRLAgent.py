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

@register_component("packet_factory.SimpleRLAgent")
class SimpleRLAgent(BaseAgent):
    def __init__(self, name=None, agent_id=None, context=None, mode: str = TRAINING, model_path: Optional[str] = None):
        """
        简单强化学习 Agent，使用 Q-learning 或策略梯度方法
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
        self.epsilon = 0.3  # exploration rate
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # 训练统计
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        
        # 状态空间维度
        self.state_dim = 10  # 状态特征维度
        self.action_space_size = 100  # 最大动作数量
        
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
        
        # 初始化 Q 值
        if state_key not in self.q_table:
            self.q_table[state_key] = {self._action_to_key(a): 0.0 for a in valid_actions}
        
        # epsilon-greedy 策略
        if random.random() < self.epsilon:
            # 探索：随机选择
            return random.choice(valid_actions)
        else:
            # 利用：选择 Q 值最大的动作
            q_values = {self._action_to_key(a): self.q_table[state_key].get(self._action_to_key(a), 0.0) 
                       for a in valid_actions}
            best_action_key = max(q_values, key=q_values.get)
            
            # 找到对应的动作
            for action in valid_actions:
                if self._action_to_key(action) == best_action_key:
                    return action
            
            return random.choice(valid_actions)
    
    def reward(self, observations: Dict[str, Any]) -> float:
        """
        计算奖励值
        :param observations: 观察信息
        :return: 奖励值
        """
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
        elif self.mode in [EVALUATION, INFERENCE]:
            # 评估和推理模式：使用贪婪策略（总是选择最优）
            if state_key not in self.q_table:
                self.q_table[state_key] = {self._action_to_key(a): 0.0 for a in valid_actions}
            
            # 选择 Q 值最大的动作
            q_values = {self._action_to_key(a): self.q_table[state_key].get(self._action_to_key(a), 0.0) 
                       for a in valid_actions}
            best_action_key = max(q_values, key=q_values.get)
            
            for action in valid_actions:
                if self._action_to_key(action) == best_action_key:
                    current_sample.append(action)
                    break
        else:
            # 默认随机策略
            action = random.choice(valid_actions) if valid_actions else None
            if action:
                current_sample.append(action)
        
        # 更新 Q 值（训练模式）
        if self.mode == TRAINING and current_sample:
            op, agv, machine = current_sample[0]
            action_key = self._action_to_key(current_sample[0])
            
            # 确保 Q 表中有该状态
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            if action_key not in self.q_table[state_key]:
                self.q_table[state_key][action_key] = 0.0
            
            # Q-learning 更新（简化版，实际应该在 update 中执行）
            # 这里只是简单增加一点 Q 值表示这个动作被选择了
            self.q_table[state_key][action_key] += self.learning_rate * (self.reward({}) - self.q_table[state_key][action_key])
        
        LOGGER.info(f"Finished jobs: {sum(1 for job in jobs if job.is_finished())}/{len(jobs)}")
        
        return current_sample, DEFAULT_STEP_TIME
    
    def train(self, episodes: int = 100, max_steps: int = 1000) -> Dict[str, Any]:
        """
        训练 Agent
        :param episodes: 训练回合数
        :param max_steps: 每回合最大步数
        :return: 训练统计信息
        """
        LOGGER.info(f"[SimpleRLAgent] 开始训练，episodes={episodes}, max_steps={max_steps}")
        
        training_stats = {
            'episodes': [],
            'rewards': [],
            'makespans': []
        }
        
        for episode in range(episodes):
            episode_reward = 0.0
            step = 0
            
            # 这里需要环境支持 reset 和 step
            # 由于训练需要在外部环境中进行，这里只做标记
            LOGGER.info(f"Episode {episode + 1}/{episodes}")
            
            # epsilon 衰减
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            training_stats['episodes'].append(episode + 1)
            training_stats['rewards'].append(episode_reward)
        
        LOGGER.info(f"[SimpleRLAgent] 训练完成")
        return training_stats
    
    def update(self, observations: Dict[str, Any], rewards: Dict[str, float]):
        """
        更新模型参数（训练时使用）
        :param observations: 观察信息
        :param rewards: 奖励字典
        """
        if self.mode != TRAINING:
            return
        
        # 获取当前 agent 的奖励
        agent_id = self.agent_id if self.agent_id else 'agent'
        reward = rewards.get(agent_id, 0.0)
        
        # 这里可以添加更复杂的更新逻辑，如策略梯度、Actor-Critic 等
        # 当前使用简化的 Q-learning 更新
        pass
    
    def save_model(self, path: str) -> bool:
        """
        保存模型到文件
        :param path: 文件路径
        :return: 是否成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 保存 Q 表和超参数
            model_data = {
                'q_table': {k: {str(k2): v2 for k2, v2 in v.items()} 
                           for k, v in self.q_table.items()},
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'mode': self.mode
            }
            
            with open(path, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            LOGGER.info(f"[SimpleRLAgent] 模型已保存到：{path}")
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
            self.q_table = {k: {tuple(map(int, k2.split(','))): v2 
                               for k2, v2 in v.items()} 
                          for k, v in model_data.get('q_table', {}).items()}
            
            # 恢复超参数
            self.learning_rate = model_data.get('learning_rate', self.learning_rate)
            self.discount_factor = model_data.get('discount_factor', self.discount_factor)
            self.epsilon = model_data.get('epsilon', self.epsilon)
            
            LOGGER.info(f"[SimpleRLAgent] 模型加载成功：{path}, Q-table size: {len(self.q_table)}")
            return True
        except Exception as e:
            LOGGER.error(f"[SimpleRLAgent] 加载模型失败：{e}")
            return False
    
    def evaluate_and_save(self, results: Dict[str, Any], save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        评估模型并保存结果
        :param results: 评估结果
        :param save_path: 保存路径
        :return: 评估报告
        """
        report = {
            'mode': self.mode,
            'model_path': self.model_path,
            'results': results,
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon
        }
        
        # 保存评估报告
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=2)
                LOGGER.info(f"[SimpleRLAgent] 评估报告已保存到：{save_path}")
            except Exception as e:
                LOGGER.error(f"[SimpleRLAgent] 保存评估报告失败：{e}")
        
        # 打印到命令行
        LOGGER.info("\n=== 评估报告 ===")
        LOGGER.info(f"模式：{self.mode}")
        LOGGER.info(f"Q-table 大小：{len(self.q_table)}")
        LOGGER.info(f"Epsilon: {self.epsilon:.4f}")
        for key, value in results.items():
            LOGGER.info(f"{key}: {value}")
        LOGGER.info("================\n")
        
        return report
    
    def get_decision_stats(self) -> Dict[str, float]:
        """获取决策统计信息"""
        stats = super().get_decision_stats()
        stats['epsilon'] = self.epsilon
        stats['q_table_size'] = len(self.q_table)
        return stats
    
    def reset_decision_stats(self):
        """重置决策统计信息"""
        super().reset_decision_stats()
        self.current_episode_reward = 0.0
    
    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.agent_id} name={self.name} mode={self.mode}>"
