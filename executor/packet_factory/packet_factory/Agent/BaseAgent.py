from abc import ABC, abstractmethod
import time
from typing import List, Tuple, Any, Dict, Optional

# Agent:
# input: 状态
# output: [(Operation, AGV, Machine), ...] （job 可以混合）


from executor.packet_factory.registry import register_component

# 默认最小步长时间（秒）
DEFAULT_STEP_TIME = 1

# Agent 运行模式 - 2x2 分类
# 界面维度
FRONTEND = "frontend"  # 有界面（启用可视化）
BACKEND = "backend"    # 无界面（不启用可视化）

# 功能维度
TRAINING = "training"      # 训练模式（进行学习和更新）
INFERENCE = "inference"    # 推理模式（使用模型决策）

@register_component("packet_factory.BaseAgent")
class BaseAgent(ABC):
    def __init__(self, name=None, agent_id=None, context=None, 
                 ui_mode: str = BACKEND, task_mode: str = TRAINING, 
                 model_path: Optional[str] = None):
        """
        通用智能体基类
        :param name: 智能体名称
        :param agent_id: 智能体 ID 或唯一标识
        :param context: 可选的上下文或环境句柄
        :param ui_mode: 界面模式 frontend | backend（是否有可视化界面）
        :param task_mode: 任务模式 training | inference（训练还是推理）
        :param model_path: 模型文件路径（inference 模式必须提供）
        """
        self.name = name or self.__class__.__name__
        self.agent_id = agent_id
        self.context = context
        self.alive = True  # 是否在线
        self.turns = 0  # 存活轮次
        
        # 决策时间统计
        self.total_decision_time = 0.0  # 总决策时间
        self.decision_count = 0  # 决策次数
        
        # 强化学习相关配置 - 2x2 模式
        self.ui_mode = ui_mode      # frontend | backend
        self.task_mode = task_mode  # training | inference
        self.model_path = model_path
        
        # 兼容旧版本的 mode 属性（由两个维度组合而成）
        self.mode = f"{ui_mode}_{task_mode}"

    def is_alive(self):
        return self.alive

    @abstractmethod
    def reward(self, *args, **kwargs) -> float:
        """Agent 计算自身的 reward"""
        pass

    @abstractmethod
    def sample(self, *args, **kwargs) -> Tuple[List[Any], float]:
        """
        Agent 推理采样核心逻辑
        :param args: 参数 (agvs, machines, jobs)
        :param kwargs: 额外参数
        :return: (decisions, step_time) 决策列表和步长时间
        """
        pass

    def before_sample(self, *args, **kwargs):
        """采样前钩子函数"""
        pass

    def after_sample(self, *args, **kwargs):
        """采样后钩子函数"""
        pass

    def decision(self, *args, **kwargs) -> Tuple[List[Any], float]:
        """
        统一的决策接口，包含时间统计
        :param args: 传递给 sample 的参数 (agvs, machines, jobs)
        :param kwargs: 额外参数
        :return: (decisions, step_time) 决策列表和步长时间
        """
        start_time = time.time()
        
        # 前置处理
        self.before_sample(*args, **kwargs)
        
        # 执行采样
        result = self.sample(*args, **kwargs)
        
        # 后置处理
        self.after_sample(*args, **kwargs)
        
        # 统计决策时间
        end_time = time.time()
        decision_time = end_time - start_time
        
        # 累加总决策时间
        self.total_decision_time += decision_time
        self.decision_count += 1
        
        # 确保 step_time 不为负
        if isinstance(result, tuple) and len(result) == 2:
            decision_list, step_time = result
            step_time = max(step_time, DEFAULT_STEP_TIME)
            return decision_list, step_time
        else:
            # 如果 sample 返回格式不正确，使用默认值
            return [], DEFAULT_STEP_TIME

    @abstractmethod
    def train(self, *args, **kwargs):
        """Agent 训练"""
        pass
    
    def get_decision_stats(self) -> Dict[str, float]:
        """
        获取决策统计信息
        :return: dict 包含总决策时间、决策次数、平均决策时间
        """
        avg_time = self.total_decision_time / self.decision_count if self.decision_count > 0 else 0
        return {
            'total_decision_time': self.total_decision_time,
            'decision_count': self.decision_count,
            'average_decision_time': avg_time
        }

    def get_training_metrics(self) -> Dict[str, Any]:
        """
        获取训练指标（用于收敛检测）。子类可覆写以提供更具体的指标。

        :return: dict 包含 episode_reward, epsilon, training_history 等
        """
        metrics = {}
        if hasattr(self, '_episode_reward'):
            metrics['episode_reward'] = self._episode_reward
        elif hasattr(self, 'current_episode_reward'):
            metrics['episode_reward'] = self.current_episode_reward
        if hasattr(self, 'epsilon'):
            metrics['epsilon'] = self.epsilon
        if hasattr(self, 'training_history') and isinstance(self.training_history, dict):
            metrics['training_history'] = self.training_history
        return metrics

    def reset_decision_stats(self):
        """重置决策统计信息"""
        self.total_decision_time = 0.0
        self.decision_count = 0

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.agent_id} name={self.name}>"
