from abc import ABC, abstractmethod
import time
from typing import List, Tuple, Any, Dict, Optional

# Agent:
# input: 状态
# output: [(Operation, AGV, Machine), ...] （job 可以混合）


from executor.packet_factory.registry import register_component

# 默认最小步长时间（秒）
DEFAULT_STEP_TIME = 1

# Agent 运行模式
TRAINING = "training"
EVALUATION = "evaluation"
INFERENCE = "inference"

@register_component("packet_factory.BaseAgent")
class BaseAgent(ABC):
    def __init__(self, name=None, agent_id=None, context=None, mode: str = TRAINING, model_path: Optional[str] = None):
        """
        通用智能体基类
        :param name: 智能体名称
        :param agent_id: 智能体 ID 或唯一标识
        :param context: 可选的上下文或环境句柄
        :param mode: 运行模式 training | evaluation | inference
        :param model_path: 模型文件路径
        """
        self.name = name or self.__class__.__name__
        self.agent_id = agent_id
        self.context = context
        self.alive = True  # 是否在线
        self.turns = 0  # 存活轮次
        
        # 决策时间统计
        self.total_decision_time = 0.0  # 总决策时间
        self.decision_count = 0  # 决策次数
        
        # 强化学习相关配置
        self.mode = mode
        self.model_path = model_path

    def is_alive(self):
        return self.alive

    @abstractmethod
    def reward(self, *args, **kwargs):
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

    def reset_decision_stats(self):
        """重置决策统计信息"""
        self.total_decision_time = 0.0
        self.decision_count = 0

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.agent_id} name={self.name}>"
