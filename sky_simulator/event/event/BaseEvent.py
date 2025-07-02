# 该文件内提供预定义的一些事件类型和事件类，同时提供事件队列，所有的事件均需要依照这种方法添加
# 具体的事件处理函数在回调中体现。
from pettingzoo import ParallelEnv

from sky_simulator.event.EventType import EventType


class BaseEvent:
    event_type = EventType.BASE_EVENT

    def __init__(self, status: str = "trigger", payload: dict = None):
        # 保证是已声明的事件类型
        self.status = status  # 事件初始行为
        self.payload = payload or {}  # 事件携带的数据
        self.env = None  # 事件可以手动绑定环境使用 或者每次传入不同的env使用

    def __repr__(self):
        return f"[Event] {self.event_type.name} - Payload: {self.payload}"

    def __call__(self, env: ParallelEnv):
        if self.status == "trigger":
            self.trigger(env)
        elif self.status == "recover":
            self.recover(env)
        else:
            raise ValueError(f"[Event] {self.event_type} Event {self.status} Wrong!")

        print(f"[Event] {self.event_type} Event {self.status} | Payload: {self.payload}")

    def trigger(self, env: ParallelEnv = None):
        """
        触发该事件
        """
        if env is not None:
            return self.event_type

    def recover(self, env: ParallelEnv = None):
        """
        恢复该事件的现场
        """
        if env is not None:
            return self.event_type

    def judge_env(self, env_type):
        """指定当前环境,确定当前事件和环境是对应的"""
        assert isinstance(self.env, env_type), f"env必须是{env_type}的实例"

    def set_env(self, env):
        self.env = env
        print(f"[Event] {self.event_type} Event combined with env")
