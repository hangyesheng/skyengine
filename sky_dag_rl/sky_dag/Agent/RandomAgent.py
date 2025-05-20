from .BaseAgent import BaseAgent
from typing import List, Optional, Tuple
from sky_dag_rl.sky_dag.sky_env.Graph.Machine import Machine
from sky_dag_rl.sky_dag.sky_env.Graph.Operation import Operation
from sky_dag_rl.sky_dag.sky_env.Graph.AGV import AGV
import time
import random

# Agent:
# input: 状态
# output: [(Operation, AGV, Machine), ...] （job可以混合）


class RandomAgent(BaseAgent):
    def __init__(self, name=None, agent_id=None, context=None):
        """
        通用智能体基类
        :param name: 智能体名称
        :param agent_id: 智能体ID或唯一标识
        :param context: 可选的上下文或环境句柄
        """
        super().__init__(name, agent_id, context)

    def reward(self, *args, **kwargs):
        """Agent 计算自身的reward"""
        pass

    def train(self, *args, **kwargs):
        """GreedyAgent 不需训练"""
        pass

    def sample(self, agvs, machines, jobs):
        """
        返回本次采样结果
        """
        time_start = time.time()
        current_sample = []

        for i, job in enumerate(jobs):
            selected_machine = None
            for j in range(job.get_operation_count()):
                op = job.get_operation(j)

                # 随机选择可处理当前操作的机器
                valid_machines = [m for m in machines if op and m.can_process(op)]
                machine = random.choice(valid_machines) if valid_machines else None

                # 随机选择可用AGV（考虑当前是否正在运输）
                available_agvs = [agv for agv in agvs if agv.is_available()]
                agv = random.choice(available_agvs) if available_agvs else None

                # 如果有机器和AGV则分配任务
                if agv and op and machine:
                    # 记录前一个操作的位置用于计算运输时间
                    prev_location = selected_machine.get_id() if selected_machine else None

                    # 更新AGV状态
                    transport_time = self.calculate_transport_time(prev_location, machine.get_id())
                    agv.assign_task(op, transport_time)

                    # 更新机器状态
                    process_time = op.get_duration(machine.get_id())
                    machine.add_occupation(agv.available_time, process_time)
                    selected_machine = machine

                current_sample.append((op, agv, machine))

        time_end = time.time()
        return current_sample, time_end - time_start

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.agent_id} name={self.name}>"