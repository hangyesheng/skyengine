from typing import List, Tuple, Optional, Dict, Any, Union
import copy

from pettingzoo import ParallelEnv

from executor.packet_factory.packet_factory.Agent import BaseAgent, DEFAULT_STEP_TIME
from executor.packet_factory.packet_factory.packet_factory_env.Job.Job import Job
from executor.packet_factory.packet_factory.packet_factory_env.Machine.Machine import Machine
from executor.packet_factory.packet_factory.packet_factory_env.Job.Operation import Operation
from executor.packet_factory.packet_factory.packet_factory_env.Agv.AGV import AGV
from executor.packet_factory.packet_factory.packet_factory_env.Graph.Graph import Graph
from executor.packet_factory.packet_factory.packet_factory_env.Utils.util import EnvStatus
from executor.packet_factory.logger.logger import LOGGER
from executor.packet_factory.registry import register_component
from executor.packet_factory.call_back.callback_manager.CallbackManager import CallbackManager


@register_component("packet_factory")
class PacketFactoryEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "packet_factory_env"}

    def __init__(self,
                 agent: BaseAgent,
                 mode: str = 'optimization'  # 'drl' 或 'optimization'
                 ):
        # 物料仓库与目标存储仓库
        self.source = []
        self.destination = []

        # 系统状态
        self.jobs = []
        self.machines = []
        self.agvs = []
        self.graph = None

        self.job_templates = []

        # 环境本身的状态，事件队列，智能体相关的状态等
        self.env_timeline: float = 0
        self.env_visualizer = None
        self.event_queue = None
        self.agent = agent
        
        # 运行模式
        # 'drl': 事件驱动模式，每次资源空闲时触发决策
        # 'optimization': 传统优化模式，一次性生成全局调度方案
        self.mode = mode
        
        self.hash_index = {
            'agvs': {},
            'machines': {},
            'jobs': {},
        }
        self.status = EnvStatus.PAUSED
        # 回调管理
        self.callback_manager: CallbackManager = CallbackManager()

    # ---------- 自定义状态更新函数 ----------
    def set_env_timeline(self, env_timeline: float):
        self.env_timeline = env_timeline

    def get_env_timeline(self) -> float:
        return self.env_timeline

    def set_callback_manager(self, callback_manager: CallbackManager):
        self.callback_manager = callback_manager
        LOGGER.info("CallbackManager Created Successfully.")

    def refresh_status(self):
        """
        刷新当前环境引入的额外组件
        :return:
        """
        # 环境创建 当场调用即可
        load_graph_callback = self.callback_manager.get('load_graph')
        if callable(load_graph_callback):
            result = load_graph_callback()
            if result is not None and isinstance(result, (list, tuple)) and len(result) == 4:
                self.jobs, self.machines, self.agvs, self.graph = result
            else:
                raise ValueError("load_graph callback must return a tuple of (jobs, machines, agvs, graph)")
        else:
            raise ValueError("load_graph callback is not callable")
        
        LOGGER.info(f"Environment loaded with {len(self.jobs)} jobs, {len(self.machines)} machines, {len(self.agvs)} AGVs.")
        
        self.job_templates = copy.deepcopy(self.jobs)
        self.createHashIndex()

        # 可视化组件赋值 不需要当场调用（支持为空）
        if self.callback_manager.has('initialize_visualizer'):
            self.env_visualizer = self.callback_manager.get('initialize_visualizer')
            if self.env_visualizer is not None and hasattr(self.env_visualizer, 'visualize_env'):
                self.env_visualizer.visualize_env(env=self)
        else:
            self.env_visualizer = None
            LOGGER.info("Visualizer is disabled.")

        # 事件队列 不需要当场调用
        if self.callback_manager.has('event_queue'):
            self.event_queue = self.callback_manager.get('event_queue')
            if self.event_queue is not None and hasattr(self.event_queue, 'set_env'):
                self.event_queue.set_env(env=self)
        
        LOGGER.info("Environment Initialized Successfully.")

    def action_space(self, agent) -> Dict[str, Any]:
        """
        生成动作空间，根据模式不同采用不同策略
        :param agent: Agent 实例
        :return: {"decisions": decisions, "step_time": step_time}
        """
        
        decisions, step_time = agent.decision(self.agvs, self.machines, self.jobs)
        
        return {
            "decisions": decisions,
            "step_time": step_time
        }

    def check_job_finished(self):
        # 检测任务是否执行完成
        if self.status == EnvStatus.FINISHED:
            return True
        else:
            cnt = 0
            for job in self.jobs:
                if job.is_finished():
                    cnt += 1
            res = False
            if cnt == len(self.jobs):
                self.agent.alive = False
                self.status = EnvStatus.WAITING
                res = True
        return res

    def deal_event(self):
        """
        调用 event_queue 取出队列中 current_time 之前的事件并调用.
        :return: bool 是否处理了事件
        """
        # ---------- 检测事件（支持可视化器为空）----------
        command_list = []
        if self.env_visualizer is not None and hasattr(self.env_visualizer, 'getBuffered'):
            try:
                command_list = self.env_visualizer.getBuffered()
            except (TypeError, AttributeError) as e:
                LOGGER.warning("Current Visualizer Don't support Event.")
                return False

        # ---------- 翻译创建事件 ----------
        for command in command_list:
            payload = {}
            if command['type'] == 'AGV':
                current_agv: AGV = command['data']
                payload = current_agv.pack()
            elif command['type'] == 'MACHINE':
                current_machine: Machine = command['data']
                payload = current_machine.pack()
            elif command['type'] == 'JOB':
                current_job: Job = command['data']
                payload['job'] = current_job
            else:
                pass

            if self.event_queue is not None and hasattr(self.event_queue, 'event_manager'):
                current_event = self.event_queue.event_manager.create_event(command['event_type'],
                                                                            *[command['event_method'], payload])
                if hasattr(self.event_queue, 'add_event'):
                    self.event_queue.add_event(self.env_timeline, event=current_event)

        # ---------- 执行事件 ----------
        ready_event = []
        if self.event_queue is not None and hasattr(self.event_queue, 'pop_ready_events'):
            ready_event = self.event_queue.pop_ready_events(self.env_timeline)
        
        for event in ready_event:
            if self.event_queue is not None and hasattr(self.event_queue, 'event_manager'):
                self.event_queue.event_manager.deal_event(event, self)

        if len(ready_event) == 0:
            return False
        else:
            return True

    def env_step(self, actions: List[Tuple[Operation, AGV, Machine]], step_time: float) -> bool:
        """
        环境的单时间片执行，返回当前有无事件发生
        :param actions: 决策列表
        :param step_time: 步长时间
        :return: bool 是否发生了事件
        """

        if len(actions) > 0:
            # ---------- 清空已调度但未执行的策略 ----------
            for agv in self.agvs:
                agv.todo_queue_clear()

            # ---------- 分配调度策略 ----------
            for operation, agv, machine in actions:
                agv.work(None, action=(operation, machine))

        event_happen = False
        if self.status == EnvStatus.WAITING:
            # ---------- 更新可视化（支持为空）----------
            if self.env_visualizer is not None and hasattr(self.env_visualizer, 'visualize_env'):
                self.env_visualizer.visualize_env()
            # ---------- 触发事件队列相关机制 ----------
            event_happen = self.deal_event()
        elif self.status == EnvStatus.PAUSED:
            # ---------- 更新可视化（支持为空）----------
            if self.env_visualizer is not None and hasattr(self.env_visualizer, 'visualize_env'):
                self.env_visualizer.visualize_env()
            # ---------- 触发事件队列相关机制 ----------
            event_happen = self.deal_event()
        elif self.status == EnvStatus.RUNNING:
            # ---------- 当前轮次时间 ----------
            current_time = self.env_timeline
            final_time = current_time + step_time

            # ---------- machine 执行已分配的工作 ----------
            for machine in self.machines:
                if len(machine.input_queue) > 0:
                    machine.work(final_time)
                machine.set_timer(final_time)

            # ---------- 执行调度策略 ----------
            for agv in self.agvs:
                agv.work(final_time)
                agv.set_timer(final_time)

            # ---------- 查看状态 ----------
            self.render_observation()

            # ---------- 更新可视化（支持为空）----------
            if self.env_visualizer is not None and hasattr(self.env_visualizer, 'visualize_env'):
                self.env_visualizer.visualize_env()

            # ---------- 触发事件队列相关机制 ----------
            event_happen = self.deal_event()

            # ---------- 处理全局时间 ----------
            self.env_timeline += step_time

        return event_happen

    def step(self, actions=None):
        """
        环境的主循环步骤，支持两种模式：
        1. DRL 事件驱动模式：每次资源空闲时触发决策
        2. 传统优化模式：使用预先生成的全局调度方案
        
        :param actions: 动作字典，包含 decisions 和 step_time
        :return: obs, rewards, terminations, truncations, infos
        """
        LOGGER.info(f"--------- 当前循环步为{self.env_timeline} (模式：{self.mode}) ---------")

        # === 0. Agent 决策动作 ===
        decisions = actions['decisions']
        step_time = actions.get('step_time', DEFAULT_STEP_TIME)

        # 记录决策统计信息
        if hasattr(self.agent, 'get_decision_stats'):
            stats = self.agent.get_decision_stats()
            LOGGER.info(f"Agent 决策统计：{stats}")

        LOGGER.info(f"step_time: {step_time}, decisions_count: {len(decisions)}")

        # === 1. 执行直到发生事件或完成 ===
        while True:
            # 确定环境是否执行完毕
            if self.check_job_finished():
                break
            
            # 执行当前决策，返回是否发生不确定性事件
            res = self.env_step(decisions, step_time)
            
            if self.mode == 'drl':
                # DRL 模式：永远需要重新决策                
                break
            else:
                # 优化模式：使用预生成的全局调度，发生不确定事件时重新决策
                if not res:
                    # 未发生事件，继续执行
                    decisions = []
                else:
                    break
                
        
        # === 2. 统计完成状态，计算奖励 ===
        agent_id = self.agent.agent_id if self.agent.agent_id is not None else 'agent'
        rewards_dict: Dict[str, Any] = {agent_id: self.agent.reward({})}
        obs = self._get_obs()
    
        # ---------- 更新可视化（支持为空）----------
        if self.env_visualizer is not None and hasattr(self.env_visualizer, 'visualize_env'):
            self.env_visualizer.visualize_env()

        LOGGER.info(f"--------- 结束当前循环步 ---------")

        return obs, rewards_dict, {}, {}, {}

    def _get_obs(self):
        """
        获得Agent观察的环境信息
        :return: 物理节点的观察信息
        """
        obs = {}
        return obs

    def reset(self, seed=None, options=None):
        """
        重置环境状态
        :param seed: 随机种子
        :param options: 额外选项
        """
        # ---------- 清理重建阶段 ----------
        self.set_env_timeline(0)
        
        # 重置 Agent 的决策统计
        if hasattr(self.agent, 'reset_decision_stats'):
            self.agent.reset_decision_stats()
                
        # 重新初始化环境
        self.refresh_status()
        
        # 返回初始观察和 info（符合 PettingZoo 接口）
        return self._get_obs(), {}

    # ---------- 渲染函数 ----------
    def render_observation(self):
        # 展示作业、机器和AGV数量
        # LOGGER.info(f"\n📊 系统资源状态:")
        LOGGER.info(f"  - 作业: {self.jobs}")
        LOGGER.info(f"  - 机器: {self.machines}")
        LOGGER.info(f"  - AGV: {self.agvs}")

    def render(self):
        """可视化系统当前状态 功能拆分到不同函数中"""
        pass

    def createHashIndex(self):
        """
        创建高效获取组件的结构
        """
        for agv in self.agvs:
            self.hash_index['agvs'][agv.id] = agv
        for job in self.jobs:
            self.hash_index['jobs'][job.id] = job
        for machine in self.machines:
            self.hash_index['machines'][machine.id] = machine

    def getJobs(self) -> List[Job]:
        return self.jobs
    
    def getJobTemplates(self) -> List[Job]:
        return self.job_templates

    def getMachines(self) -> List[Machine]:
        return self.machines

    def getAGVs(self) -> List[AGV]:
        return self.agvs

    def getGraph(self) -> Optional[Graph]:
        return self.graph

    def env_is_finished(self) -> bool:
        return self.status == EnvStatus.FINISHED

    def event_set_paused(self):
        self.status = EnvStatus.PAUSED

    def event_set_running(self):
        self.status = EnvStatus.RUNNING

    def event_set_restart(self):
        self.status = EnvStatus.EXCEPTION


if __name__ == '__main__':
    pass
