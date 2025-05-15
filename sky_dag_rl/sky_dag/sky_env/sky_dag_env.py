from typing import Union

from pettingzoo import ParallelEnv
import numpy as np

from sky_dag_rl.sky_dag.Agent import BaseAgent
from sky_dag_rl.sky_dag.env.Graph.Node import Node
from sky_dag_rl.sky_dag.env.Graph.Job import Job
from sky_dag_rl.sky_dag.env.Graph.Operation import Operation
from sky_dag_rl.sky_dag.env.Event.Event import Event,EventQueue
import json


class SkyDagEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "sky_dag_env"}

    def __init__(self,
                 node_config_path="node_config.json",
                 job_config_path="job_config.json",
                 assign_scheduler=None,
                 graph_agent: BaseAgent = None,
                 underlay_agent: BaseAgent = None,
                 overlay_agent: BaseAgent = None,
                 operation_agent: Union[BaseAgent, list] = None,
                 node_agent: Union[BaseAgent, list] = None,
                 ):
        self.node_config_path = node_config_path
        self.job_config_path = job_config_path

        # 模拟源点和终点
        self.source=None
        self.destination=None

        # underlay图状态
        self.nodes = {}
        self.grid_size = ()

        # overlay图状态
        self.jobs = {}
        self.operations = []

        # 环境本身的状态 带宽等
        self.env_timeline = 0
        self.reward=0
        self.event_queue = EventQueue()

        # 智能体相关的状态
        self.graph_agent = None
        self.underlay_agent = None
        self.overlay_agent = None
        self.operation_agents = []
        self.node_agents = []

    # ---------- 自定义状态更新函数 ----------
    def set_env_timeline(self, count):
        assert count >= 0 and isinstance(count, int)
        self.env_timeline = count

    def get_env_timeline(self) -> int:
        return self.env_timeline

    def refresh_underlay(self):
        """
        刷新当前环境的underlay
        :return:
        """
        self.nodes.clear()

    def refresh_overlay(self):
        """
        刷新当前环境的overlay
        :return:
        """
        self.jobs.clear()
        self.operations.clear()

    def clear_graph(self):
        """
        清空状态
        :return:None
        """
        # ----------初始化图本身----------
        self.refresh_overlay()
        self.refresh_underlay()

        print("Environment Initialized Successfully.")

    def load_node_graph(self):
        """
        读取并创建underlay图
        :return: None
        """
        with open(self.node_config_path, 'r') as f:
            config = json.load(f)
        for node_cfg in config.get('nodes', []):
            node = Node(
                node_id=node_cfg['id'],
                position=tuple(node_cfg['position']),
                cpu_capacity=node_cfg['cpu'],
                mem_capacity=node_cfg['memory']
            )
            self.nodes[node.id] = node

        for link in config.get('links', []):
            self.nodes[link['from']].connect_to(self.nodes[link['to']], link['delay'])

    def load_job_graph(self):
        """
        读取job图的信息
        :return: None
        """
        with open(self.job_config_path, 'r') as f:
            config = json.load(f)

        for job_cfg in config.get('jobs', []):
            job = Job(job_cfg['id'],job_cfg['target_count'])
            op_dict = {}

            # 创建 operation 实例并加入 job
            for op_cfg in job_cfg['operations']:
                op = Operation(
                    op_id=op_cfg['id'],
                    cpu_req=op_cfg['cpu_req'],
                    mem_req=op_cfg['mem_req'],
                    duration=op_cfg['duration']
                )
                job.add_operation(op)
                op_dict[op.id] = op
                self.operations.append(op)

            # 建立操作之间的依赖关系
            for dep in job_cfg.get('dependencies', []):
                from_op = op_dict[dep['from']]
                to_op = op_dict[dep['to']]
                from_op.add_dependency(to_op)
                to_op.add_dependency(from_op)

            self.jobs[job.id] = job  # 存储 Job 对象，确保是 Job 类型

    def deal_event(self,event_list):
        for event in event_list:
            if event.event_type == "just_test":
                print(event.payload)
            elif event.event_type == "task_finish":
                op = event.payload
            elif event.event_type == "machine_fail":
                for node_id in event.payload['nodes']:
                    self.nodes[node_id].fail()

    def step(self, actions=None):
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # === 0. 处理全局时间 ===
        self.env_timeline += 1
        obs = self._get_obs()

        # === 1. Agent 动作处理（支持 Job 或 Central）===
        if actions:
            self._apply_agent_actions(actions)

        # === 2. 处理 EventQueue 中的事件 ===
        current_event_list = self.event_queue.pop_ready_events(self.env_timeline)
        self.deal_event(current_event_list)

        # === 3. 提取state,发送给状态转移函数并返回 ===
        for node in self.nodes.values():
            finished_ops = node.step(self.env_timeline)
            for op in finished_ops:
                self.event_queue.add_event(Event(
                    timestamp=self.env_timeline,
                    event_type="task_finish",
                    payload=op
                ))

        # === 4. 统计 Job 完成状态，计算奖励 ===
        for job in self.jobs:
            done = job.is_finished()
            job_id = job.id
            rewards[job_id] = 1.0 if done and not self.done_flags.get(job_id, False) else 0.0
            terminations[job_id] = done
            truncations[job_id] = False
            infos[job_id] = {}
            if done:
                self.done_flags[job_id] = True



        # === 5. 判断是否需要自动重调度 ===
        if self.should_trigger_reassign():
            self.reschedule_operations()

        return obs, rewards, terminations, truncations, infos

    def _get_obs(self):
        """
        获得环境观察的信息
        :return: 物理节点的观察信息
        """
        obs = {}
        for name, node in self.nodes.items():
            cpu_load = sum(o.cpu_req for o in node.running_operations) / node.cpu_capacity
            mem_load = sum(o.mem_req for o in node.running_operations) / node.mem_capacity
            obs[name] = np.array([
                min(cpu_load, 1.0),
                min(mem_load, 1.0),
                len(node.running_operations)
            ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        """
        返回环境初始状态下智能体的观察值,使环境整体的状态回到开始时的样子。
        :param seed: 随机种子,复现实验
        :param options:选项
        :return:
        """
        # ---------- 清理阶段 ----------
        self.set_env_timeline(0)
        self.clear_graph()
        # ---------- 重建阶段 ----------
        self.load_node_graph()
        self.load_job_graph()
        obs = self._get_obs()
        return obs

    def render(self):
        """
        给出观察的渲染结果
        :return:
        """
        print(f"\n=== Step {self.get_env_timeline()} | Grid State ===")
        for name, node in self.nodes.items():
            print(f"{name} at {node.position}: {len(node.running_operations)} ops running")

    def print_all_status(self):
        """
        打印所有节点、作业、操作及其当前状态
        :return: None
        """
        print("\n[节点状态]")
        for node_id, node in self.nodes.items():
            print(
                f"Node {node_id} - CPU: {node.cpu_capacity}, Memory: {node.mem_capacity}, Connections: {len(node.connections)}")

        print("\n[作业状态]")
        for job_id, job in self.jobs.items():
            print(f"Job {job_id} - Operations: {len(job.operations)}")
            for op in job.operations:
                print(
                    f"  Operation {op.id} - State: {op.state}, Assigned Node: {op.assigned_node.id if op.assigned_node else 'None'}")

        print("\n[待运行操作]")

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def should_trigger_reassign(self):
        pass

    def reschedule_operations(self):
        pass

    def get_op_by_id(self, op_id):
        for job in self.jobs:
            for op in job.operations:
                if op.id == op_id:
                    return op
        return None

    def _apply_agent_actions(self, actions):
        """
        执行 agent 提供的动作，支持集中式与 Job 式调度。
        actions 格式：
            - job-based: {"job_id1": {"op_id": "op1", "target_node": "node1"}, ...}
            - centralized: {"scheduler": [{"op_id": "op1", "target_node": "node1"}, ...]}
        """
        if "scheduler" in actions:
            # Centralized Scheduler 模式
            for assignment in actions["scheduler"]:
                op = self.get_op_by_id(assignment["op_id"])
                node = self.nodes.get(assignment["target_node"])
                if op and node and op.state == "idle" and node.can_accept(op):
                    node.assign_operation(op)
                    if op in self.pending_operations:
                        self.pending_operations.remove(op)
        else:
            # Job Agents 独立调度
            for job_id, assignment in actions.items():
                op = self.get_op_by_id(assignment["op_id"])
                node = self.nodes.get(assignment["target_node"])
                if op and node and op.state == "idle" and node.can_accept(op):
                    node.assign_operation(op)
                    if op in self.pending_operations:
                        self.pending_operations.remove(op)

