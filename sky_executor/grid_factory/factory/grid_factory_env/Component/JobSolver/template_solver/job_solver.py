'''
@Project ：SkyEngine 
@File    ：job_solver.py
@IDE     ：PyCharm
@Author  ：Skyrimforest
@Date    ：2025/10/27 22:49
'''


class JobSolver:
    def __init__(self, env):
        self.env = env
        self.current_plan = None  # 当前调度结果缓存
        self.pending_ops = []  # 尚未分配的操作
        self.active_ops = []  # 正在执行的操作
        self.finished_ops = []  # 已完成操作
        self.transfer_requests = []  # 交由RouteSolver处理的搬运请求

    def observe(self, obs: dict):
        """
        接收环境状态观察。
        obs 通常包括：
        - 当前机器状态（idle / busy / occupied_until）
        - Job 状态（ready / running / finished）
        - 当前时间步
        - Operation 就绪情况（前序是否完成）
        - Transfer 完成信息（AGV完成物料搬运）

        返回内部可用的 job/task 信息
        """
        pass

    def plan(self, obs: dict) -> dict:
        """
        在当前时刻进行一次决策：
        输入: 观测obs
        输出:
        {
            'machine_actions': List[MachineAction],
            'transfer_requests': List[TransferRequest]
        }

        machine_actions 示例:
        [
            {'machine_id': 0, 'job_id': 2, 'op_id': 1, 'start_time': 10.0, 'expected_end': 18.0},
            ...
        ]

        transfer_requests 示例:
        [
            {'job_id': 2, 'op_id': 1, 'from_machine': 1, 'to_machine': 3, 'ready_time': 12.0}
        ]
        """
        pass

    def update_after_step(self, infos):
        """
        每个时间步后由环境调用，更新执行状态：
        - 哪些任务完成了
        - 哪些转运完成
        - 哪些机器空闲
        """
        pass

