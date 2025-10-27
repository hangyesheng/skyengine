'''
@Project ：SkyEngine 
@File    ：solver.py
@IDE     ：PyCharm
@Author  ：Skyrimforest
@Date    ：2025/10/27 23:02
'''

class GreedyJobSolver(JobSolver):
    def __init__(self, env):
        super().__init__(env)
        self.initialized = False

    def plan(self, obs):
        if not self.initialized:
            # 初始一次全局规划
            self.current_plan = priority_greedy(
                self.env.jobs, self.env.machines,
                transfer_time_estimator=self.env.transfer_time_estimator
            )
            self.initialized = True

        # 在当前时间步筛选可以启动的任务
        t = obs['time']
        ready_actions = []
        for mid, tasks in self.current_plan.machine_schedule.items():
            for (s, e, jid, oid) in tasks:
                if abs(s - t) < 1e-6:  # 当前时刻刚好要开始
                    ready_actions.append({
                        "machine_id": mid, "job_id": jid, "op_id": oid,
                        "start_time": s, "expected_end": e
                    })

        # 输出当前时间步可执行任务 + 对应的转运请求
        return {
            "machine_actions": ready_actions,
            "transfer_requests": self.current_plan.transfer_requests
        }

