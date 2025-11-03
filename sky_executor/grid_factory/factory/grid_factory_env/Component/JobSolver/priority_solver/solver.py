"""
@Project ：SkyEngine
@File    ：solver.py
@IDE     ：PyCharm
@Author  ：Skyrimforest
@Date    ：2025/10/27 23:02
"""

from sky_executor.grid_factory.factory.grid_factory_env.Component.JobSolver.template_solver.job_solver import (
    JobSolver,
)
from sky_executor.grid_factory.factory.grid_factory_env.Component.JobSolver.utils.op_priority_greedy import (
    priority_greedy,
)
from sky_executor.grid_factory.factory.grid_factory_env.Utils.structure import (
    JobSolverResult,
)


#  在线动态调度 (Online Scheduling) 场景下：
# 环境本身就维护所有状态（机器是否忙、AGV是否空闲、任务是否完成）；
# JobSolver 每步都能从 obs 获取完整信息；
# → 因此可以是 无状态（stateless）决策器。


# 但是我们这里是离线静态调度 (Offline Scheduling) 所以还是需要维护缓存的。
# 不接收新的job
class GreedyJobSolver(JobSolver):
    """
    贪心调度器，首次执行，后续根据离线调度的结果一步步返回相关信息：
    - 首次调用 plan() 时进行全局调度；
    - 后续每步根据时间推进任务状态。
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.time_stamp = 0
        self.current_plan: JobSolverResult = None
        self.pending_ops = []  # 未启动的操作 (waiting)
        self.active_ops = []  # 执行中的操作 (running)
        self.finished_ops = []  # 已完成的操作 (done)
        self.transfer_requests = []  # 待处理的转运请求 (for RouteSolver)

    def plan(self, obs):
        """
        在当前时刻生成决策。
        obs = {
            "jobs": [Job],
            "machines": [Machine],"
            "pending_ops": [Op],"
            "active_ops": [Op],"
        }
        """
        t = obs.get("time", 0.0)

        # === 初始化阶段 ===
        if not self.initialized:
            assert self.env is not None, "GreedyJobSolver需要env以获取jobs/machines"
            self.current_plan = priority_greedy(
                self.env.jobs,
                self.env.machines,
                transfer_time_estimator=self.env.transfer_time_estimator,
            )
            self.initialized = True

            # 初始化 pending_ops（所有任务均未执行）
            for mid, tasks in self.current_plan.machine_schedule.items():
                for s, e, jid, oid in tasks:
                    self.pending_ops.append(
                        {
                            "machine_id": mid,
                            "job_id": jid,
                            "op_id": oid,
                            "start_time": s,
                            "expected_end": e,
                        }
                    )

            # 初始化转运任务缓存
            self.transfer_requests = list(self.current_plan.transfer_requests)

        # === 状态推进 ===
        self._update_status(t)

        # === 当前时刻可启动的任务 ===
        ready_actions = []
        for op in list(self.pending_ops):
            if abs(op["start_time"] - t) < 1e-6:  # 刚好到启动时刻
                ready_actions.append(op)
                self.active_ops.append(op)
                self.pending_ops.remove(op)

        # === 输出 ===
        return {
            "machine_actions": ready_actions,
            "transfer_requests": self._collect_ready_transfers(t),
        }

    # ---------------------------------------------------------
    # 状态更新逻辑
    # ---------------------------------------------------------
    def _update_status(self, current_time: float):
        """根据当前时间推进 active_ops 状态"""
        finished = []
        for op in list(self.active_ops):
            if op["expected_end"] <= current_time + 1e-6:
                finished.append(op)
                self.finished_ops.append(op)
                self.active_ops.remove(op)

    def _collect_ready_transfers(self, current_time: float):
        """筛选出到时间可触发的转运请求"""
        ready = []
        for req in list(self.transfer_requests):
            if req["ready_time"] <= current_time + 1e-6:
                ready.append(req)
                self.transfer_requests.remove(req)
        return ready

    def update_after_step(self, infos: dict):
        """
        infos 示例:
        {
            "finished_transfers": [ {"job_id":..., "op_id":..., "time": t_actual}, ... ],
            "finished_ops": [ {"job_id":..., "op_id":..., "time": t_actual}, ... ],
            "machine_updates": [ {"machine_id":..., "status":"down"/"up"}, ...]
        }
        """
        # 1) 标记 transfer 完成 -> 影响 op 的 ready 状态
        finished_transfers = infos.get("finished_transfers", [])
        for ft in finished_transfers:
            # 将对应 transfer 标记为已完成（或从待转列表移除）
            for req in list(self.transfer_requests):
                if req["job_id"] == ft["job_id"] and req["op_id"] == ft["op_id"]:
                    # 记录实际到达时间，方便后续调整
                    req["actual_ready_time"] = ft.get("time", None)
                    # keep it for record or remove if you only fire once
                    # here we keep and mark
                    req["done"] = True

        # 2) 标记已完成的 ops（环境可能提前或推迟）
        finished_ops = infos.get("finished_ops", [])
        for fo in finished_ops:
            # find active op and move to finished
            for op in list(self.active_ops):
                if op["job_id"] == fo["job_id"] and op["op_id"] == fo["op_id"]:
                    self.active_ops.remove(op)
                    self.finished_ops.append(op)
                    # update expected_end if actual time provided
                    if "time" in fo:
                        op["expected_end"] = fo["time"]

        # 3) 如果 transfer 延迟导致机器开工时间需调整，可选择：
        #    - 将受影响 op 的 start_time 推迟到 actual_ready_time
        #    - 或者触发 replan（更激进）
        # 简单策略：延迟受影响的 op
        for req in list(self.transfer_requests):
            if req.get("done", False):
                # find target op in pending_ops and adjust start_time
                for op in self.pending_ops:
                    if op["job_id"] == req["job_id"] and op["op_id"] == req["op_id"]:
                        # 如果实际到达晚于原计划，则推迟启动
                        ar = req.get("actual_ready_time")
                        if ar is not None and ar > op["start_time"]:
                            op["start_time"] = ar
