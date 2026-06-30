from executor.packet_factory.event.event.BaseEvent import BaseEvent
from executor.packet_factory.event.EventType import EventType
from executor.packet_factory.registry.registry import register_event


@register_event('packet_factory.JOB_ADD')
class EventJobAdd(BaseEvent):
    event_type = EventType.JOB_ADD

    def __init__(self,status:str="trigger",payload:dict=None):
        super().__init__(status,payload)

        assert payload is not None, "payload不能为None"
        assert 'job' in payload, "payload必须包含job字段"
        self.job=payload['job']

    def _next_operation_id(self) -> int:
        """
        计算全局唯一的下一个 operation id。

        Graph 系列 Agent (GraphPPO/GRPO/Dual/DP) 使用 op.id 作为图节点键，
        因此 operation id 必须跨所有 Job 全局唯一，否则新增作业的 operation id
        会与已有作业冲突，破坏图结构。BackendMapLoader 通过递增的 operation_count
        保证这一点，这里沿用同样的策略为动态新增的作业分配 id。
        """
        max_op_id = -1
        for j in self.env.jobs:
            for i in range(j.get_operation_count()):
                op = j.get_operation(i)
                if op.id > max_op_id:
                    max_op_id = op.id
        return max_op_id + 1

    def _build_job(self):
        """
        将 payload 中的 job 描述转换为真正的 Job 对象。

        支持两种输入：
        1. dict：来自 event_timeline (JSON) 的原始字典，格式与 generate_uncertainty_events
           生成的一致：{"id":..., "operations":[{"id":..., "machines":[{"id","time"}, ...]}, ...]}
           （兼容 job_config 的 {"operation": {...}} 包装格式）
        2. Job：来自 EnvVisualizer 路径的已克隆 Job 对象。

        无论如何输入，都重新分配全局唯一的 operation id，避免与已有作业冲突。
        """
        from executor.packet_factory.packet_factory.packet_factory_env.Job.Job import Job
        from executor.packet_factory.packet_factory.packet_factory_env.Job.Operation import Operation
        from executor.packet_factory.packet_factory.packet_factory_env.Utils.util import OperationStatus

        src = self.job

        # 已经是 Job 对象（EnvVisualizer 路径）：提取 durations 重建
        if isinstance(src, Job):
            next_op_id = self._next_operation_id()
            operations = []
            for i in range(src.get_operation_count()):
                src_op = src.get_operation(i)
                durations = list(src_op.durations)
                operations.append(Operation(next_op_id, OperationStatus.WAITING, durations))
                next_op_id += 1
            return Job(src.id, operations, target_count=src.target_count)

        # dict 路径（event_timeline / JSON）
        operations = []
        next_op_id = self._next_operation_id()
        for op_entry in src.get('operations', []):
            # 兼容 {"operation": {...}} 包装格式与未包装格式
            op_data = op_entry.get('operation', op_entry)
            durations = [
                (int(m['id']), float(m['time']))
                for m in op_data.get('machines', [])
            ]
            operations.append(Operation(next_op_id, OperationStatus.WAITING, durations))
            next_op_id += 1

        return Job(int(src['id']), operations)

    def trigger(self):
        """
        触发该事件：向环境中追加一个新作业。
        """
        from executor.packet_factory.packet_factory.packet_factory_env.packet_factory_env import PacketFactoryEnv
        self.judge_env(PacketFactoryEnv)
        self.env: PacketFactoryEnv

        new_job = self._build_job()

        self.env.jobs.append(new_job)
        # 同步更新 hash_index，保持与 createHashIndex 的一致性
        # 使用 setdefault 避免覆盖已存在的同 ID 作业（如 EnvVisualizer 克隆路径）
        if hasattr(self.env, 'hash_index'):
            self.env.hash_index['jobs'].setdefault(new_job.id, new_job)




