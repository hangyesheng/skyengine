from typing import Optional, Tuple, List

from .Operation import Operation
from .util import OperationStatus, MachineStatus
from sky_simulator.packet_factory.packet_factory_env.Utils.logger import LOGGER


class Machine:
    def __init__(self, machine_id: int, x: float, y: float, operation: Optional[Operation]):
        """
        :param id_: 机器 ID
        :param x: 坐标 X
        :param y: 坐标 Y
        :param operation: 当前正在执行的操作
        """
        # todo 状态转移需要实现

        self.id: int = machine_id
        self.x: float = x
        self.y: float = y
        self.timer: float = 0.0
        self.operation: Optional[Operation] = operation

        self.status: MachineStatus = MachineStatus.READY
        # 缓存的Operation
        self.input_queue: List[Operation] = []
        self.output_queue: List[Operation] = []

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.id}>"

    # ---------- 模拟Machine运行 ----------
    def is_available(self):
        op = self.get_operation()
        if op is None:
            return True
        else:
            return op.is_finished()

    def get_id(self) -> int:
        return self.id

    def get_xy(self) -> Tuple[float, float]:
        return self.x, self.y

    def get_timer(self) -> float:
        return self.timer

    def set_timer(self, timer: float) -> None:
        self.timer = timer

    def get_operation(self) -> Optional[Operation]:
        return self.operation

    def set_operation(self, operation: Optional[Operation]) -> None:
        self.operation = operation

    def get_status(self) -> MachineStatus:
        return self.status

    def set_status(self, status: MachineStatus):
        self.status = status

    def work(self, final_time: float) -> bool:
        if self.operation is not None:
            if self.operation.get_status() != OperationStatus.WORKING:
                return False
            duration = self.operation.get_duration(self.id)
            work_time: float = duration - self.operation.process_time
            if self.timer + work_time > final_time:
                self.operation.process_time += final_time - self.timer
                self.set_timer(final_time)
                return False

            self.timer += work_time
            self.operation.set_process_time(duration)
            self.operation.set_finished(True)
            self.operation.set_status(OperationStatus.FINISHED)

            self.operation.set_current_machine(None)
            self.operation = self.operation.get_next_operation()
            if self.operation is not None:
                self.operation.set_status(OperationStatus.READY)
                self.operation.set_current_machine(self)
            return True
        else:
            LOGGER.info(f"Machine {self.id} is idle")
            return False

    # ---------- 模拟异常事件 ----------
    def machine_fail(self):
        self.status=MachineStatus.FAILED