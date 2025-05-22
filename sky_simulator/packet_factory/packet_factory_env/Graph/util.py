from enum import Enum


class OperationStatus(Enum):
    # 前序任务未满足
    WAITING = 0
    # 可以参与此次调度
    READY = 1
    # 在AGV上移动
    MOVING = 2
    # 开始执行
    WORKING = 3
    # 该任务执行就绪
    FINISHED = 4

    # 意外状态
    EXCEPTION = 99


class AGVStatus(Enum):
    # AGV可以运送Operation
    READY = 0
    # AGV正在运行(不含Operation)
    MOVING = 1
    # AGV正在运行(含Operation)
    LOADED = 2

    # 意外状态
    EXCEPTION = 99


class MachineStatus(Enum):
    # 当前机器可以接收Operation
    READY = 0
    # 当前机器正在运行Operation
    WORKING = 1
    # 该任务执行就绪
    FINISHED = 2
    # 当前机器宕机
    FAILED = 3

    # 意外状态
    EXCEPTION = 99


class JobStatus(Enum):
    # 该Job是否已经开始
    STARTED = 0
    # 该Job是否已经结束
    FINISHED = 1

    # 意外状态
    EXCEPTION = 99
