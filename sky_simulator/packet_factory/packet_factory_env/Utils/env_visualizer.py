import pygame

from sky_simulator.packet_factory.packet_factory_env.packet_factory_env import PacketFactoryEnv
from sky_simulator.packet_factory.packet_factory_env.Graph.util import OperationStatus, MachineStatus, AGVStatus, JobStatus

from sky_simulator.packet_factory.packet_factory_env.Graph.Machine import Machine
from sky_simulator.packet_factory.packet_factory_env.Graph.AGV import AGV
from sky_simulator.packet_factory.packet_factory_env.Graph.Operation import Operation


def scale(pos, scale=100):
    return (int(pos[0] * scale + scale), int(pos[1] * scale + scale))

class Env_visualizer:
    # 配置
    WIDTH, HEIGHT = 800, 600


    # 颜色
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    GREEN = (0, 200, 0)
    BLUE = (0, 100, 255)
    ORANGE = (255, 165, 0)
    RED = (200, 0, 0)

    OPERATION_STATE_COLOR = {
        OperationStatus.WAITING: (230, 230, 230),   # 浅灰 - 等待
        OperationStatus.READY: (255, 160, 0),       # 橙色 - 就绪
        OperationStatus.MOVING: (255, 100, 0),      # 深橙 - 移动中
        OperationStatus.WORKING: (255, 50, 0),      # 红橙 - 执行中
        OperationStatus.FINISHED: (200, 0, 0),      # 暗红 - 完成
        OperationStatus.EXCEPTION: (139, 0, 0)      # 深红 - 异常
    }

    AGV_STATE_COLOR = {
        AGVStatus.READY: (0, 180, 0),               # 鲜绿 - 可用
        AGVStatus.ASSIGNED: (0, 140, 0),            # 中绿 - 已分配
        AGVStatus.LOADED: (0, 100, 0),              # 深绿 - 已装载
        AGVStatus.EXCEPTION: (0, 60, 0)             # 极深绿 - 异常
    }

    MACHINE_STATE_COLOR = {
        MachineStatus.READY: (100, 100, 255),        # 浅蓝 - 就绪
        MachineStatus.WORKING: (70, 70, 200),        # 中蓝 - 执行中
        MachineStatus.FINISHED: (30, 30, 150),       # 深蓝 - 完成
        MachineStatus.FAILED: (139, 0, 139),         # 紫色 - 故障
        MachineStatus.EXCEPTION: (100, 0, 100)       # 深紫 - 异常
    }

    JOB_STATE_COLOR = {
        JobStatus.B4START: (200, 200, 200),         # 灰色 - 未开始
        JobStatus.STARTED: (255, 165, 0),           # 橙色 - 进行中
        JobStatus.FINISHED: (0, 200, 0),            # 绿色 - 已完成
        JobStatus.EXCEPTION: (200, 0, 0)            # 红色 - 异常
    }

    def __init__(self, env: PacketFactoryEnv) -> None:
        self.env = env

        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

    def draw_agv(self, screen, agv: AGV):
        color = self.AGV_STATE_COLOR.get(agv.status, self.BLACK)
        pos = scale(agv.get_xy())
        pygame.draw.circle(screen, color, pos, 15)
        font = pygame.font.SysFont(None, 24)
        label = font.render(str(agv.id), True, self.BLACK)
        screen.blit(label, (pos[0], pos[1]))

    def draw_machine(self, screen, machine: Machine):
        color = self.MACHINE_STATE_COLOR.get(machine.status, self.BLACK)
        position = scale(machine.get_xy())
        rect = pygame.Rect(position[0], position[1], 40, 40)
        pygame.draw.rect(screen, color, rect)
        font = pygame.font.SysFont(None, 24)
        label = font.render(str(machine.id), True, self.BLACK)
        screen.blit(label, (position[0], position[1]))

    def draw_operation(self, screen, operation: Operation, position):
        color = self.OPERATION_STATE_COLOR.get(operation.status, self.BLACK)
        rect = pygame.Rect(position[0]+10, position[1]+10, 16, 16)
        pygame.draw.rect(screen, color, rect)
        font = pygame.font.SysFont(None, 24)
        label = font.render(str(operation.id), True, self.WHITE)
        screen.blit(label, (position[0]+10, position[1]+10))

    def visualize_env(self, fps):
        # 渲染
        self.screen.fill(self.WHITE)

        for machine in self.env.machines:
            self.draw_machine(self.screen, machine)
            machine_operation = machine.get_operation()
            if machine_operation is not None:
                self.draw_operation(self.screen, machine_operation, scale(machine.get_xy()))

        for agv in self.env.agvs:
            self.draw_agv(self.screen, agv)
            agv_operation = agv.get_operation()
            if agv_operation is not None:
                self.draw_operation(self.screen, agv_operation, scale(agv.get_xy()))

        pygame.display.flip()
        self.clock.tick(fps)