from typing import List
import pygame
import os

from executor.packet_factory.call_back.EnvCallback import EnvCallback
from executor.packet_factory.logger.logger import LOGGER
from executor.packet_factory.registry import register_component
from executor.packet_factory.packet_factory.packet_factory_env.Job.Job import Job
from executor.packet_factory.packet_factory.packet_factory_env.Machine.Machine import Machine
from executor.packet_factory.packet_factory.packet_factory_env.Agv.AGV import AGV
from executor.packet_factory.packet_factory.packet_factory_env.Job.Operation import Operation
from executor.packet_factory.packet_factory.packet_factory_env.Utils.util import OperationStatus, MachineStatus, AGVStatus


# 仿真环境创建前的初始化
@register_component("base_callback.Visualizer")
class EnvVisualizer(EnvCallback):
    WIDTH, HEIGHT = 1920, 1080

    # 颜色
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    OPERATION_STATE_COLOR = {
        OperationStatus.WAITING: (240, 240, 240),  # 浅灰 - 等待（更接近白，柔和）
        OperationStatus.READY: (255, 200, 0),  # 浅黄橙 - 就绪
        OperationStatus.MOVING: (255, 150, 0),  # 中橙 - 移动中
        OperationStatus.WORKING: (255, 90, 0),  # 橙红 - 执行中
        OperationStatus.FINISHED: (200, 50, 0),  # 暗红橙 - 完成
        OperationStatus.EXCEPTION: (255, 0, 0)  # 鲜红 - 异常（醒目突出）
    }

    AGV_STATE_COLOR = {
        AGVStatus.READY: (0, 200, 0),  # 鲜绿 - 可用
        AGVStatus.ASSIGNED: (0, 160, 0),  # 中绿 - 已分配
        AGVStatus.LOADED: (0, 120, 0),  # 深绿 - 已装载
        AGVStatus.EXCEPTION: (255, 0, 0)  # 鲜红 - 异常（与正常绿色系强烈对比）
    }

    MACHINE_STATE_COLOR = {
        MachineStatus.READY: (130, 180, 255),  # 浅蓝 - 就绪
        MachineStatus.WORKING: (80, 130, 200),  # 中蓝 - 执行中
        MachineStatus.FAILED: (170, 0, 170),  # 紫红 - 故障
        MachineStatus.EXCEPTION: (255, 0, 0)  # 鲜红 - 异常（与蓝色/紫色形成强对比）
    }

    AGV_SHIFT = (0, -0.10)
    AGV_OPERATION_SHIFT = (0.10, -0.10)
    MACHINE_SHIFT = (0, 0)
    MACHINE_INPUT_SHIFT = (-0.10, 0.12)
    MACHINE_OUTPUT_SHIFT = (0.34, 0.12)

    def __init__(self, _fps=3) -> None:
        super().__init__()
        self.fps = _fps
        self.env = None
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.clock = pygame.time.Clock()

        self.should_restart = False
        self.should_pause = False
        self.should_run = False

        self.agv_pause_queue = []
        self.agv_resume_queue = []
        self.machine_pause_queue = []
        self.machine_resume_queue = []
        self.job_add_queue = []
        
        self.uncertainty_event_queue = []

        self.overall_scale = None
        self.overall_shift = None
        
        # 加载 Emoji 字体（支持彩色 Emoji）
        self._load_emoji_fonts()

    def _load_emoji_fonts(self):
        """加载支持 Emoji 的字体"""
        # 尝试加载系统自带的 Emoji 字体
        emoji_font_paths = [
            "seguiemj.ttf",  # Windows Segoe UI Emoji
            "Apple Color Emoji.ttc",  # macOS
            "NotoColorEmoji.ttf",  # Linux Noto Color Emoji
            "Symbola.ttf",  # Linux Symbola
        ]
        
        self.emoji_font = None
        for font_path in emoji_font_paths:
            try:
                # 尝试从系统字体目录加载
                system_font_dirs = [
                    os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts'),
                    "/usr/share/fonts/truetype/",
                    "/System/Library/Fonts/",
                ]
                
                for font_dir in system_font_dirs:
                    full_path = os.path.join(font_dir, font_path)
                    if os.path.exists(full_path):
                        self.emoji_font = pygame.font.Font(full_path, 1)
                        LOGGER.info(f"✅ 成功加载 Emoji 字体: {full_path}")
                        return
                        
            except Exception as e:
                continue
        
        # 如果找不到专门的 Emoji 字体，使用默认字体（部分系统也支持 Emoji）
        try:
            self.emoji_font = pygame.font.SysFont("segoeuiemoji, apple color emoji, noto color emoji, symbola, arial", 1)
            LOGGER.info("✅ 使用系统默认字体（可能支持 Emoji）")
        except:
            self.emoji_font = pygame.font.Font(None, 1)
            LOGGER.warning("⚠️ 未找到 Emoji 字体，将使用简单符号替代")

    def __call__(self):
        """使类的实例可以像函数一样被调用"""
        self.visualize_env()

    def calculate_scale_and_shift(self, left, top, right, bottom):
        all_points = []
        
        # 收集所有点的位置
        graph = self.env.getGraph()
        for point in graph.points:
            all_points.append((point.x, point.y))
        for machine in self.env.getMachines():
            machine_x, machine_y = machine.get_xy()
            all_points.append((machine_x + self.MACHINE_SHIFT[0], machine_y + self.MACHINE_SHIFT[1]))
            all_points.append((machine_x + self.MACHINE_INPUT_SHIFT[0], machine_y + self.MACHINE_INPUT_SHIFT[1]))
            all_points.append((machine_x + self.MACHINE_OUTPUT_SHIFT[0], machine_y + self.MACHINE_OUTPUT_SHIFT[1]))
        for agv in self.env.getAGVs():
            agv_x, agv_y = agv.get_xy()
            all_points.append((agv_x + self.AGV_SHIFT[0], agv_y + self.AGV_SHIFT[1]))
            all_points.append((agv_x + self.AGV_OPERATION_SHIFT[0], agv_y + self.AGV_OPERATION_SHIFT[1]))

        # 计算边界
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)

        # 根据窗口大小计算缩放比例
        width_scale = (right - left - 200) / (max_x - min_x)  # 留出边距
        height_scale = (bottom - top - 200) / (max_y - min_y)  # 留出边距

        scale_factor = min(width_scale, height_scale)  # 选择较小的比例以适应窗口

        shift_x = left + 100 - min_x * scale_factor  # 确保最左端有100像素的边距
        shift_y = top + 100 - min_y * scale_factor  # 确保最上端有100像素的边距

        return scale_factor, (shift_x, shift_y)

    def scaling(self, pos, shift=(0,0)):
        return (int((pos[0] + shift[0]) * self.overall_scale + self.overall_shift[0]), int((pos[1] + shift[1]) * self.overall_scale + self.overall_shift[1] ))

    def draw_agv(self, screen, agv: AGV):
        """绘制 AGV - 使用 Emoji 图标"""
        color = self.AGV_STATE_COLOR.get(agv.status, self.BLACK)
        position = self.scaling(agv.get_xy(), shift=self.AGV_SHIFT)
        
        # 根据 AGV 状态选择不同的 Emoji
        emoji_map = {
            AGVStatus.READY: "🚗",      # 绿色小车 - 可用
            AGVStatus.ASSIGNED: "🚙",  # 蓝色小车 - 已分配
            AGVStatus.LOADED: "🚚",    # 货车 - 已装载
            AGVStatus.EXCEPTION: "⚠️",  # 警告标志 - 异常
        }
        
        emoji = emoji_map.get(agv.status, "🔵")  # 默认蓝色圆点
        
        # 计算字体大小（基于缩放比例）
        font_size = max(int(0.35 * self.overall_scale), 16)
        
        # 创建字体对象并渲染 Emoji
        try:
            agv_font = pygame.font.SysFont("segoeuiemoji, apple color emoji, noto color emoji, symbola, arial", font_size)
            emoji_surface = agv_font.render(emoji, True, color)
        except:
            # 降级方案：使用彩色圆圈
            radius = int(0.15 * self.overall_scale)
            pygame.draw.circle(screen, color, position, radius)
            # 添加边框
            pygame.draw.circle(screen, self.BLACK, position, radius, 2)
            
            # 绘制 AGV ID 标签
            label_font_size = max(int(0.20 * self.overall_scale), 12)
            label_font = pygame.font.SysFont(None, label_font_size)
            label = label_font.render(str(agv.id), True, self.BLACK)
            label_rect = label.get_rect(center=(position[0], position[1] + radius + label_font_size // 2 + 3))
            screen.blit(label, label_rect)
            return
        
        # 将 Emoji 居中绘制在 AGV 位置
        emoji_rect = emoji_surface.get_rect(center=position)
        screen.blit(emoji_surface, emoji_rect)
        
        # 绘制 AGV ID 标签（在 Emoji 下方）
        label_font_size = max(int(0.20 * self.overall_scale), 12)
        label_font = pygame.font.SysFont(None, label_font_size)
        label = label_font.render(str(agv.id), True, self.BLACK)
        label_rect = label.get_rect(center=(position[0], position[1] + font_size // 2 + 5))
        screen.blit(label, label_rect)

    def draw_machine(self, screen, machine: Machine):
        """绘制机器 - 使用 Emoji 图标"""
        color = self.MACHINE_STATE_COLOR.get(machine.status, self.BLACK)
        position = self.scaling(machine.get_xy(), shift=self.MACHINE_SHIFT)
        
        # 根据机器状态选择不同的 Emoji
        emoji_map = {
            MachineStatus.READY: "🏭",      # 工厂 - 就绪
            MachineStatus.WORKING: "⚙️",    # 齿轮 - 工作中
            MachineStatus.FAILED: "🔧",     # 扳手 - 故障（需要维修）
            MachineStatus.EXCEPTION: "❌",  # 叉号 - 异常
        }
        
        emoji = emoji_map.get(machine.status, "🏢")  # 默认建筑物
        
        # 计算字体大小（机器比 AGV 大一些）
        font_size = max(int(0.45 * self.overall_scale), 20)
        
        # 创建字体对象并渲染 Emoji
        try:
            machine_font = pygame.font.SysFont("segoeuiemoji, apple color emoji, noto color emoji, symbola, arial", font_size)
            emoji_surface = machine_font.render(emoji, True, color)
        except:
            # 降级方案：使用彩色方块
            rect_size = int(0.40 * self.overall_scale)
            rect = pygame.Rect(position[0], position[1], rect_size, rect_size)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, self.BLACK, rect, 2)  # 边框
            
            # 绘制机器 ID 标签
            label_font_size = max(int(0.20 * self.overall_scale), 12)
            label_font = pygame.font.SysFont(None, label_font_size)
            label = label_font.render(str(machine.id), True, self.BLACK)
            label_rect = label.get_rect(center=(position[0] + rect_size // 2, position[1] + rect_size + label_font_size // 2 + 3))
            screen.blit(label, label_rect)
            return
        
        # 将 Emoji 居中绘制在机器位置
        emoji_rect = emoji_surface.get_rect(center=position)
        screen.blit(emoji_surface, emoji_rect)
        
        # 绘制机器 ID 标签（在 Emoji 下方，增加偏移量以确保可见性）
        label_font_size = max(int(0.20 * self.overall_scale), 12)
        label_font = pygame.font.SysFont(None, label_font_size)
        label = label_font.render(str(machine.id), True, self.BLACK)
        # 增加垂直偏移量，从原来的 +8 改为 +12，确保 ID 不会被 Emoji 遮挡
        label_rect = label.get_rect(center=(position[0], position[1] + font_size // 2 + 12))
        screen.blit(label, label_rect)

    def draw_operation(self, screen, operation: Operation, position):
        """绘制操作/工件 - 使用 Emoji 图标"""
        color = self.OPERATION_STATE_COLOR.get(operation.status, self.BLACK)
        
        # 根据操作状态选择不同的 Emoji
        emoji_map = {
            OperationStatus.WAITING: "⏸️",     # 暂停 - 等待
            OperationStatus.READY: "📦",       # 包裹 - 就绪
            OperationStatus.MOVING: "📦",      # 运输中
            OperationStatus.WORKING: "📦",     # 锤子 - 加工中
            OperationStatus.FINISHED: "✅",    # 对勾 - 完成
            OperationStatus.EXCEPTION: "❗",   # 感叹号 - 异常
        }
        
        emoji = emoji_map.get(operation.status, "📋")  # 默认剪贴板
        
        # 计算字体大小（操作图标较小）
        font_size = max(int(0.22 * self.overall_scale), 14)
        
        # 创建字体对象并渲染 Emoji
        try:
            op_font = pygame.font.SysFont("segoeuiemoji, apple color emoji, noto color emoji, symbola, arial", font_size)
            emoji_surface = op_font.render(emoji, True, color)
        except:
            # 降级方案：使用彩色小方块
            rect_size = int(0.16 * self.overall_scale)
            rect = pygame.Rect(position[0], position[1], rect_size, rect_size)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, self.BLACK, rect, 1)  # 细边框
            
            # 即使在降级模式下也显示 ID
            label_font_size = max(int(0.14 * self.overall_scale), 10)
            label_font = pygame.font.SysFont(None, label_font_size)
            label = label_font.render(str(operation.id), True, self.WHITE)
            label_rect = label.get_rect(center=(position[0] + rect_size // 2, position[1] + rect_size + label_font_size // 2))
            screen.blit(label, label_rect)
            return
        
        # 将 Emoji 居中绘制
        emoji_rect = emoji_surface.get_rect(center=position)
        screen.blit(emoji_surface, emoji_rect)
        
        # 绘制操作 ID 标签（始终显示）
        label_font_size = max(int(0.14 * self.overall_scale), 10)
        label_font = pygame.font.SysFont(None, label_font_size)
        label = label_font.render(str(operation.id), True, self.WHITE)
        label_rect = label.get_rect(center=(position[0], position[1] + font_size // 2 + 6))
        screen.blit(label, label_rect)

    def draw_point(self, screen, point):
        """绘制路径节点 - 使用小圆点"""
        pos = self.scaling(point)
        radius = int(0.06 * self.overall_scale)
        pygame.draw.circle(screen, self.BLACK, pos, radius)
        # 添加白色边框使其更明显
        pygame.draw.circle(screen, self.WHITE, pos, radius, 1)

    def draw_link(self, screen, point1, point2):
        """绘制路径连线 - 使用虚线表示可通行路径"""
        pos1 = self.scaling(point1)
        pos2 = self.scaling(point2)
        line_width = max(int(0.02 * self.overall_scale), 1)
        
        # 使用灰色虚线表示路径
        gray_color = (150, 150, 150)
        pygame.draw.line(screen, gray_color, pos1, pos2, line_width)

    def visualize_env(self, env=None):
        # 渲染
        if self.env is None and env is not None:
            self.env = env

        if self.env is None:
            LOGGER.error("请先初始化环境")

        if self.overall_scale is None or self.overall_shift is None:
            self.overall_scale, self.overall_shift = self.calculate_scale_and_shift(0, 0, self.WIDTH, self.HEIGHT)

        self.screen.fill(self.WHITE)

        graph = self.env.getGraph()
        for point in graph.points:
            self.draw_point(self.screen, (point.x, point.y))
        for link in graph.links:
            point1 = graph.get_point_by_id(link.point1)
            point2 = graph.get_point_by_id(link.point2)
            self.draw_link(self.screen, (point1.x, point1.y), (point2.x, point2.y))

        for machine in self.env.getMachines():
            self.draw_machine(self.screen, machine)
            for operation in machine.input_queue:
                self.draw_operation(self.screen, operation, self.scaling(machine.get_xy(), shift=self.MACHINE_INPUT_SHIFT))
            for operation in machine.output_queue:
                self.draw_operation(self.screen, operation, self.scaling(machine.get_xy(), shift=self.MACHINE_OUTPUT_SHIFT))

        for agv in self.env.getAGVs():
            self.draw_agv(self.screen, agv)
            agv_operation = agv.get_operation()
            if agv_operation is not None:
                self.draw_operation(self.screen, agv_operation, self.scaling(agv.get_xy(), shift=self.AGV_OPERATION_SHIFT))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def restart(self):
        self.should_restart = True

    def shouldRestart(self) -> bool:
        """
        :return: True if env should restart
        """
        should_restart = self.should_restart
        self.should_restart = False
        return should_restart
    
    def pause(self):
        self.should_pause = True

    def shouldPause(self) -> bool:
        """
        :return: True if env should pause
        """
        should_pause = self.should_pause
        self.should_pause = False
        return should_pause
    
    def run(self):
        self.should_run = True

    def shouldRun(self)->bool:
        """
        :return: True if env should running
        """
        should_run = self.should_run
        self.should_run = False
        return should_run
    
    def change_speed(self, speed: int):
        self.fps = speed

    def pause_agv(self, agv_id: int):
        """暂停指定AGV运行"""
        for agv in self.env.getAGVs():
            if agv.get_id() == agv_id:
                self.insertNewUncertaintyEvent(f"AGV {agv_id} paused!")
                if agv.status == AGVStatus.EXCEPTION or agv in self.agv_pause_queue:
                    break
                else:
                    self.agv_pause_queue.append(agv)
                    break

    def getPausedAGVs(self) -> List[AGV]:
        """
        :return: 距离上次调用, 哪些AGV被暂停了
        """
        agv_pause_queue = self.agv_pause_queue
        self.agv_pause_queue = []
        return agv_pause_queue

    def resume_agv(self, agv_id: int):
        """恢复指定AGV运行"""
        for agv in self.env.getAGVs():
            if agv.get_id() == agv_id:
                self.agv_resume_queue.append(agv)
                self.insertNewUncertaintyEvent(f"AGV {agv_id} resumed!")
                break

    def getResumedAGVs(self) -> List[AGV]:
        """
        :return: 距离上次调用, 哪些AGV被恢复运行了
        """
        agv_resume_queue = self.agv_resume_queue
        self.agv_resume_queue = []
        return agv_resume_queue

    def pause_machine(self, machine_id: int):
        """暂停指定 Machine"""
        for machine in self.env.getMachines():
            if machine.get_id() == machine_id:
                self.insertNewUncertaintyEvent(f"Machine {machine_id} paused!")
                if machine.status == MachineStatus.FAILED or machine in self.machine_pause_queue:
                    break
                else:
                    self.machine_pause_queue.append(machine)
                    break

    def getPausedMachines(self) -> List[Machine]:
        """
        :return: 距离上次调用, 哪些Machine被暂停了
        """
        machine_pause_queue = self.machine_pause_queue
        self.machine_pause_queue = []
        return machine_pause_queue

    def resume_machine(self, machine_id: int):
        """恢复指定 Machine"""
        for machine in self.env.getMachines():
            if machine.get_id() == machine_id:
                self.machine_resume_queue.append(machine)
                self.insertNewUncertaintyEvent(f"Machine {machine_id} resumed!")
                break

    def getResumedMachines(self) -> List[Machine]:
        """
        :return: 距离上次调用, 哪些Machine被恢复运行了
        """
        machine_resume_queue = self.machine_resume_queue
        self.machine_resume_queue = []
        return machine_resume_queue

    def add_job(self, job_id: int):
        """添加指定 Job"""
        for job in self.env.getJobs():
            if job.get_id() == job_id:
                self.job_add_queue.append(job.clone())
                self.insertNewUncertaintyEvent(f"Job {job_id} added!")
                break

    def getAddedJobs(self) -> List[Job]:
        """
        :return: 距离上次调用, 哪些Job被添加了
        """
        job_add_queue = self.job_add_queue
        self.job_add_queue = []
        return job_add_queue

    def insertNewUncertaintyEvent(self, event: str):
        self.uncertainty_event_queue.append(event)

    def getNewUncertaintyEvents(self) -> List[str]:
        """
        :return: 距离上次调用, 哪些新的不确定性事件发生了, 以字符串形式format后传递, 最后将由visualizer直接显示
        """
        result = self.uncertainty_event_queue
        self.uncertainty_event_queue = []
        return result

    def getBuffered(self):
        """
        :return: [
                {
                    event_type: "packet_factory.ENV_PAUSED" / "packet_factory.AGV_FAIL" / ...
                    event_method: "trigger" / "recover"
                    type: "" / "AGV" / "Machine" / "Job" 
                    data: None / AGV / Machine / Job 实例
                }
                ]
        """
        result = []

        if self.shouldRestart():
            result.append({
                "event_type": "packet_factory.ENV_RESTART",
                "event_method": "trigger",
                "type": "",
                "data": None
            })

        if self.shouldPause():
            result.append({
                "event_type": "packet_factory.ENV_PAUSED",
                "event_method": "trigger",
                "type": "",
                "data": None
            })

        if self.shouldRun():
            result.append({
                "event_type": "packet_factory.ENV_RECOVER",
                "event_method": "trigger",
                "type": "",
                "data": None
            })


        for paused_agv in self.getPausedAGVs():
            result.append({
                "event_type": "packet_factory.AGV_FAIL",
                "event_method": "trigger",
                "type": "AGV",
                "data": paused_agv,
            })

        for resumed_agv in self.getResumedAGVs():
            result.append({
                "event_type": "packet_factory.AGV_FAIL",
                "event_method": "recover",
                "type": "AGV",
                "data": resumed_agv,
            })

        for paused_machine in self.getPausedMachines():
            result.append({
                "event_type": "packet_factory.MACHINE_FAIL",
                "event_method": "trigger",
                "type": "MACHINE",
                "data": paused_machine,
            })

        for resumed_machine in self.getResumedMachines():
            result.append({
                "event_type": "packet_factory.MACHINE_FAIL",
                "event_method": "recover",
                "type": "MACHINE",
                "data": resumed_machine,
            })

        for added_job in self.getAddedJobs():
            result.append({
                "event_type": "packet_factory.JOB_ADD",
                "event_method": "trigger",
                "type": "JOB",
                "data": added_job,
            })

        return result
