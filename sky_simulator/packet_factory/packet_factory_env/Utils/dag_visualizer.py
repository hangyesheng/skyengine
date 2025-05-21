import pygame
import json
import random
import sys

# 配置
WIDTH, HEIGHT = 800, 600
NODE_RADIUS = 25
FONT_SIZE = 14
GRID_SIZE = 5

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
GREEN = (0, 200, 0)
BLUE = (0, 100, 255)
ORANGE = (255, 165, 0)
RED = (200, 0, 0)

# Node的状态颜色映射
STATE_COLOR = {
    "free": GRAY,
    "pending":BLUE,
    "running": ORANGE,
    "finished": GREEN,
    "failed": RED,
}

def draw_arrow(surface, start, end, color=BLACK, width=2):
    pygame.draw.line(surface, color, start, end, width)
    # 箭头头部
    dx, dy = end[0]-start[0], end[1]-start[1]
    length = max((dx**2 + dy**2)**0.5, 0.001)
    unit_dx, unit_dy = dx/length, dy/length
    arrow_size = 10
    left = (end[0] - arrow_size * (unit_dx + unit_dy),
            end[1] - arrow_size * (unit_dy - unit_dx))
    right = (end[0] - arrow_size * (unit_dx - unit_dy),
             end[1] - arrow_size * (unit_dy + unit_dx))
    pygame.draw.polygon(surface, color, [end, left, right])


def visualize_env(env, scale=100, fps=1):
    pygame.init()
    width, height = env.grid_size[0] * scale, env.grid_size[1] * scale
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("SkyDagEnv Visualizer")
    clock = pygame.time.Clock()

    running = True
    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 绘制连接线
        for node in env.nodes.values():
            x1, y1 = node.position[0] * scale, node.position[1] * scale
            for edge in node.successors:
                x2, y2 = edge.target_node.position[0] * scale, edge.target_node.position[1] * scale
                draw_arrow(screen, (x1, y1), (x2, y2), GRAY)

        # 绘制节点
        for name, node in env.nodes.items():
            x, y = node.position[0] * scale, node.position[1] * scale
            color = {
                "idle": GRAY,
                "running": BLUE,
                "finished": GREEN
            }.get(node.state, RED)

            pygame.draw.circle(screen, color, (x, y), NODE_RADIUS)
            font = pygame.font.SysFont(None, 24)
            text = font.render(name, True, BLACK)
            screen.blit(text, (x - NODE_RADIUS, y - NODE_RADIUS - 18))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    sys.exit()

# 节点类
class UINode:
    def __init__(self, node_id, position):
        self.id = node_id
        self.position = position

# 可视化操作节点
class UIOperation:
    def __init__(self, op, assigned_node_id):
        self.op = op
        self.assigned_node_id = assigned_node_id

# 主类
class OverlayUnderlayVisualizer:
    def __init__(self, underlay_config, overlay_jobs):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Underlay-Overlay Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, FONT_SIZE)

        self.nodes = {n["id"]: UINode(n["id"], (n["position"][0] * 120 + 100, n["position"][1] * 100 + 100))
                      for n in underlay_config["nodes"]}
        self.links = underlay_config["links"]
        self.jobs = overlay_jobs
        self.job_index = 0
        self.frame_count = 0

        self.current_operations = self.extract_operations(self.jobs[self.job_index])

    def extract_operations(self, job_config):
        return [
            UIOperation(op, op.assigned_node.id if op.assigned_node else None)
            for op in job_config["operations"]
        ]

    def draw_text(self, text, pos, color=BLACK):
        label = self.font.render(text, True, color)
        self.screen.blit(label, pos)

    def draw_node(self, node_id, position):
        pygame.draw.circle(self.screen, BLUE, position, NODE_RADIUS)
        self.draw_text(node_id, (position[0] - 10, position[1] - 8), WHITE)

    def draw_link(self, from_pos, to_pos):
        pygame.draw.line(self.screen, BLACK, from_pos, to_pos, 2)

    def draw_operation(self, op: UIOperation):
        if op.assigned_node_id and op.assigned_node_id in self.nodes:
            node_pos = self.nodes[op.assigned_node_id].position
            op_color = STATE_COLOR.get(op.op.state, BLACK)
            pygame.draw.rect(self.screen, op_color, pygame.Rect(
                node_pos[0] - 10 + random.randint(-5, 5),
                node_pos[1] + 30 + random.randint(-5, 5),
                20, 10))
            self.draw_text(op.op.id, (node_pos[0] - 5, node_pos[1] + 42))

    def run(self):
        running = True
        while running:
            self.screen.fill(WHITE)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 画链接
            for link in self.links:
                from_pos = self.nodes[link["from"]].position
                to_pos = self.nodes[link["to"]].position
                self.draw_link(from_pos, to_pos)

            # 画节点
            for node in self.nodes.values():
                self.draw_node(node.id, node.position)

            # 更新 operation 状态（模拟）
            for op in self.current_operations:
                if op.op.state == "running":
                    op.op.step()
                elif op.op.state == "pending" and op.op.is_ready():
                    op.op.state = "running"

            # 画 operation
            for op in self.current_operations:
                self.draw_operation(op)

            # 模拟切换 job
            self.frame_count += 1
            if self.frame_count > 300:  # 每5秒切换
                self.job_index = (self.job_index + 1) % len(self.jobs)
                self.current_operations = self.extract_operations(self.jobs[self.job_index])
                self.frame_count = 0

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
