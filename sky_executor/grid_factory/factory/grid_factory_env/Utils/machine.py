'''
@Project ：SkyEngine 
@File    ：machine.py
@IDE     ：PyCharm
@Author  ：Skyrimforest
@Date    ：2025/10/17 15:13
'''
import random
from typing import List, Tuple
from pydantic import BaseModel


class Machine:
    """Machine 逻辑节点"""
    def __init__(self, machine_id: int, location: Tuple[int, int]):
        self.id = machine_id
        self.location = location

    def __repr__(self):
        return f"Machine(id={self.id}, location={self.location})"


class MachineConfig(BaseModel):
    """Machine 配置"""
    num_machines: int = 5
    strategy: str = "random"
    seed: int = 42
    zones: int = 4
    grid_spacing: int = 5
    noise: float = 1.0


def generate_machines(grid: List[List[int]], machine_config: MachineConfig) -> List[Machine]:
    """在 Pogema 地图上生成机器节点"""
    cfg = machine_config.dict()  # ✅ pydantic BaseModel 转普通 dict

    num_machines = cfg["num_machines"]
    strategy = cfg["strategy"]
    seed = cfg["seed"]
    zones = cfg["zones"]
    grid_spacing = cfg["grid_spacing"]
    noise = cfg["noise"]

    random.seed(seed)

    height = len(grid)
    width = len(grid[0]) if height else 0
    if not height or not width:
        raise ValueError("Grid is empty")

    empty_cells = [(r, c) for r in range(height) for c in range(width) if grid[r][c] == 0]
    if not empty_cells:
        raise ValueError("No empty cells to place machines.")

    selected = []

    # --- 策略选择 ---
    if strategy == "random":
        random.shuffle(empty_cells)
        selected = empty_cells[:num_machines]

    elif strategy == "grid":
        n_side = int(num_machines ** 0.5) + 1
        for i in range(n_side):
            for j in range(n_side):
                if len(selected) >= num_machines:
                    break
                x, y = i * grid_spacing, j * grid_spacing
                if 0 <= x < height and 0 <= y < width and grid[x][y] == 0:
                    selected.append((x, y))
        if len(selected) < num_machines:
            remaining = [c for c in empty_cells if c not in selected]
            random.shuffle(remaining)
            selected += remaining[:num_machines - len(selected)]

    elif strategy == "grid+noise":
        n_side = int(num_machines ** 0.5) + 1
        for i in range(n_side):
            for j in range(n_side):
                if len(selected) >= num_machines:
                    break
                x = int(round(i * grid_spacing + random.uniform(-noise, noise)))
                y = int(round(j * grid_spacing + random.uniform(-noise, noise)))
                x = max(0, min(x, height - 1))
                y = max(0, min(y, width - 1))
                if grid[x][y] == 0 and (x, y) not in selected:
                    selected.append((x, y))
        if len(selected) < num_machines:
            remaining = [c for c in empty_cells if c not in selected]
            random.shuffle(remaining)
            selected += remaining[:num_machines - len(selected)]

    elif strategy == "zones":
        zone_h = max(1, height // zones)
        zone_w = max(1, width // zones)
        for i in range(zones):
            for j in range(zones):
                if len(selected) >= num_machines:
                    break
                r0, r1 = i * zone_h, min((i + 1) * zone_h, height)
                c0, c1 = j * zone_w, min((j + 1) * zone_w, width)
                candidates = [(r, c) for r in range(r0, r1)
                              for c in range(c0, c1)
                              if grid[r][c] == 0 and (r, c) not in selected]
                random.shuffle(candidates)
                if candidates:
                    selected.append(candidates[0])
        if len(selected) < num_machines:
            remaining = [c for c in empty_cells if c not in selected]
            random.shuffle(remaining)
            selected += remaining[:num_machines - len(selected)]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return [Machine(i, loc) for i, loc in enumerate(selected[:num_machines])]


if __name__ == "__main__":
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]

    config = MachineConfig(num_machines=4, strategy="grid+noise", seed=123, noise=1.5)
    machines = generate_machines(grid, config)
    print(machines)
