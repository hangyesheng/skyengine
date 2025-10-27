'''
@Project ：SkyEngine 
@File    ：offline_solver.py
@IDE     ：PyCharm
@Author  ：Skyrimforest
@Date    ：2025/10/27 22:11
'''
from pogema import GridConfig
from pogema import pogema_v0
import numpy as np
from sky_executor.grid_factory.factory.grid_factory_env.Component.BaseSolver import BaseSolver

from sky_executor.grid_factory.factory.grid_factory_env.Utils.structure import RouteProblem, RouteTask


class PogemaRouteSolver(BaseSolver):
    def __init__(self, map_name: str = "custom", policy_name: str = "astar", max_steps: int = 256):
        self.map_name = map_name
        self.policy_name = policy_name
        self.max_steps = max_steps

    def solve(self, problem: RouteProblem, **kwargs):
        """使用 Pogema 环境进行路径规划"""
        # 1. 构造网格配置
        grid_config = GridConfig(
            map_name=problem.map_name,
            num_agents=len(problem.tasks),
            max_episode_steps=problem.max_steps,
            obs_radius=5,
        )
        env = pogema_v0(grid_config)
        env.reset()

        # 2. 设置起点与目标
        starts = [task.start for task in problem.tasks]
        goals = [task.goal for task in problem.tasks]
        env.set_start_positions(starts)
        env.set_target_positions(goals)

        # 3. 初始化策略
        if self.policy_name == "astar":
            from pogema import AStarPolicy
            policy = AStarPolicy()
        elif self.policy_name == "random":
            from pogema import RandomPolicy
            policy = RandomPolicy()
        else:
            raise ValueError(f"Unsupported policy: {self.policy_name}")

        # 4. 路径规划执行
        obs, done = env.reset(), [False] * len(problem.tasks)
        trajectories = [[] for _ in problem.tasks]
        t = 0

        while not all(done) and t < problem.max_steps:
            actions = policy.act(obs)
            obs, _, done, _ = env.step(actions)
            positions = env.get_agents_xy()
            for i, pos in enumerate(positions):
                trajectories[i].append(pos)
            t += 1

        result = {
            "trajectories": trajectories,
            "steps": t,
            "success_rate": sum(done) / len(done),
        }
        return result

    def compute_metrics(self, result):
        """计算路径相关指标"""
        trajs = result["trajectories"]
        total_length = sum(len(tr) for tr in trajs)
        avg_length = np.mean([len(tr) for tr in trajs])
        return {
            "success_rate": result["success_rate"],
            "avg_length": avg_length,
            "total_length": total_length,
            "steps": result["steps"],
        }
