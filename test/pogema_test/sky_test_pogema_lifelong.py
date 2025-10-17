'''
@Project ：SkyEngine 
@File    ：sky_test_pogema_lifelong.py
@IDE     ：PyCharm
@Author  ：Skyrimforest
@Date    ：2025/10/17 20:13
'''
from sky_executor.grid_factory.factory.grid_factory_env.Utils.pic_drawer import draw_svg

'''
@Project ：SkyEngine 
@File    ：sky_test_pogema_lifelong.py
@IDE     ：PyCharm
@Author  ：Skyrimforest
@Date    ：2025/10/17 20:13
'''

import numpy as np
from pogema import GridConfig
from pogema.envs import PogemaLifeLong

def test_pogema_lifelong():
    """
    测试 PogemaLifeLong 环境
    """
    # ---------- 1. 配置环境 ----------
    num_agents = 3
    size = 10

    # 自定义可能目标位置 (比如 Machine 位置)
    possible_targets_xy = [
        (2, 2),
        (7, 7),
        (5, 1),
        (8, 3)
    ]

    grid_config = GridConfig(
        num_agents=num_agents,
        size=size,
        obs_radius=2,
        max_episode_steps=20,
        targets_xy=None,  # 不提供固定序列
        possible_targets_xy=possible_targets_xy,
        seed=42
    )

    # ---------- 2. 初始化 PogemaLifeLong ----------
    from pogema import AnimationMonitor

    env = PogemaLifeLong(grid_config)
    env = AnimationMonitor(env)

    # ---------- 3. 重置环境 ----------
    obs, info = env.reset(seed=42)
    print("重置后的观测:")
    print(obs)
    print("初始 info:")
    for idx, inf in enumerate(info):
        print(f"Agent {idx}: {inf}")

    # ---------- 4. 获取初始 agent 与目标位置 ----------
    print("初始目标位置:")
    for idx in range(num_agents):
        goal = env.grid.finishes_xy[idx]
        print(f"Agent {idx} target: {goal}")

    # ---------- 5. 随机移动 agents，模拟几个 step ----------
    for step in range(5):
        # 随机动作: 0-4，假设上、下、左、右、停
        actions = [np.random.randint(0, 5) for _ in range(num_agents)]
        obs, rewards, terminated, truncated, infos = env.step(actions)
        print(f"\nStep {step + 1}:")
        print("Actions:", actions)
        print("Rewards:", rewards)
        for idx, inf in enumerate(infos):
            print(f"Agent {idx} info:", inf)
        # 打印当前目标位置
        print("当前目标位置:")
        for idx in range(num_agents):
            goal = env.grid.finishes_xy[idx]
            print(f"Agent {idx} target: {goal}")

    # ---------- 6. 渲染环境 ----------
    print("\n渲染环境 (SVG)")
    svg_str = draw_svg(env,1)
    print(svg_str[:500], "...")  # 只打印前500字符预览

if __name__ == "__main__":
    test_pogema_lifelong()
