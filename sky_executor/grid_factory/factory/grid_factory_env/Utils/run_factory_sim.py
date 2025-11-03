'''
@Project ：SkyEngine 
@File    ：run_factory_sim.py
@IDE     ：PyCharm
@Author  ：Skyrimforest
@Date    ：2025/10/31 16:59
'''
from sky_executor.grid_factory.factory.grid_factory_env.grid_factory_env import GridFactoryEnv
from SkyEngine.sky_executor.grid_factory.factory.grid_factory_env.Component.JobSolver.utils.op_priority_greedy import priority_greedy


# === Step 1: 离线求解 ===
jobs = [...]  # 你的Job生成逻辑
machines = [...]  # 你的Machine配置
res = priority_greedy(jobs, machines, priority_rule="SPT")

# === Step 2: 环境初始化 ===
env = GridFactoryEnv()
env.machine_reset()
env.load_transfer_requests(res.transfer_requests)
obs, info = env.reset()

# === Step 3: 仿真循环 ===
for t in range(200):
    actions = {"agent_actions": {}, "machine_actions": {}}
    obs, rew, done, trunc, info = env.step(actions)
    env.render()
