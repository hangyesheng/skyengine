from sky_simulator.packet_factory.env.Utils.dag_visualizer import visualize_env
from sky_dag_env import SkyDagEnv

# 已创建并 load_from_config 的环境
env = SkyDagEnv()
env.reset()

# 启动可视化（每秒更新一次）
visualize_env(env, scale=100, fps=1)
