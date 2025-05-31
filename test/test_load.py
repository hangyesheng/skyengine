'''
@Project ：tiangong 
@File    ：test_load.py
@IDE     ：PyCharm 
@Author  ：Skyrim
@Date    ：2025/5/31 1:30 
'''
from sky_simulator.utils import load_config
from config import PF_DIR
config = load_config(f"{PF_DIR}/template.yaml")

if __name__ == '__main__':

    print(config["env_type"])       # 'simulation'
    print(config["log_level"])      # 'INFO'
    print(config["agent_name"])     # 'RandomAgent'
