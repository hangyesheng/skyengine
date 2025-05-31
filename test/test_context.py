'''
@Project ：tiangong 
@File    ：test_context.py
@IDE     ：PyCharm 
@Author  ：Skyrim
@Date    ：2025/5/31 23:01 
'''


from sky_simulator.lifecycle import context_creator
from sky_simulator.registry import component_registry,scan_and_register_components
from sky_simulator.utils import load_config
from config import PF_DIR
config = load_config(f"{PF_DIR}/sample.yaml")

if __name__ == '__main__':
    scan_and_register_components()
    context_creator.create_context(config)
