# 测试事件动态启动 管理等功能
from sky_simulator.event import EventTest
from sky_simulator.lifecycle import context_creator
from sky_simulator.registry import component_registry,scan_and_register_components,create_component_by_id
from sky_simulator.registry.scanner import load_config
from sky_simulator.event.event_manager.EventManager import EventManager

config_path = '../../config/application_config.yaml'

if __name__ == '__main__':
    # load_config(config_path)
    # scan_and_register_components()
    # create_component_by_id()
    event_manager=EventManager()
    print(event_manager)
