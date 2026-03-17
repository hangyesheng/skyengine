from executor.packet_factory.event.event_manager.EventManager import EventManager
from executor.packet_factory.registry.factory import create_component_by_id
import config as global_config
from pathlib import Path

def initialize_event_manager(config):
    event_config = config.get(config.get("env_type")).get('event_config')
    
    event_manager = EventManager()
    event_manager.load_event(event_config)

    return event_manager
