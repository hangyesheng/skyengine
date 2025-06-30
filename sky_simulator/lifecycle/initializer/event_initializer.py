from sky_simulator.registry.factory import create_component_by_id

from sky_simulator.event.event_manager.EventManager import EventManager

def initialize_event(config):
    event_config = config.get(config.get("env_type")).get("event_config")
    event_manager=EventManager()
    event_queue="todo"
    return event_queue