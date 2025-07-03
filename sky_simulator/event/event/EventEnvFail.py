'''
@Project ：tiangong 
@File    ：EventEnvFail.py
@IDE     ：PyCharm 
@Author  ：Skyrim
@Date    ：2025/7/2 17:29 
'''
from sky_simulator.event.event.BaseEvent import BaseEvent
from sky_simulator.event.EventType import EventType
from sky_simulator.registry.registry import register_event


@register_event('packet_factory.ENV_FAIL')
class EventEnvFail(BaseEvent):
    event_type = EventType.ENV_FAIL

    def __init__(self,status:str="trigger",payload:dict=None):
        super().__init__(status,payload)
        assert payload is not None, "payload不能为None"

    def trigger(self):
        """
        触发该事件
        """
        return self.event_type

    def recover(self):
        """
        恢复该事件的现场
        """
        return self.event_type