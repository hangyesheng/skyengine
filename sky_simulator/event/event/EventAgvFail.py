from sky_simulator.event.event.BaseEvent import BaseEvent
from sky_simulator.event.EventType import EventType
from sky_simulator.registry.registry import register_event


@register_event('packet_factory.AGV_FAIL')
class EventAgvFail(BaseEvent):
    event_type = EventType.AGV_FAIL

    def __init__(self,status:str="trigger",payload:dict=None):
        super().__init__(status,payload)
        assert payload is not None, "payload不能为None"
        assert 'agv_id' in payload, "payload必须包含agv_id字段"
        assert 'fail_time' in payload, "payload必须包含fail_time字段"

        self.agv_id=payload['agv_id']
        self.fail_time=payload['fail_time']

    def trigger(self,env):
        """
        触发该事件
        """
        print('EventTest.trigger()')


        return self.event_type

    def recover(self,env):
        """
        恢复该事件的现场
        """
        print('EventTest.trigger()')

        return self.event_type



