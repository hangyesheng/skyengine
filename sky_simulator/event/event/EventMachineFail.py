from sky_simulator.event.event.BaseEvent import BaseEvent
from sky_simulator.event.EventType import EventType
from sky_simulator.packet_factory.packet_factory_env.Graph.Machine import Machine
from sky_simulator.packet_factory.packet_factory_env.packet_factory_env import PacketFactoryEnv
from sky_simulator.registry.registry import register_event


@register_event('packet_factory.MACHINE_FAIL')
class EventMachineFail(BaseEvent):
    event_type = EventType.MACHINE_FAIL
    def __init__(self,status:str="trigger",payload:dict=None):
        super().__init__(status,payload)

        assert payload is not None, "payload不能为None"
        assert 'machine_id' in payload, "payload必须包含machine_id字段"
        assert 'fail_time' in payload, "payload必须包含fail_time字段"

        self.machine_id=payload['machine_id']
        self.fail_time=payload['fail_time']

    def trigger(self,env:PacketFactoryEnv):
        """
        触发该事件
        """
        env.hash_index['machines'][self.machine_id]:Machine.machine_fail()
        return self.event_type

    def recover(self,env:PacketFactoryEnv):
        """
        恢复该事件的现场
        """
        print('EventMachineFail.trigger()')

        return self.event_type



