

from sky_simulator.event import BaseEvent,EventTest,EventAgvFail

if __name__ == '__main__':
    event=BaseEvent()
    event_test=EventTest(status='trigger',payload={'DATA':233})
    event_test_recover=EventTest(status='recover',payload={'DATA':233})
    event()
    event_test()
    event_test_recover()
    print(EventTest.event_type)



