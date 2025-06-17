# 一定需要返回jobs, machines, agvs这三个类型的列表

import dataset
from typing import List, Tuple
from sky_simulator.packet_factory.packet_factory_env.Graph.Machine import Machine
from sky_simulator.packet_factory.packet_factory_env.Graph.AGV import AGV
from sky_simulator.packet_factory.packet_factory_env.Graph.util import OperationStatus
from sky_simulator.packet_factory.packet_factory_env.Graph.Operation import Operation
from sky_simulator.packet_factory.packet_factory_env.Graph.Job import Job

def read_agv_instance_data(relative_file_path):
    prefix = dataset.AGV_DATA_DIR
    file_path = prefix + relative_file_path
    file_content = open(file_path, 'r').read()
    data = file_content.split()
    idx = 0

    n = int(data[idx])
    idx += 1
    m = int(data[idx])
    idx += 1
    k = int(data[idx])
    idx += 1

    jobs: List[Job] = []
    operation_count = 0
    for job_id in range(n):
        num_operations = int(data[idx])
        idx += 1
        operations: List[Operation] = []
        for _ in range(num_operations):
            num_machines = int(data[idx])
            idx += 1
            durations: List[Tuple[int, float]] = []
            for _ in range(num_machines):
                machine_id = int(data[idx])
                idx += 1
                duration = float(data[idx])
                idx += 1
                durations.append((machine_id, duration))
            operations.append(Operation(operation_count, OperationStatus.WAITING, durations))
            operation_count += 1
        jobs.append(Job(job_id, operations))

    machines: List[Machine] = []
    for i in range(m):
        x = float(data[idx])
        idx += 1
        y = float(data[idx])
        idx += 1
        machines.append(Machine(i, x, y))

    agvs: List[AGV] = []
    for i in range(k):
        x = float(data[idx])
        idx += 1
        y = float(data[idx])
        idx += 1
        velocity = float(data[idx])
        idx += 1
        agvs.append(AGV(i, x, y, velocity))

    return jobs, machines, agvs
