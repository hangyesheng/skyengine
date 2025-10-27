'''
@Project ：SkyEngine 
@File    ：test_solver.py
@IDE     ：PyCharm
@Author  ：Skyrimforest
@Date    ：2025/10/27 22:05
'''

import random
from sky_executor.grid_factory.factory.grid_factory_env.Utils.structure import Job, Machine, Operation
from sky_executor.grid_factory.factory.grid_factory_env.Component.JobSolver.template_solver.offline_solver import priority_greedy
from sky_executor.grid_factory.factory.grid_factory_env.Component.JobSolver.utils.draw_metrics import (plot_machine_gantt,
                                                                                                       plot_machine_util_bar)

if __name__ == "__main__":
    # build small instance: 4 machines, 6 jobs with 2 ops each
    machines = [Machine(i) for i in range(4)]
    jobs = []
    for j in range(6):
        p1 = random.randint(2, 6)
        p2 = random.randint(3, 7)
        ops = [Operation(job_id=j, op_id=0, machine_options=[0, 1], proc_time=p1),
               Operation(job_id=j, op_id=1, machine_options=[2, 3], proc_time=p2)]
        jobs.append(Job(job_id=j, ops=ops))

    # solve ignoring AGV (transfer_time_estimator returns 0)
    res = priority_greedy(jobs, machines, priority_rule="SPT", transfer_time_estimator=lambda a, b: 0.0)
    print("Makespan (greedy):", res.stats['makespan'])
    print("Transfer requests (to hand to AGV module):", res.transfer_requests)
    plot_machine_gantt(res.machine_schedule, machines)
    plot_machine_util_bar(res.stats)
