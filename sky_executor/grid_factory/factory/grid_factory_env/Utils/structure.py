'''
@Project ：SkyEngine 
@File    ：structure.py.py
@IDE     ：PyCharm
@Author  ：Skyrimforest
@Date    ：2025/10/17 19:27
'''

# 此处列举了常用的结构

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable

# ========= 任务层  =========
@dataclass
class Operation:
    job_id: int
    op_id: int
    machine_options: List[int]
    proc_time: int
    release: float = 0
    due: Optional[float] = None

@dataclass
class Job:
    job_id: int
    ops: List[Operation]

@dataclass
class Machine:
    m_id: int

@dataclass
class JobSolverResult:
    machine_schedule: Dict[int, List[Tuple[float,float,int,int]]]
    op_meta: Dict[Tuple[int,int], Dict]
    transfer_requests: List[Dict]
    stats: Dict


# ========= 路由层  =========
@dataclass
class RouteTask:
    agv_id: int
    start: Tuple[int, int]
    goal: Tuple[int, int]

@dataclass
class RouteProblem:
    map_name: str
    tasks: List[RouteTask]
    policy_name: str = "astar"  # or 'ppo', 'rra', etc.
    max_steps: int = 256



