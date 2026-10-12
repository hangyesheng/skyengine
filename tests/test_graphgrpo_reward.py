import math
from types import SimpleNamespace

import numpy as np

from executor.packet_factory.packet_factory.Agent.GraphGRPOAgent import (
    GraphGRPOAgent,
    RunningMeanStd,
)
from executor.packet_factory.packet_factory.packet_factory_env.Job.Job import Job
from executor.packet_factory.packet_factory.packet_factory_env.Job.Operation import Operation
from executor.packet_factory.packet_factory.packet_factory_env.Utils.util import OperationStatus


def _agent_for_reward(mode="lower_bound_delta"):
    """Create the reward state without allocating neural networks."""
    agent = object.__new__(GraphGRPOAgent)
    agent.context = None
    agent.reward_mode = mode
    agent._previous_lower_bound = None
    agent._pending_lower_bound_reward = None
    agent._lower_bound_reward_count = 0
    agent._lower_bound_reward_sum = 0.0
    agent._episode_initial_lower_bound = None
    agent._episode_final_lower_bound = None
    agent._episode_reward = 0.0
    agent._previous_makespan = None
    agent._pending_step_reward = None
    agent._makespan_reward_count = 0
    agent._makespan_reward_sum = 0.0
    agent._episode_initial_makespan = None
    agent._episode_final_makespan = None
    return agent


def _job(duration=20.0):
    return Job(1, [Operation(1, OperationStatus.WAITING, [(1, duration)])])


def test_lower_bound_reward_telescopes_on_raw_values():
    agent = _agent_for_reward()
    job = _job()

    first = agent._compute_lower_bound_delta([job], 0.0)
    second = agent._compute_lower_bound_delta([job], 3.0)
    job.get_operation(0).set_status(OperationStatus.FINISHED)
    third = agent._compute_lower_bound_delta([job], 20.0)

    initial = agent._episode_initial_lower_bound
    final = agent._episode_final_lower_bound
    assert first == 0.0
    assert math.isclose(first + second + third, initial - final, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(agent._lower_bound_reward_sum, initial - final, rel_tol=0.0, abs_tol=1e-8)


def test_environment_reward_keeps_one_pending_transition_value():
    agent = _agent_for_reward()
    job = _job()
    env_info = {"makespan": 0.0, "jobs": [job], "machines": [], "agvs": []}

    assert agent.reward(env_info) == 0.0
    assert agent._pending_lower_bound_reward == 0.0

    env_info["makespan"] = 3.0
    value = agent.reward(env_info)
    assert value == -3.0
    assert agent._pending_lower_bound_reward == value
    assert agent._lower_bound_reward_count == 2


def test_first_action_is_included_when_rollout_commits_pending_reward():
    class CaptureBuffer:
        def __init__(self):
            self.records = []

        def push(self, **kwargs):
            self.records.append(kwargs)

    agent = _agent_for_reward()
    agent.task_mode = "training"
    agent.context = SimpleNamespace(env_timeline=0.0, jobs=[], machines=[], agvs=[])
    agent._rollout_step_count = 0
    agent._prev_global_state = None
    agent._prev_seq_features = None
    agent._prev_graph_data = None
    agent._prev_graph_node_type_offsets = None
    agent._prev_decisions_info = []
    agent._strict_group_id = None
    agent._strict_trajectory_id = None
    agent._strict_decision_step = 0
    agent.strict_group_sampling = False
    agent._grpo_group_id = 0
    agent.rollout_buffer = CaptureBuffer()
    agent.reward_normalizer = RunningMeanStd()

    job = _job()
    agent.context.jobs = [job]
    decision = {
        "seq_action": 0,
        "seq_log_prob": 0.0,
        "seq_candidate_count": 1,
        "route_action": -1,
        "route_log_prob": 0.0,
        "agv_action": -1,
        "agv_log_prob": 0.0,
    }

    # The first decision primes L(s_0) before any action is executed.
    agent._store_transition(
        np.zeros(4, dtype=np.float32), np.zeros(2, dtype=np.float32), [decision],
        None, None, None,
    )
    assert math.isclose(agent._episode_initial_lower_bound, 20.0)

    agent.context.env_timeline = 3.0
    reward = agent.reward({"makespan": 3.0, "jobs": [job], "machines": [], "agvs": []})
    assert reward == -3.0
    agent._store_transition(
        np.ones(4, dtype=np.float32), np.zeros(2, dtype=np.float32), [],
        None, None, None,
    )

    assert agent._pending_lower_bound_reward is None
    assert math.isclose(agent._lower_bound_reward_sum, -3.0)
    assert len(agent.rollout_buffer.records) == 1
    # Objective-aligned rewards stay on their raw scale until GRPO returns
    # are assembled; per-step normalization would erase the telescoping
    # interpretation for mostly unit-duration time advances.
    assert math.isclose(agent.rollout_buffer.records[0]["reward"], -3.0)


def test_makespan_delta_reward_telescopes_to_negative_cmax():
    agent = _agent_for_reward("makespan_delta")

    assert agent._compute_makespan_delta(0.0) == 0.0
    assert agent._compute_makespan_delta(3.0) == -3.0
    assert agent._compute_makespan_delta(20.0) == -17.0

    assert math.isclose(agent._makespan_reward_sum, -20.0)
    assert math.isclose(agent._episode_initial_makespan, 0.0)
    assert math.isclose(agent._episode_final_makespan, 20.0)


def test_resource_bound_delta_keeps_lower_bound_accounting_path():
    agent = _agent_for_reward("resource_bound_delta")
    job = _job()

    first = agent._compute_lower_bound_delta([job], 0.0, machines=[], agvs=[])
    second = agent._compute_lower_bound_delta([job], 5.0, machines=[], agvs=[])

    assert first == 0.0
    assert math.isclose(first + second,
                        agent._episode_initial_lower_bound
                        - agent._episode_final_lower_bound,
                        rel_tol=0.0, abs_tol=1e-8)

