"""PacketFactory adaptation of Moon et al.'s Heterogeneous Graph Scheduler.

The paper's HGS uses operation, machine and vehicle nodes, a
structure-aware heterogeneous attention encoder, a three-stage O-M-V
decoder, and a makespan lower-bound-difference reward trained with
REINFORCE.  This module keeps those externally visible semantics while
reusing PacketFactory's graph builder and SAGE encoder as an explicit
engineering adaptation.  The differences are recorded in
``docs/thesis/literature/reproduction/HGS_复现卡.md``; this class must not be
reported as an exact reproduction of the original implementation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data

from .BaseAgent import BaseAgent, BACKEND, TRAINING, DEFAULT_STEP_TIME
from .device_utils import resolve_device, log_device
from .GraphGRPOAgent import FactoryGraphBuilder, GNNStateEncoder
from executor.packet_factory.packet_factory.packet_factory_env.Job.Operation import Operation
from executor.packet_factory.packet_factory.packet_factory_env.Machine.Machine import Machine
from executor.packet_factory.packet_factory.packet_factory_env.Agv.AGV import AGV
from executor.packet_factory.packet_factory.packet_factory_env.Job.Job import Job
from executor.packet_factory.packet_factory.packet_factory_env.Utils.util import (
    OperationStatus, AGVStatus,
)
from executor.packet_factory.logger.logger import LOGGER
from executor.packet_factory.registry import register_component


@dataclass
class HGSDecision:
    """One O-M-V action and the graph data needed for policy replay."""

    graph_data: Data
    node_type_offsets: Dict[str, Tuple[int, int]]
    operation_nodes: List[int]
    machine_nodes: List[int]
    agv_nodes: List[int]
    operation_action: int
    machine_action: int
    agv_action: int
    done: bool = False


class HGSRolloutBuffer:
    """Small on-policy buffer for terminal-return REINFORCE updates."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.transitions: List[HGSDecision] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.returns: Optional[np.ndarray] = None
        self.advantages: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.transitions)

    def push(self, decision: HGSDecision, reward: float, done: bool):
        self.transitions.append(decision)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def compute_returns(self, gamma: float):
        n = len(self.rewards)
        if n == 0:
            return
        returns = np.zeros(n, dtype=np.float32)
        running = 0.0
        for i in reversed(range(n)):
            if self.dones[i]:
                running = 0.0
            running = self.rewards[i] + gamma * running
            returns[i] = running
        std = float(returns.std())
        self.returns = returns
        self.advantages = (returns - float(returns.mean())) / (std + 1e-8)


class HGSDecoder(nn.Module):
    """Three-stage attention decoder over typed graph node embeddings."""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()
        )
        self.operation_query = nn.Linear(hidden_dim, hidden_dim)
        self.machine_query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.agv_query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.node_key = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def _score(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        q = query.unsqueeze(0) if query.dim() == 1 else query
        q = q.expand(candidates.size(0), -1)
        return (self.node_key(candidates) * q).sum(dim=-1) / self.scale

    def operation_logits(self, global_state: torch.Tensor,
                         operation_embeddings: torch.Tensor) -> torch.Tensor:
        ctx = self.context(global_state)
        return self._score(self.operation_query(ctx), operation_embeddings)

    def machine_logits(self, global_state: torch.Tensor,
                       operation_embedding: torch.Tensor,
                       machine_embeddings: torch.Tensor) -> torch.Tensor:
        ctx = self.context(global_state)
        query = self.machine_query(torch.cat([ctx, operation_embedding], dim=-1))
        return self._score(query, machine_embeddings)

    def agv_logits(self, global_state: torch.Tensor,
                   operation_embedding: torch.Tensor,
                   agv_embeddings: torch.Tensor) -> torch.Tensor:
        ctx = self.context(global_state)
        query = self.agv_query(torch.cat([ctx, operation_embedding], dim=-1))
        return self._score(query, agv_embeddings)


@register_component("packet_factory.HGSAdaptedAgent")
class HGSAdaptedAgent(BaseAgent):
    """HGS-style O-M-V scheduler adapted to PacketFactory."""

    def __init__(self, name=None, agent_id=None, context=None,
                 ui_mode: str = BACKEND, task_mode: str = TRAINING,
                 model_path: Optional[str] = None,
                 hidden_dim: int = 64, gamma: float = 1.0,
                 lr: float = 3e-4, entropy_coeff: float = 0.01,
                 max_grad_norm: float = 0.5, rollout_steps: int = 128,
                 device: Optional[str] = None, **kwargs):
        super().__init__(name, agent_id, context, ui_mode, task_mode, model_path)
        self.hidden_dim = hidden_dim
        self.state_dim = hidden_dim * 4
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.device = resolve_device(device, tag="HGSAdaptedAgent")
        log_device(self.device, tag="HGSAdaptedAgent")

        self.graph_builder = FactoryGraphBuilder(device=str(self.device))
        self.gnn_encoder = GNNStateEncoder(hidden_dim=hidden_dim, num_layers=2).to(self.device)
        self.decoder = HGSDecoder(self.state_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )
        self.rollout_buffer = HGSRolloutBuffer()
        self._rollout_step_count = 0
        self._prev_decision: Optional[HGSDecision] = None
        self._previous_lb: Optional[float] = None
        self._episode_reward = 0.0
        self._train_step = 0
        self.training_history = {
            'policy_loss': [], 'entropy': [], 'total_loss': [],
            'makespans': [], 'episodes': [],
        }

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    @staticmethod
    def _snapshot_graph(data: Data) -> Data:
        return Data(
            x=data.x.detach().cpu().clone(),
            node_type=data.node_type.detach().cpu().clone(),
            edge_index=data.edge_index.detach().cpu().clone(),
        )

    def _get_factory_graph(self, agvs):
        if agvs and hasattr(agvs[0], 'graph'):
            return agvs[0].graph
        if self.context and hasattr(self.context, 'graph'):
            return self.context.graph
        return None

    def _get_current_time(self) -> float:
        return float(getattr(self.context, 'env_timeline', 0.0)) if self.context else 0.0

    def _lower_bound(self, jobs: List[Job], current_time: float) -> float:
        remaining_by_job = []
        for job in jobs:
            remaining = 0.0
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() == OperationStatus.FINISHED:
                    continue
                if op.durations:
                    remaining += min(float(duration) for _, duration in op.durations)
            remaining_by_job.append(remaining)
        return float(current_time + (max(remaining_by_job) if remaining_by_job else 0.0))

    def _compute_reward(self, jobs: List[Job], current_time: float) -> float:
        current_lb = self._lower_bound(jobs, current_time)
        if self._previous_lb is None:
            self._previous_lb = current_lb
            return 0.0
        reward = self._previous_lb - current_lb
        self._previous_lb = current_lb
        return float(reward)

    def _candidate_entities(self, agvs, machines, jobs):
        ready_ops = [
            op for job in jobs if not job.is_finished()
            for i in range(job.get_operation_count())
            for op in [job.get_operation(i)]
            if op.get_status() == OperationStatus.READY
        ]
        return ready_ops

    def _evaluate_decision(self, decision: HGSDecision):
        graph = decision.graph_data.to(self.device)
        with torch.enable_grad():
            global_state, node_embeddings = self.gnn_encoder(
                graph, decision.node_type_offsets
            )
            op_nodes = torch.tensor(decision.operation_nodes, dtype=torch.long, device=self.device)
            machine_nodes = torch.tensor(decision.machine_nodes, dtype=torch.long, device=self.device)
            agv_nodes = torch.tensor(decision.agv_nodes, dtype=torch.long, device=self.device)
            op_embs = node_embeddings[op_nodes]
            op_logits = self.decoder.operation_logits(global_state, op_embs)
            op_dist = Categorical(logits=op_logits)
            op_action = min(decision.operation_action, len(decision.operation_nodes) - 1)
            op_log_prob = op_dist.log_prob(torch.tensor(op_action, device=self.device))

            selected_op = op_embs[op_action]
            machine_embs = node_embeddings[machine_nodes]
            machine_logits = self.decoder.machine_logits(global_state, selected_op, machine_embs)
            machine_dist = Categorical(logits=machine_logits)
            machine_action = min(decision.machine_action, len(decision.machine_nodes) - 1)
            machine_log_prob = machine_dist.log_prob(torch.tensor(machine_action, device=self.device))

            agv_embs = node_embeddings[agv_nodes]
            agv_logits = self.decoder.agv_logits(global_state, selected_op, agv_embs)
            agv_dist = Categorical(logits=agv_logits)
            agv_action = min(decision.agv_action, len(decision.agv_nodes) - 1)
            agv_log_prob = agv_dist.log_prob(torch.tensor(agv_action, device=self.device))
            entropy = (op_dist.entropy() + machine_dist.entropy() + agv_dist.entropy()) / 3.0
            return op_log_prob + machine_log_prob + agv_log_prob, entropy

    def _collect_previous(self, reward: float, done: bool):
        self._episode_reward += float(reward)
        if self._prev_decision is not None:
            self.rollout_buffer.push(self._prev_decision, reward, done)
            self._prev_decision = None

    def sample(self, agvs: List[AGV], machines: List[Machine],
               jobs: List[Job]):
        current_time = self._get_current_time()
        reward = self._compute_reward(jobs, current_time)
        ready_ops = self._candidate_entities(agvs, machines, jobs)

        if not ready_ops:
            done = all(job.is_finished() for job in jobs)
            self._collect_previous(reward, done)
            if done:
                self.alive = False
                return [], 0
            self._rollout_step_count += 1
            return [], DEFAULT_STEP_TIME

        factory_graph = self._get_factory_graph(agvs)
        graph_result = self.graph_builder.build(
            agvs, machines, jobs, factory_graph, current_time
        )
        graph_snapshot = self._snapshot_graph(graph_result.data)
        offsets = dict(graph_result.node_type_offsets)
        op_id_to_node = {
            op.id: graph_result.id_to_node_idx[('operation', op.id)]
            for op in ready_ops if ('operation', op.id) in graph_result.id_to_node_idx
        }
        operation_nodes = [op_id_to_node[op.id] for op in ready_ops if op.id in op_id_to_node]
        if not operation_nodes:
            self._collect_previous(reward, False)
            self._rollout_step_count += 1
            return [], DEFAULT_STEP_TIME

        # The first chosen operation determines the machine and AGV candidate
        # sets.  Candidate arrays are stored as graph node indices so replay
        # uses the exact mask observed during rollout.
        self.gnn_encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            global_state, node_embeddings = self.gnn_encoder(
                graph_result.data, graph_result.node_type_offsets
            )
            op_nodes_t = torch.tensor(operation_nodes, dtype=torch.long, device=self.device)
            op_logits = self.decoder.operation_logits(global_state, node_embeddings[op_nodes_t])
            op_dist = Categorical(logits=op_logits)
            op_action_t = op_dist.sample() if self.task_mode == TRAINING else torch.argmax(op_logits)
            selected_op = ready_ops[int(op_action_t.item())]

            capable_machines = [
                machine for machine in machines
                if selected_op.is_machine_capable(machine.id) and machine.is_available()
            ]
            available_agvs = [agv for agv in agvs if agv.get_status() == AGVStatus.READY]
            if not capable_machines or not available_agvs:
                self._collect_previous(reward, False)
                self._rollout_step_count += 1
                return [], DEFAULT_STEP_TIME

            machine_nodes = [graph_result.id_to_node_idx[('machine', m.id)] for m in capable_machines]
            machine_nodes_t = torch.tensor(machine_nodes, dtype=torch.long, device=self.device)
            machine_logits = self.decoder.machine_logits(
                global_state, node_embeddings[op_nodes_t[int(op_action_t.item())]],
                node_embeddings[machine_nodes_t]
            )
            machine_dist = Categorical(logits=machine_logits)
            machine_action_t = machine_dist.sample() if self.task_mode == TRAINING else torch.argmax(machine_logits)
            selected_machine = capable_machines[int(machine_action_t.item())]

            agv_nodes = [graph_result.id_to_node_idx[('agv', a.id)] for a in available_agvs]
            agv_nodes_t = torch.tensor(agv_nodes, dtype=torch.long, device=self.device)
            agv_logits = self.decoder.agv_logits(
                global_state, node_embeddings[op_nodes_t[int(op_action_t.item())]],
                node_embeddings[agv_nodes_t]
            )
            agv_dist = Categorical(logits=agv_logits)
            agv_action_t = agv_dist.sample() if self.task_mode == TRAINING else torch.argmax(agv_logits)
            selected_agv = available_agvs[int(agv_action_t.item())]

        self._collect_previous(reward, False)
        self._prev_decision = HGSDecision(
            graph_data=graph_snapshot,
            node_type_offsets=offsets,
            operation_nodes=operation_nodes,
            machine_nodes=machine_nodes,
            agv_nodes=agv_nodes,
            operation_action=int(op_action_t.item()),
            machine_action=int(machine_action_t.item()),
            agv_action=int(agv_action_t.item()),
        )
        self._rollout_step_count += 1
        return [(selected_op, selected_agv, selected_machine)], DEFAULT_STEP_TIME

    def reward(self, *args, **kwargs) -> float:
        # PacketFactory computes the learning transition reward internally;
        # this method remains available for the common Agent interface.
        return 0.0

    def flush_pending_transition(self, done: bool = True):
        """Commit the final O-M-V decision after an episode terminates.

        PacketFactory can finish immediately after applying the last action,
        so no subsequent ``sample`` call is available to collect the reward
        associated with ``_prev_decision``.  The shared training loop calls
        this hook for all agents that implement it.
        """
        if self.task_mode != TRAINING or self._prev_decision is None:
            return
        jobs = getattr(self.context, 'jobs', []) if self.context is not None else []
        reward = self._compute_reward(jobs, self._get_current_time())
        self._collect_previous(reward, bool(done))

    def train(self, *args, **kwargs):
        if self.task_mode != TRAINING or self._rollout_step_count < self.rollout_steps:
            return
        if len(self.rollout_buffer) == 0:
            self._rollout_step_count = 0
            return
        self.rollout_buffer.compute_returns(self.gamma)
        losses, entropies = [], []
        self.gnn_encoder.train()
        self.decoder.train()
        for decision, advantage in zip(
                self.rollout_buffer.transitions, self.rollout_buffer.advantages):
            log_prob, entropy = self._evaluate_decision(decision)
            advantage_t = torch.tensor(float(advantage), dtype=torch.float32, device=self.device)
            losses.append(-log_prob * advantage_t)
            entropies.append(entropy)
        if losses:
            policy_loss = torch.stack(losses).mean()
            entropy = torch.stack(entropies).mean()
            total_loss = policy_loss - self.entropy_coeff * entropy
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.gnn_encoder.parameters()) + list(self.decoder.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()
            self.training_history['policy_loss'].append(float(policy_loss.item()))
            self.training_history['entropy'].append(float(entropy.item()))
            self.training_history['total_loss'].append(float(total_loss.item()))
            self._train_step += 1
        self.rollout_buffer.reset()
        self._rollout_step_count = 0

    def new_episode(self):
        self._episode_reward = 0.0
        self._prev_decision = None
        self._previous_lb = None
        self.rollout_buffer.reset()
        self._rollout_step_count = 0
        self.training_history['episodes'].append(len(self.training_history['episodes']) + 1)

    def save_model(self, path: Optional[str] = None) -> bool:
        try:
            path = path or self.model_path or 'training_logs/models/HGSAdaptedAgent/agent_model.pt'
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            torch.save({
                'gnn_encoder': self.gnn_encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'hyperparams': {
                    'hidden_dim': self.hidden_dim, 'gamma': self.gamma,
                    'entropy_coeff': self.entropy_coeff,
                    'max_grad_norm': self.max_grad_norm,
                    'rollout_steps': self.rollout_steps,
                },
                'train_step': self._train_step,
                'training_history': self.training_history,
            }, path)
            return True
        except Exception as exc:
            LOGGER.error(f"[HGSAdaptedAgent] save failed: {exc}")
            return False

    def load_model(self, path: str) -> bool:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])
            self.decoder.load_state_dict(checkpoint['decoder'])
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self._train_step = checkpoint.get('train_step', 0)
            self.training_history.update(checkpoint.get('training_history', {}))
            return True
        except Exception as exc:
            LOGGER.error(f"[HGSAdaptedAgent] load failed: {exc}")
            return False

    def get_training_metrics(self) -> Dict[str, Any]:
        metrics = {
            'episode_reward': self._episode_reward,
            'training_history': self.training_history,
        }
        for key in ('policy_loss', 'entropy', 'total_loss'):
            if self.training_history.get(key):
                metrics[key] = self.training_history[key][-1]
        return metrics

