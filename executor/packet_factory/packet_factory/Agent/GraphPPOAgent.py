"""
GraphPPOAgent - GNN-enhanced PPO Agent for FJSP

Combines GNN-based state encoding (from GraphDualAgent) with PPO
(Proximal Policy Optimization) for more stable training:

Architecture:
  FactoryGraphBuilder -> GNNStateEncoder -> global_state [256] + node_embs [N, 64]
                                                |
                     +--------------------------+--------------------------+
                     |                          |                          |
              SequencingActor/Critic    RoutingActor/Critic    AGVSelectionActor/Critic
           (operation prioritization)  (machine assignment)       (AGV pick)

Key differences from GraphDualAgent (DQN):
- Actor-Critic architecture instead of online/target DQN pairs
- PPO clipped surrogate loss instead of MSE TD error
- On-policy RolloutBuffer with GAE instead of ReplayBuffer
- Stochastic policy sampling + entropy bonus instead of ε-greedy
- No target networks needed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from .BaseAgent import BaseAgent, DEFAULT_STEP_TIME, FRONTEND, BACKEND, TRAINING, INFERENCE
from .device_utils import resolve_device, log_device
from executor.packet_factory.packet_factory.packet_factory_env.Job.Operation import Operation
from executor.packet_factory.packet_factory.packet_factory_env.Machine.Machine import Machine
from executor.packet_factory.packet_factory.packet_factory_env.Agv.AGV import AGV
from executor.packet_factory.packet_factory.packet_factory_env.Job.Job import Job
from executor.packet_factory.logger.logger import LOGGER
from executor.packet_factory.packet_factory.packet_factory_env.Utils.util import (
    OperationStatus, MachineStatus, AGVStatus
)
from executor.packet_factory.registry import register_component

import numpy as np
import math
import os
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass
import time
import random


# ============================================================
# Data structures
# ============================================================

@dataclass
class GraphBuildResult:
    """Result of building a factory graph, with metadata for action encoding."""
    data: Data
    node_type_offsets: Dict[str, Tuple[int, int]]
    id_to_node_idx: Dict[Tuple[str, int], int]
    num_nodes: int


class RunningMeanStd:
    """Running mean/std for reward normalization (Welford's online algorithm)."""

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = len(x)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = max(new_var, 1e-6)
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# ============================================================
# Factory Graph Builder (reused from GraphDualAgent)
# ============================================================

class FactoryGraphBuilder:
    """Converts factory state into a PyG homogeneous Data graph with typed nodes.

    Node layout: [machines | operations | agvs | jobs]
    Node features are padded to a uniform max_dim so a single GNN input projection works.
    """

    MACHINE_DIM = 9
    OPERATION_DIM = 10
    AGV_DIM = 7
    JOB_DIM = 5
    MAX_DIM = 10  # pad all to this size

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def build(self, agvs: List[AGV], machines: List[Machine], jobs: List[Job],
              factory_graph=None, current_time: float = 0.0) -> GraphBuildResult:
        """Build graph from factory state."""
        machine_feats, machine_id_map = self._machine_features(machines, current_time)
        op_feats, op_id_map = self._operation_features(jobs)
        agv_feats, agv_id_map = self._agv_features(agvs)
        job_feats, job_id_map = self._job_features(jobs)

        n_m = len(machine_feats)
        n_o = len(op_feats)
        n_a = len(agv_feats)
        n_j = len(job_feats)

        all_feats = np.concatenate([
            self._pad(machine_feats),
            self._pad(op_feats),
            self._pad(agv_feats),
            self._pad(job_feats),
        ], axis=0)

        node_type = np.concatenate([
            np.zeros(n_m, dtype=np.int64),
            np.ones(n_o, dtype=np.int64),
            np.full(n_a, 2, dtype=np.int64),
            np.full(n_j, 3, dtype=np.int64),
        ])

        data = Data(
            x=torch.tensor(all_feats, dtype=torch.float),
            node_type=torch.tensor(node_type, dtype=torch.long),
        )

        # Build edges
        src_list, dst_list = [], []

        # Operation <-> Machine (capability)
        for job in jobs:
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.id not in op_id_map:
                    continue
                op_idx = n_m + op_id_map[op.id]
                for m_id, _ in op.durations:
                    if m_id in machine_id_map:
                        m_idx = machine_id_map[m_id]
                        src_list.extend([op_idx, m_idx])
                        dst_list.extend([m_idx, op_idx])

        # Job <-> Operation (contains)
        for job in jobs:
            if job.id not in job_id_map:
                continue
            job_idx = n_m + n_o + n_a + job_id_map[job.id]
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.id in op_id_map:
                    op_idx = n_m + op_id_map[op.id]
                    src_list.extend([job_idx, op_idx])
                    dst_list.extend([op_idx, job_idx])

        # Operation -> Operation (sequence, directed)
        for job in jobs:
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                next_op = op.get_next_operation()
                if op.id in op_id_map and next_op and next_op.id in op_id_map:
                    src_list.append(n_m + op_id_map[op.id])
                    dst_list.append(n_m + op_id_map[next_op.id])

        # AGV <-> Machine (reachable)
        for agv in agvs:
            if agv.id not in agv_id_map:
                continue
            agv_idx = n_m + n_o + agv_id_map[agv.id]
            for m in machines:
                if m.id not in machine_id_map:
                    continue
                m_idx = machine_id_map[m.id]
                if factory_graph and hasattr(factory_graph, 'get_path'):
                    if not factory_graph.get_path(agv.point_id, m.point_id):
                        continue
                src_list.extend([agv_idx, m_idx])
                dst_list.extend([m_idx, agv_idx])

        if src_list:
            data.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)

        id_to_node_idx: Dict[Tuple[str, int], int] = {}
        for mid, idx in machine_id_map.items():
            id_to_node_idx[("machine", mid)] = idx
        for oid, idx in op_id_map.items():
            id_to_node_idx[("operation", oid)] = n_m + idx
        for aid, idx in agv_id_map.items():
            id_to_node_idx[("agv", aid)] = n_m + n_o + idx
        for jid, idx in job_id_map.items():
            id_to_node_idx[("job", jid)] = n_m + n_o + n_a + idx

        offsets = {
            'machine': (0, n_m),
            'operation': (n_m, n_o),
            'agv': (n_m + n_o, n_a),
            'job': (n_m + n_o + n_a, n_j),
        }

        data = data.to(self.device)
        return GraphBuildResult(
            data=data,
            node_type_offsets=offsets,
            id_to_node_idx=id_to_node_idx,
            num_nodes=n_m + n_o + n_a + n_j,
        )

    def _pad(self, feats: list) -> np.ndarray:
        if not feats:
            return np.zeros((1, self.MAX_DIM), dtype=np.float32)
        arr = np.array(feats, dtype=np.float32)
        if arr.shape[1] < self.MAX_DIM:
            arr = np.pad(arr, ((0, 0), (0, self.MAX_DIM - arr.shape[1])))
        return arr

    def _machine_features(self, machines, current_time):
        feats, id_map = [], {}
        for idx, m in enumerate(machines):
            id_map[m.id] = idx
            timer = m.timer / 1000.0 if m.status == MachineStatus.WORKING else current_time / 1000.0
            load = len(m.input_queue) / 20.0
            soh = [0.0] * 4
            soh[min(m.status.value, 3)] = 1.0
            avail = 1.0 if m.is_available() else 0.0
            feats.append([timer, load] + soh + [m.x / 100.0, m.y / 100.0, avail])
        return feats or [[0.0] * self.MACHINE_DIM], id_map

    def _operation_features(self, jobs):
        feats, id_map = [], {}
        for job in jobs:
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                id_map[op.id] = len(feats)
                durs = [d for _, d in op.durations] if op.durations else [0.0]
                dur_mean = float(np.mean(durs)) / 100.0
                dur_std = float(np.std(durs)) / 100.0 if len(durs) > 1 else 0.0
                dur_min = min(durs) / 100.0
                soh = [0.0] * 5
                soh[min(op.get_status().value, 4)] = 1.0
                remaining = 0
                nxt = op.get_next_operation()
                while nxt is not None:
                    remaining += 1
                    nxt = nxt.get_next_operation()
                progress = min(op.process_time / (dur_mean * 100.0 + 1e-5), 1.0)
                feats.append([dur_mean, dur_std, dur_min] + soh + [remaining / 10.0, progress])
        return feats or [[0.0] * self.OPERATION_DIM], id_map

    def _agv_features(self, agvs):
        feats, id_map = [], {}
        for idx, agv in enumerate(agvs):
            id_map[agv.id] = idx
            soh = [0.0] * 3
            soh[min(agv.get_status().value, 2)] = 1.0
            feats.append([agv.timer / 1000.0, agv.velocity / 10.0] + soh
                         + [agv.x / 100.0, agv.y / 100.0])
        return feats or [[0.0] * self.AGV_DIM], id_map

    def _job_features(self, jobs):
        feats, id_map = [], {}
        for idx, job in enumerate(jobs):
            id_map[job.id] = idx
            n_ops = job.get_operation_count()
            n_fin = sum(1 for i in range(n_ops)
                        if job.get_operation(i).get_status() == OperationStatus.FINISHED)
            progress = n_fin / n_ops if n_ops > 0 else 0.0
            rem_time = 0.0
            for i in range(n_ops):
                op = job.get_operation(i)
                if op.get_status() != OperationStatus.FINISHED and op.durations:
                    rem_time += op.get_duration(op.durations[0][0])
            rem_time /= 1000.0
            feats.append([progress, rem_time, n_ops / 20.0,
                          1.0 if job.is_finished() else 0.0, len(id_map) / 100.0])
        return feats or [[0.0] * self.JOB_DIM], id_map


# ============================================================
# GNN State Encoder (reused from GraphDualAgent)
# ============================================================

class GNNStateEncoder(nn.Module):
    """Heterogeneous-aware GNN that encodes factory graph into a global state vector.

    Uses type embeddings + type-specific projections, then SAGEConv message passing,
    then per-type mean pooling to produce a fixed-size global state vector.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2,
                 machine_dim: int = 10, operation_dim: int = 10,
                 agv_dim: int = 10, job_dim: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.type_embedding = nn.Embedding(4, hidden_dim)

        self.machine_proj = nn.Linear(machine_dim, hidden_dim)
        self.operation_proj = nn.Linear(operation_dim, hidden_dim)
        self.agv_proj = nn.Linear(agv_dim, hidden_dim)
        self.job_proj = nn.Linear(job_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, data: Data,
                node_type_offsets: Dict[str, Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        node_type = data.node_type

        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

        for type_idx, proj in enumerate([self.machine_proj, self.operation_proj,
                                          self.agv_proj, self.job_proj]):
            mask = (node_type == type_idx)
            if mask.any():
                h[mask] = F.leaky_relu(proj(x[mask]))

        type_embs = self.type_embedding(node_type)
        h = h + type_embs
        h = F.leaky_relu(h)

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, data.edge_index)
            h = norm(h + h_new)
            h = F.leaky_relu(h)

        parts = []
        for type_name in ['machine', 'operation', 'agv', 'job']:
            start, count = node_type_offsets[type_name]
            if count > 0:
                pooled = h[start:start + count].mean(dim=0)
            else:
                pooled = torch.zeros(self.hidden_dim, device=h.device)
            parts.append(pooled)

        global_state = torch.cat(parts, dim=-1)
        return global_state, h


# ============================================================
# Rollout Buffer (on-policy trajectory storage with GAE)
# ============================================================

class RolloutBuffer:
    """On-policy rollout buffer for PPO with per-sub-agent trajectory storage.

    Stores transitions collected during one rollout phase.
    After collection, compute_returns_and_advantages() calculates GAE-based
    advantages and returns for each sub-agent.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        """Clear all stored transitions."""
        # Shared
        self.states = []
        self.seq_features = []
        self.dones = []

        # Sequencing
        self.seq_actions = []
        self.seq_log_probs = []
        self.seq_values = []

        # Routing
        self.route_actions = []
        self.route_log_probs = []
        self.route_values = []
        self.route_features = []

        # AGV Selection
        self.agv_actions = []
        self.agv_log_probs = []
        self.agv_values = []
        self.agv_features = []

        # Rewards (shared across sub-agents per step)
        self.rewards = []

        # Computed after rollout
        self.seq_advantages = None
        self.seq_returns = None
        self.route_advantages = None
        self.route_returns = None
        self.agv_advantages = None
        self.agv_returns = None

    def __len__(self):
        return len(self.states)

    def push(self, state: np.ndarray, seq_features: np.ndarray,
             reward: float, done: bool,
             # Sequencing
             seq_action: int = -1, seq_log_prob: float = 0.0, seq_value: float = 0.0,
             # Routing
             route_action: int = -1, route_log_prob: float = 0.0, route_value: float = 0.0,
             route_feat: Optional[np.ndarray] = None,
             # AGV
             agv_action: int = -1, agv_log_prob: float = 0.0, agv_value: float = 0.0,
             agv_feat: Optional[np.ndarray] = None):
        """Store one transition step."""
        self.states.append(state)
        self.seq_features.append(seq_features)
        self.dones.append(done)
        self.rewards.append(reward)

        self.seq_actions.append(seq_action)
        self.seq_log_probs.append(seq_log_prob)
        self.seq_values.append(seq_value)

        self.route_actions.append(route_action)
        self.route_log_probs.append(route_log_prob)
        self.route_values.append(route_value)
        self.route_features.append(route_feat if route_feat is not None else np.zeros(1, dtype=np.float32))

        self.agv_actions.append(agv_action)
        self.agv_log_probs.append(agv_log_prob)
        self.agv_values.append(agv_value)
        self.agv_features.append(agv_feat if agv_feat is not None else np.zeros(1, dtype=np.float32))

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute GAE-based advantages and returns for all sub-agents.

        Since the three sub-agents share the same reward signal at each step,
        we compute one set of advantages from the shared rewards and values.

        Args:
            last_value: V(s_T) from the critic for the last state (bootstrap value)
            gamma: discount factor
            gae_lambda: GAE lambda parameter
        """
        n = len(self.rewards)
        if n == 0:
            return

        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # Use the average of all three critic values as the shared value estimate
        # This avoids privileging any single sub-agent's value function
        seq_values = np.array(self.seq_values, dtype=np.float32)
        route_values = np.array(self.route_values, dtype=np.float32)
        agv_values = np.array(self.agv_values, dtype=np.float32)
        values = (seq_values + route_values + agv_values) / 3.0

        # Append bootstrap value for the last state
        values = np.append(values, last_value)

        # Compute GAE
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:-1]

        # Normalize advantages (per rollout)
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Share the same advantages and returns across all sub-agents
        self.seq_advantages = advantages
        self.seq_returns = returns
        self.route_advantages = advantages
        self.route_returns = returns
        self.agv_advantages = advantages
        self.agv_returns = returns

    def get_tensors(self, sub_agent: str) -> Dict[str, torch.Tensor]:
        """Convert stored data to tensors for PPO update.

        Args:
            sub_agent: 'seq', 'route', or 'agv'

        Returns:
            Dictionary of torch tensors
        """
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        seq_feats = torch.tensor(np.array(self.seq_features), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(self.dones), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(self.rewards), dtype=torch.float32, device=self.device)

        if sub_agent == 'seq':
            actions = torch.tensor(np.array(self.seq_actions), dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor(np.array(self.seq_log_probs), dtype=torch.float32, device=self.device)
            advantages = torch.tensor(self.seq_advantages, dtype=torch.float32, device=self.device)
            returns = torch.tensor(self.seq_returns, dtype=torch.float32, device=self.device)
            return {
                'states': states, 'seq_features': seq_feats,
                'actions': actions, 'old_log_probs': old_log_probs,
                'advantages': advantages, 'returns': returns,
            }
        elif sub_agent == 'route':
            actions = torch.tensor(np.array(self.route_actions), dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor(np.array(self.route_log_probs), dtype=torch.float32, device=self.device)
            advantages = torch.tensor(self.route_advantages, dtype=torch.float32, device=self.device)
            returns = torch.tensor(self.route_returns, dtype=torch.float32, device=self.device)
            route_feats = torch.tensor(np.array(self.route_features), dtype=torch.float32, device=self.device)
            return {
                'states': states, 'route_features': route_feats,
                'actions': actions, 'old_log_probs': old_log_probs,
                'advantages': advantages, 'returns': returns,
            }
        elif sub_agent == 'agv':
            actions = torch.tensor(np.array(self.agv_actions), dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor(np.array(self.agv_log_probs), dtype=torch.float32, device=self.device)
            advantages = torch.tensor(self.agv_advantages, dtype=torch.float32, device=self.device)
            returns = torch.tensor(self.agv_returns, dtype=torch.float32, device=self.device)
            agv_feats = torch.tensor(np.array(self.agv_features), dtype=torch.float32, device=self.device)
            return {
                'states': states, 'agv_features': agv_feats,
                'actions': actions, 'old_log_probs': old_log_probs,
                'advantages': advantages, 'returns': returns,
            }
        else:
            raise ValueError(f"Unknown sub_agent: {sub_agent}")


# ============================================================
# Actor Networks (Policy Networks)
# ============================================================

class SequencingActor(nn.Module):
    """Sequencing Actor: outputs action logits for operation prioritization.

    Input: global_state [state_dim] + seq_features [seq_feature_dim]
    Output: logits [max_seq_output_dim] for categorical distribution over operation slots
    """

    def __init__(self, state_dim: int, seq_feature_dim: int, max_output_dim: int,
                 hidden: int = 128):
        super().__init__()
        input_dim = state_dim + seq_feature_dim
        self.state_norm = nn.InstanceNorm1d(1, affine=False)
        self.feat_norm = nn.InstanceNorm1d(1, affine=False)

        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 64)
        self.fc_out = nn.Linear(64, max_output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor,
                seq_features: torch.Tensor) -> torch.Tensor:
        gs = self.state_norm(global_state.unsqueeze(1)).squeeze(1)
        sf = self.feat_norm(seq_features.unsqueeze(1)).squeeze(1)
        x = torch.cat([gs, sf], dim=-1)

        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc_out(x)


class RoutingActor(nn.Module):
    """Routing Actor: outputs action logits for machine assignment.

    Input: global_state [state_dim] + route_features [route_feature_dim]
    Output: logits [max_route_output_dim] for categorical distribution over machine slots
    """

    def __init__(self, state_dim: int, route_feature_dim: int, max_output_dim: int,
                 hidden: int = 128):
        super().__init__()
        input_dim = state_dim + route_feature_dim
        self.state_norm = nn.InstanceNorm1d(1, affine=False)
        self.feat_norm = nn.InstanceNorm1d(1, affine=False)

        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, 64)
        self.fc_out = nn.Linear(64, max_output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor,
                route_features: torch.Tensor) -> torch.Tensor:
        gs = self.state_norm(global_state.unsqueeze(1)).squeeze(1)
        rf = self.feat_norm(route_features.unsqueeze(1)).squeeze(1)
        x = torch.cat([gs, rf], dim=-1)

        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = F.leaky_relu(self.fc2(x))
        return self.fc_out(x)


class AGVSelectionActor(nn.Module):
    """AGV Selection Actor: outputs action logits for AGV selection.

    Input: global_state [state_dim] + agv_features [agv_feature_dim]
    Output: logits [max_agv_output_dim] for categorical distribution over AGV slots
    """

    def __init__(self, state_dim: int, agv_feature_dim: int, max_output_dim: int,
                 hidden: int = 64):
        super().__init__()
        input_dim = state_dim + agv_feature_dim
        self.state_norm = nn.InstanceNorm1d(1, affine=False)
        self.feat_norm = nn.InstanceNorm1d(1, affine=False)

        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, 32)
        self.fc_out = nn.Linear(32, max_output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor,
                agv_features: torch.Tensor) -> torch.Tensor:
        gs = self.state_norm(global_state.unsqueeze(1)).squeeze(1)
        af = self.feat_norm(agv_features.unsqueeze(1)).squeeze(1)
        x = torch.cat([gs, af], dim=-1)

        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = F.leaky_relu(self.fc2(x))
        return self.fc_out(x)


# ============================================================
# Critic Networks (Value Networks)
# ============================================================

class SequencingCritic(nn.Module):
    """Sequencing Critic: estimates state value V(s) for the sequencing sub-problem.

    Input: global_state [state_dim] + seq_features [seq_feature_dim]
    Output: V(s) [1]
    """

    def __init__(self, state_dim: int, seq_feature_dim: int,
                 hidden: int = 128):
        super().__init__()
        input_dim = state_dim + seq_feature_dim
        self.state_norm = nn.InstanceNorm1d(1, affine=False)
        self.feat_norm = nn.InstanceNorm1d(1, affine=False)

        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, 64)
        self.fc_out = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor,
                seq_features: torch.Tensor) -> torch.Tensor:
        gs = self.state_norm(global_state.unsqueeze(1)).squeeze(1)
        sf = self.feat_norm(seq_features.unsqueeze(1)).squeeze(1)
        x = torch.cat([gs, sf], dim=-1)

        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = F.leaky_relu(self.fc2(x))
        return self.fc_out(x).squeeze(-1)


class RoutingCritic(nn.Module):
    """Routing Critic: estimates state value V(s) for the routing sub-problem.

    Input: global_state [state_dim] + route_features [route_feature_dim]
    Output: V(s) [1]
    """

    def __init__(self, state_dim: int, route_feature_dim: int,
                 hidden: int = 128):
        super().__init__()
        input_dim = state_dim + route_feature_dim
        self.state_norm = nn.InstanceNorm1d(1, affine=False)
        self.feat_norm = nn.InstanceNorm1d(1, affine=False)

        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, 64)
        self.fc_out = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor,
                route_features: torch.Tensor) -> torch.Tensor:
        gs = self.state_norm(global_state.unsqueeze(1)).squeeze(1)
        rf = self.feat_norm(route_features.unsqueeze(1)).squeeze(1)
        x = torch.cat([gs, rf], dim=-1)

        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = F.leaky_relu(self.fc2(x))
        return self.fc_out(x).squeeze(-1)


class AGVSelectionCritic(nn.Module):
    """AGV Selection Critic: estimates state value V(s) for the AGV selection sub-problem.

    Input: global_state [state_dim] + agv_features [agv_feature_dim]
    Output: V(s) [1]
    """

    def __init__(self, state_dim: int, agv_feature_dim: int,
                 hidden: int = 64):
        super().__init__()
        input_dim = state_dim + agv_feature_dim
        self.state_norm = nn.InstanceNorm1d(1, affine=False)
        self.feat_norm = nn.InstanceNorm1d(1, affine=False)

        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, 32)
        self.fc_out = nn.Linear(32, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor,
                agv_features: torch.Tensor) -> torch.Tensor:
        gs = self.state_norm(global_state.unsqueeze(1)).squeeze(1)
        af = self.feat_norm(agv_features.unsqueeze(1)).squeeze(1)
        x = torch.cat([gs, af], dim=-1)

        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = F.leaky_relu(self.fc2(x))
        return self.fc_out(x).squeeze(-1)


# ============================================================
# GraphPPOAgent
# ============================================================

@register_component("packet_factory.GraphPPOAgent")
class GraphPPOAgent(BaseAgent):
    """GNN-enhanced PPO Agent for FJSP.

    Combines GNN state encoding with PPO (Proximal Policy Optimization):
    1. GNN encodes factory graph -> global_state [256] + node_embs
    2. SequencingActor/Critic ranks READY operations by priority
    3. For each operation in priority order:
       a. RoutingActor/Critic selects the best capable machine
       b. AGVSelectionActor/Critic selects the best available AGV
    4. Training: on-policy PPO with clipped surrogate, GAE, entropy bonus
    """

    def __init__(self, name=None, agent_id=None, context=None,
                 ui_mode: str = BACKEND, task_mode: str = TRAINING,
                 model_path: Optional[str] = None,
                 # Graph encoder
                 hidden_dim: int = 64,
                 # PPO common
                 gamma: float = 0.99,
                 # Learning rates
                 lr_gnn: float = 1e-4,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 # PPO hyperparameters
                 clip_epsilon: float = 0.2,
                 gae_lambda: float = 0.95,
                 ppo_epochs: int = 4,
                 ppo_batch_size: int = 64,
                 entropy_coeff: float = 0.01,
                 value_loss_coeff: float = 0.5,
                 max_grad_norm: float = 0.5,
                 rollout_steps: int = 128,
                 # Output dimensions (max action space sizes)
                 max_seq_output_dim: int = 100,
                 max_route_output_dim: int = 50,
                 max_agv_output_dim: int = 20,
                 # Feature dimensions
                 seq_feature_dim: int = 24,
                 route_feature_dim: int = 16,
                 agv_feature_dim: int = 12,
                 # Misc
                 allow_agv_reassignment: bool = False,
                 device: Optional[str] = None,
                 **kwargs):
        super().__init__(name, agent_id, context, ui_mode, task_mode, model_path)

        self.hidden_dim = hidden_dim
        self.state_dim = hidden_dim * 4
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.max_seq_output_dim = max_seq_output_dim
        self.max_route_output_dim = max_route_output_dim
        self.max_agv_output_dim = max_agv_output_dim
        self.seq_feature_dim = seq_feature_dim
        self.route_feature_dim = route_feature_dim
        self.agv_feature_dim = agv_feature_dim
        self.allow_agv_reassignment = allow_agv_reassignment

        # Device
        self.device = resolve_device(device, tag="GraphPPOAgent")
        log_device(self.device, tag="GraphPPOAgent")

        # Graph builder
        self.graph_builder = FactoryGraphBuilder(device=str(self.device))

        # GNN encoder (shared across all Actor-Critic pairs)
        self.gnn_encoder = GNNStateEncoder(
            hidden_dim=hidden_dim, num_layers=2,
        ).to(self.device)

        # Initialize all Actor-Critic pairs
        self._initialize_networks(lr_gnn, lr_actor, lr_critic)

        # Rollout buffer (on-policy)
        self.rollout_buffer = RolloutBuffer(self.device)

        # Reward normalization
        self.reward_normalizer = RunningMeanStd()

        # Rollout step counter (for deciding when to update)
        self._rollout_step_count = 0

        # State tracking for transition collection
        self._prev_global_state: Optional[np.ndarray] = None
        self._prev_seq_features: Optional[np.ndarray] = None
        self._prev_decisions_info: List[Dict] = []

        # Training statistics
        self.training_history: Dict[str, list] = {
            'episodes': [], 'actor_loss': [], 'critic_loss': [],
            'entropy': [], 'total_loss': [], 'makespans': [],
            'kl_approx': [],
        }
        self._episode_reward = 0.0
        self._train_step = 0

        # Load model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            LOGGER.info(f"[GraphPPOAgent] Loaded model from {model_path}")
        else:
            LOGGER.info(f"[GraphPPOAgent] Initialized on {self.device}, "
                        f"hidden_dim={hidden_dim}, state_dim={self.state_dim}, "
                        f"clip_epsilon={clip_epsilon}, gae_lambda={gae_lambda}")

    def _initialize_networks(self, lr_gnn, lr_actor, lr_critic):
        """Initialize all Actor-Critic pairs and optimizers."""
        # Sequencing Actor-Critic
        self.seq_actor = SequencingActor(
            self.state_dim, self.seq_feature_dim, self.max_seq_output_dim,
        ).to(self.device)
        self.seq_critic = SequencingCritic(
            self.state_dim, self.seq_feature_dim,
        ).to(self.device)

        # Routing Actor-Critic
        self.route_actor = RoutingActor(
            self.state_dim, self.route_feature_dim, self.max_route_output_dim,
        ).to(self.device)
        self.route_critic = RoutingCritic(
            self.state_dim, self.route_feature_dim,
        ).to(self.device)

        # AGV Selection Actor-Critic
        self.agv_actor = AGVSelectionActor(
            self.state_dim, self.agv_feature_dim, self.max_agv_output_dim,
        ).to(self.device)
        self.agv_critic = AGVSelectionCritic(
            self.state_dim, self.agv_feature_dim,
        ).to(self.device)

        # Optimizers: each sub-agent has a joint optimizer for actor + critic + GNN
        # This mirrors GraphDualAgent's pattern where GNN is in all three optimizers
        self.optimizer_seq = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) +
            list(self.seq_actor.parameters()) +
            list(self.seq_critic.parameters()),
            lr=lr_actor,
        )
        self.optimizer_route = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) +
            list(self.route_actor.parameters()) +
            list(self.route_critic.parameters()),
            lr=lr_actor,
        )
        self.optimizer_agv = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) +
            list(self.agv_actor.parameters()) +
            list(self.agv_critic.parameters()),
            lr=lr_actor,
        )

        # Separate GNN optimizer for save/load state dict only (like GraphDualAgent)
        self.optimizer_gnn = torch.optim.Adam(
            self.gnn_encoder.parameters(), lr=lr_gnn,
        )

    # ----------------------------------------------------------
    # Feature Extraction (unchanged from GraphDualAgent)
    # ----------------------------------------------------------

    def _extract_seq_features(self, machines: List[Machine], jobs: List[Job],
                              current_time: float) -> np.ndarray:
        """Extract sequencing-specific features (24-dim).

        Captures system-level scheduling state for operation prioritization.
        """
        features = []

        # [0-1] READY / WAITING op counts
        total_ops = 0
        ready_count = 0
        waiting_count = 0
        ready_durations = []
        jobs_with_ready = 0
        remaining_ops_per_job = []

        for job in jobs:
            if job.is_finished():
                continue
            job_has_ready = False
            job_remaining = 0
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                total_ops += 1
                status = op.get_status()
                if status == OperationStatus.READY:
                    ready_count += 1
                    job_has_ready = True
                    if op.durations:
                        ready_durations.append(sum(d for _, d in op.durations) / len(op.durations))
                elif status == OperationStatus.WAITING:
                    waiting_count += 1
                if status != OperationStatus.FINISHED:
                    job_remaining += 1
            if job_has_ready:
                jobs_with_ready += 1
                remaining_ops_per_job.append(job_remaining)

        features.append(ready_count / 50.0)
        features.append(waiting_count / max(total_ops, 1))

        # [2-5] READY op duration stats
        if ready_durations:
            features.append(float(np.mean(ready_durations)) / 100.0)
            features.append(float(np.std(ready_durations)) / 100.0 if len(ready_durations) > 1 else 0.0)
            features.append(min(ready_durations) / 100.0)
            features.append(max(ready_durations) / 100.0)
        else:
            features.extend([0.0] * 4)

        # [6-7] Jobs with READY ops
        total_jobs = len(jobs) if jobs else 1
        features.append(jobs_with_ready / total_jobs)
        features.append(float(np.mean(remaining_ops_per_job)) / 10.0 if remaining_ops_per_job else 0.0)

        # [8-9] System urgency
        remaining_time_sum = 0.0
        total_remaining = 0
        for job in jobs:
            if job.is_finished():
                continue
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() != OperationStatus.FINISHED and op.durations:
                    remaining_time_sum += sum(d for _, d in op.durations) / len(op.durations)
                    total_remaining += 1
        features.append(remaining_time_sum / 1000.0)
        features.append(total_remaining / max(total_ops, 1))

        # [10-11] Machine queue stats
        if machines:
            queue_lens = [len(m.input_queue) for m in machines]
            features.append(float(np.mean(queue_lens)) / 20.0)
            features.append(float(max(queue_lens)) / 20.0)
        else:
            features.extend([0.0, 0.0])

        # [12-13] Machine utilization & availability
        if machines:
            working = sum(1 for m in machines if m.status == MachineStatus.WORKING)
            features.append(working / len(machines))
            available = sum(1 for m in machines if m.is_available())
            features.append(available / len(machines))
        else:
            features.extend([0.0, 0.0])

        # [14-15] Imminent completion times
        imminent = []
        for m in machines:
            if m.input_queue:
                head_op = m.input_queue[0]
                time_left = head_op.get_duration(m.id) - head_op.process_time
                imminent.append(max(0.0, time_left))
        if imminent:
            features.append(min(imminent) / 100.0)
            features.append(float(np.mean(imminent)) / 100.0)
        else:
            features.extend([0.0, 0.0])

        # [16-17] Load balance & job completion
        if machines:
            loads = [len(m.input_queue) for m in machines]
            if len(loads) > 1 and np.mean(loads) > 0:
                cv = float(np.std(loads)) / (float(np.mean(loads)) + 1e-5)
                features.append(cv)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        completed = sum(1 for j in jobs if j.is_finished())
        features.append(completed / total_jobs)

        # [18-23] Extended features
        waiting_durs = []
        for job in jobs:
            if job.is_finished():
                continue
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() == OperationStatus.WAITING and op.durations:
                    waiting_durs.append(sum(d for _, d in op.durations) / len(op.durations))
        features.append(float(np.mean(waiting_durs)) / 100.0 if waiting_durs else 0.0)
        features.append(len(waiting_durs) / 50.0)

        # Urgent jobs ratio (jobs with >50% ops still pending)
        urgent_jobs = 0
        for job in jobs:
            if job.is_finished():
                continue
            n_ops = job.get_operation_count()
            n_fin = sum(1 for i in range(n_ops)
                        if job.get_operation(i).get_status() == OperationStatus.FINISHED)
            if n_fin / max(n_ops, 1) < 0.5:
                urgent_jobs += 1
        features.append(urgent_jobs / total_jobs)

        # Longest remaining chain
        max_chain = 0
        for job in jobs:
            if job.is_finished():
                continue
            chain = 0
            for i in range(job.get_operation_count()):
                if job.get_operation(i).get_status() != OperationStatus.FINISHED:
                    chain += 1
            max_chain = max(max_chain, chain)
        features.append(max_chain / 20.0)

        # Shortest imminent (repeated for padding)
        features.append(min(imminent) / 100.0 if imminent else 0.0)
        features.append(float(np.std(imminent)) / 100.0 if len(imminent) > 1 else 0.0)

        # Pad/truncate to seq_feature_dim
        features = features[:self.seq_feature_dim]
        while len(features) < self.seq_feature_dim:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _precompute_context(self, machines: List[Machine], jobs: List[Job]):
        """Pre-compute shared data structures once per sample() call (avoids O(n^2)).

        Returns:
            op_to_job: {op_id: job}
            competing_ops: {machine_id: count of READY ops that can run on it}
            avg_queue: average machine input_queue length
            job_progress: {job_id: finished_ratio}
        """
        op_to_job = {}
        job_progress = {}
        for job in jobs:
            if job.is_finished():
                continue
            n_ops = job.get_operation_count()
            n_fin = 0
            for i in range(n_ops):
                op = job.get_operation(i)
                op_to_job[op.id] = job
                if op.get_status() == OperationStatus.FINISHED:
                    n_fin += 1
            job_progress[job.id] = n_fin / max(n_ops, 1)

        # Competing READY ops per machine (computed once, O(n*m))
        competing_ops = {}
        ready_ops_set = []
        for job in jobs:
            if job.is_finished():
                continue
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() == OperationStatus.READY:
                    ready_ops_set.append(op)

        for m in machines:
            cnt = 0
            for op in ready_ops_set:
                if op.is_machine_capable(m.id):
                    cnt += 1
            competing_ops[m.id] = cnt

        avg_queue = float(np.mean([len(m.input_queue) for m in machines])) if machines else 0.0

        return op_to_job, competing_ops, avg_queue, job_progress

    def _extract_route_features(self, op: Operation, machine: Machine,
                                machines: List[Machine], jobs: List[Job],
                                current_time: float, op_to_job, competing_ops,
                                avg_queue, job_progress) -> np.ndarray:
        """Extract routing features (16-dim) for a specific (op, machine) pair."""
        features = []

        # [0] Processing time on this machine / max_dur
        dur = op.get_duration(machine.id) if op.is_machine_capable(machine.id) else 0.0
        all_durs = [d for _, d in op.durations] if op.durations else [0.0]
        max_dur = max(all_durs) if all_durs else 1.0
        features.append(dur / max(max_dur, 1e-5))

        # [1] Machine timer
        timer = machine.timer / max(current_time + 1e-5, 1.0) if machine.status == MachineStatus.WORKING else 0.0
        features.append(timer)

        # [2] Queue length
        features.append(len(machine.input_queue) / 20.0)

        # [3] Machine availability
        features.append(1.0 if machine.is_available() else 0.0)

        # [4-5] Machine position
        features.append(machine.x / 100.0)
        features.append(machine.y / 100.0)

        # [6] Duration std across capable machines
        dur_std = float(np.std(all_durs)) if len(all_durs) > 1 else 0.0
        features.append(dur_std / 100.0)

        # [7] Remaining successor ops
        remaining = 0
        nxt = op.get_next_operation()
        while nxt is not None:
            remaining += 1
            nxt = nxt.get_next_operation()
        features.append(remaining / 10.0)

        # [8] Job completion progress (from precomputed map)
        features.append(job_progress.get(op_to_job[op.id].id, 0.0) if op.id in op_to_job else 0.0)

        # [9] Queue deviation from average
        features.append((len(machine.input_queue) - avg_queue) / 20.0)

        # [10] Efficiency ratio: this dur / min capable dur
        min_dur = min(all_durs) if all_durs else 1.0
        features.append(dur / max(min_dur, 1e-5))

        # [11] Estimated completion time
        est_completion = (machine.timer if machine.status == MachineStatus.WORKING else 0.0) + dur
        features.append(est_completion / 1000.0)

        # [12] Is first op in job
        is_first = 0.0
        job = op_to_job.get(op.id)
        if job:
            first_op = job.get_operation(0)
            if first_op and first_op.id == op.id:
                is_first = 1.0
        features.append(is_first)

        # [13] Is last op in job
        is_last = 0.0
        if job:
            last_op = job.get_operation(job.get_operation_count() - 1)
            if last_op and last_op.id == op.id:
                is_last = 1.0
        features.append(is_last)

        # [14] Machine working status
        features.append(1.0 if machine.status == MachineStatus.WORKING else 0.0)

        # [15] Competing READY ops count for this machine (from precomputed map)
        features.append(competing_ops.get(machine.id, 0) / 20.0)

        features = features[:self.route_feature_dim]
        while len(features) < self.route_feature_dim:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _extract_agv_features(self, agv: AGV, machine: Machine,
                              current_time: float, factory_graph,
                              n_available_agvs: int, n_total_agvs: int) -> np.ndarray:
        """Extract AGV-specific features (12-dim) for a candidate AGV."""
        features = []

        # [0] AGV timer
        features.append(agv.timer / 1000.0)

        # [1] Velocity
        features.append(agv.velocity / 10.0)

        # [2] Status (READY=1)
        features.append(1.0 if agv.get_status() == AGVStatus.READY else 0.0)

        # [3-4] Position
        features.append(agv.x / 100.0)
        features.append(agv.y / 100.0)

        # [5] Estimated travel distance via graph
        travel_dist = 0.0
        if factory_graph and hasattr(factory_graph, 'get_path'):
            path = factory_graph.get_path(agv.point_id, machine.point_id)
            if path and hasattr(factory_graph, 'get_path_weight'):
                travel_dist = factory_graph.get_path_weight(path)
        features.append(travel_dist / 100.0)

        # [6] Estimated travel time
        travel_time = travel_dist / max(agv.velocity, 0.1)
        features.append(travel_time / 100.0)

        # [7] Todo queue length
        features.append(len(agv.todo_queue) / 10.0)

        # [8] AGV already at machine location
        at_location = 1.0 if abs(agv.x - machine.x) < 1.0 and abs(agv.y - machine.y) < 1.0 else 0.0
        features.append(at_location)

        # [9] Idle time
        idle_time = max(0.0, current_time - agv.timer)
        features.append(idle_time / 1000.0)

        # [10] Euclidean distance
        eucl_dist = math.sqrt((agv.x - machine.x) ** 2 + (agv.y - machine.y) ** 2)
        features.append(eucl_dist / 100.0)

        # [11] Available AGV ratio
        features.append(n_available_agvs / max(n_total_agvs, 1))

        features = features[:self.agv_feature_dim]
        while len(features) < self.agv_feature_dim:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    # ----------------------------------------------------------
    # Core: Decision Flow (PPO policy sampling)
    # ----------------------------------------------------------

    def sample(self, agvs: List[AGV], machines: List[Machine],
               jobs: List[Job]) -> Tuple[List[Tuple[Operation, AGV, Machine]], float]:
        """Three-phase decision: Sequence -> Route -> AGV Select.

        In training mode, actions are sampled from the policy distribution (stochastic).
        In inference mode, actions are selected greedily (argmax).

        1. GNN encodes factory state
        2. SequencingActor ranks ready operations by priority
        3. For each op in priority order:
           a. RoutingActor selects machine
           b. AGVSelectionActor selects AGV
        """
        decisions = []

        # 1. Build and encode current state
        factory_graph = self._get_factory_graph(agvs)
        current_time = self._get_current_time()

        graph_result = self.graph_builder.build(
            agvs, machines, jobs, factory_graph, current_time,
        )

        self.gnn_encoder.eval()
        self.seq_actor.eval()
        self.route_actor.eval()
        self.agv_actor.eval()
        self.seq_critic.eval()
        self.route_critic.eval()
        self.agv_critic.eval()

        with torch.no_grad():
            global_state, node_embs = self.gnn_encoder(
                graph_result.data, graph_result.node_type_offsets,
            )

        global_state_np = global_state.cpu().numpy()

        # 2. Collect READY operations
        ready_ops = []
        for job in jobs:
            if job.is_finished():
                continue
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() == OperationStatus.READY:
                    ready_ops.append(op)

        if not ready_ops:
            # Check completion
            if all(j.is_finished() for j in jobs):
                self.alive = False
                return [], 0
            # Store transition with no decisions
            self._store_transition(global_state_np, None, None, None, None, None)
            return [], DEFAULT_STEP_TIME

        # 3. Sequencing: rank operations by policy
        seq_features = self._extract_seq_features(machines, jobs, current_time)

        with torch.no_grad():
            gs_t = torch.tensor(global_state_np, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            sf_t = torch.tensor(seq_features, dtype=torch.float32,
                                device=self.device).unsqueeze(0)

            # Get action logits and value
            seq_logits = self.seq_actor(gs_t, sf_t).squeeze(0)
            seq_value = self.seq_critic(gs_t, sf_t).item()

            # Mask invalid actions (only first len(ready_ops) are valid)
            valid_logits = seq_logits[:len(ready_ops)]

            if self.task_mode == TRAINING:
                # Sample from categorical distribution (stochastic policy)
                dist = Categorical(logits=valid_logits)
                seq_action = dist.sample()
                seq_log_prob = dist.log_prob(seq_action).item()
            else:
                # Greedy selection (argmax)
                seq_action = torch.argmax(valid_logits)
                seq_log_prob = 0.0

        # Build priority order from the sequencing action
        # For sequencing, we need to rank ALL operations, not just pick one
        # We use the full logits to determine priority order
        with torch.no_grad():
            all_logits = seq_logits[:len(ready_ops)]
            if self.task_mode == TRAINING:
                # Stochastic priority: sample permutation weighted by softmax probabilities
                probs = F.softmax(all_logits, dim=-1)
                # Sample without replacement using Gumbel-Top-k trick
                # Or simpler: rank by probability with some noise
                priority_order = torch.multinomial(probs, len(ready_ops), replacement=False).tolist()
            else:
                # Greedy: rank by logit value (descending)
                priority_order = torch.argsort(all_logits, descending=True).tolist()

        # 4. Pre-compute shared data structures once (avoids O(n^2))
        op_to_job, competing_ops, avg_queue, job_progress = self._precompute_context(machines, jobs)

        # 5. For each op in priority order: Route + AGV Select
        assigned_agvs = set()
        current_decisions_info = []
        route_action = -1
        route_log_prob = 0.0
        route_value = 0.0
        agv_action = -1
        agv_log_prob = 0.0
        agv_value = 0.0
        selected_route_feat = None
        selected_agv_feat = None

        for rank_idx in priority_order:
            op = ready_ops[rank_idx]

            # Find capable machines
            capable_machines = [m for m in machines
                                if op.is_machine_capable(m.id) and m.is_available()]
            if not capable_machines:
                continue

            # Find available AGVs
            available_agvs = [a for a in agvs
                              if a.get_status() == AGVStatus.READY
                              and (self.allow_agv_reassignment or a.id not in assigned_agvs)]
            if not available_agvs:
                continue

            # === ROUTING DECISION ===
            route_feats_all = []
            for m in capable_machines:
                rf = self._extract_route_features(
                    op, m, machines, jobs, current_time,
                    op_to_job, competing_ops, avg_queue, job_progress,
                )
                route_feats_all.append(rf)

            route_feats_np = np.array(route_feats_all, dtype=np.float32)
            with torch.no_grad():
                rf_t = torch.tensor(route_feats_np, dtype=torch.float32,
                                    device=self.device)
                gs_batch = gs_t.expand(len(capable_machines), -1)
                route_logits = self.route_actor(gs_batch, rf_t)

                # Get value from critic for the first candidate (shared state value)
                route_value = self.route_critic(gs_t, rf_t[0:1]).item()

                valid_route_logits = route_logits[:, 0]  # first logit per candidate

                if self.task_mode == TRAINING:
                    dist = Categorical(logits=valid_route_logits)
                    route_action = dist.sample()
                    route_log_prob = dist.log_prob(route_action).item()
                else:
                    route_action = torch.argmax(valid_route_logits)
                    route_log_prob = 0.0

            selected_machine = capable_machines[route_action.item()]

            # === AGV SELECTION DECISION ===
            n_avail = len(available_agvs)
            agv_feats_all = []
            for a in available_agvs:
                af = self._extract_agv_features(
                    a, selected_machine, current_time, factory_graph,
                    n_avail, len(agvs),
                )
                agv_feats_all.append(af)

            agv_feats_np = np.array(agv_feats_all, dtype=np.float32)
            with torch.no_grad():
                af_t = torch.tensor(agv_feats_np, dtype=torch.float32,
                                    device=self.device)
                gs_batch = gs_t.expand(len(available_agvs), -1)
                agv_logits = self.agv_actor(gs_batch, af_t)

                agv_value = self.agv_critic(gs_t, af_t[0:1]).item()

                valid_agv_logits = agv_logits[:, 0]  # first logit per candidate

                if self.task_mode == TRAINING:
                    dist = Categorical(logits=valid_agv_logits)
                    agv_action = dist.sample()
                    agv_log_prob = dist.log_prob(agv_action).item()
                else:
                    agv_action = torch.argmax(valid_agv_logits)
                    agv_log_prob = 0.0

            selected_agv = available_agvs[agv_action.item()]

            decisions.append((op, selected_agv, selected_machine))
            assigned_agvs.add(selected_agv.id)

            selected_route_feat = route_feats_np[route_action.item()]
            selected_agv_feat = agv_feats_np[agv_action.item()]

            current_decisions_info.append({
                'seq_action': rank_idx,
                'route_action': route_action.item(),
                'agv_action': agv_action.item(),
                'seq_features': seq_features,
                'route_features': selected_route_feat,
                'agv_features': selected_agv_feat,
                'seq_log_prob': seq_log_prob,
                'route_log_prob': route_log_prob,
                'agv_log_prob': agv_log_prob,
                'seq_value': seq_value,
                'route_value': route_value,
                'agv_value': agv_value,
            })

        # Store transition data for training
        self._store_transition(
            global_state_np, seq_features, current_decisions_info,
            None, None, None,
        )

        return decisions, DEFAULT_STEP_TIME

    # ----------------------------------------------------------
    # Reward
    # ----------------------------------------------------------

    def reward(self, *args, **kwargs) -> float:
        """Compute composite reward: makespan penalty + utilization + balance + waiting."""
        r = self._compute_reward()
        self._episode_reward += r
        return r

    def _compute_reward(self) -> float:
        """Pure reward computation without side effects."""
        reward = 0.0

        if self.context and hasattr(self.context, 'env_timeline'):
            reward -= 0.005 * self.context.env_timeline

        if self.context and hasattr(self.context, 'jobs'):
            jobs = self.context.jobs
            total = len(jobs) if jobs else 1
            completed = sum(1 for j in jobs if j.is_finished())
            reward += 50.0 * (completed / total)

            waiting = sum(
                1 for j in jobs
                for i in range(j.get_operation_count())
                if j.get_operation(i).get_status() == OperationStatus.WAITING
            )
            reward -= 0.5 * waiting

        if self.context and hasattr(self.context, 'machines'):
            machines = self.context.machines
            total_m = len(machines) if machines else 1
            working = sum(1 for m in machines if m.status == MachineStatus.WORKING)
            reward += 2.0 * (working / total_m)

            loads = [len(m.input_queue) for m in machines]
            if len(loads) > 1 and np.mean(loads) > 0:
                cv = float(np.std(loads)) / (float(np.mean(loads)) + 1e-5)
                reward -= 3.0 * cv

        if self.context and hasattr(self.context, 'agvs'):
            agvs = self.context.agvs
            total_a = len(agvs) if agvs else 1
            active = sum(1 for a in agvs if a.get_status() != AGVStatus.READY)
            reward += 1.5 * (active / total_a)

        return reward

    # ----------------------------------------------------------
    # Transition Collection
    # ----------------------------------------------------------

    def _store_transition(self, global_state_np, seq_features, decisions_info,
                          next_global_state_np, next_seq_features, reward):
        """Store current step data and collect transition from previous step into rollout buffer."""
        if self.task_mode != TRAINING:
            self._prev_global_state = global_state_np
            self._prev_seq_features = seq_features
            self._prev_decisions_info = []
            return

        # Collect transitions from previous step into rollout buffer
        if self._prev_global_state is not None and self._prev_decisions_info:
            step_reward = self._compute_reward()

            # Normalize reward
            self.reward_normalizer.update(np.array([step_reward]))
            norm_reward = self.reward_normalizer.normalize(np.array([step_reward]))[0]

            done = False
            if self.context and hasattr(self.context, 'jobs'):
                done = all(j.is_finished() for j in self.context.jobs)

            next_gs = global_state_np if global_state_np is not None else self._prev_global_state
            next_sf = seq_features if seq_features is not None else self._prev_seq_features

            n_decisions = len(self._prev_decisions_info)
            shared_reward = norm_reward / max(n_decisions, 1)

            # Use the first decision's log_probs and values for this step
            # (they share the same global state and reward)
            first_info = self._prev_decisions_info[0]

            self.rollout_buffer.push(
                state=self._prev_global_state,
                seq_features=self._prev_seq_features,
                reward=shared_reward,
                done=float(done),
                # Sequencing
                seq_action=first_info.get('seq_action', -1),
                seq_log_prob=first_info.get('seq_log_prob', 0.0),
                seq_value=first_info.get('seq_value', 0.0),
                # Routing
                route_action=first_info.get('route_action', -1),
                route_log_prob=first_info.get('route_log_prob', 0.0),
                route_value=first_info.get('route_value', 0.0),
                route_feat=first_info.get('route_features'),
                # AGV
                agv_action=first_info.get('agv_action', -1),
                agv_log_prob=first_info.get('agv_log_prob', 0.0),
                agv_value=first_info.get('agv_value', 0.0),
                agv_feat=first_info.get('agv_features'),
            )

            self._rollout_step_count += 1

        # Update previous state
        self._prev_global_state = global_state_np
        self._prev_seq_features = seq_features
        self._prev_decisions_info = decisions_info if decisions_info else []

    def after_sample(self, *args, **kwargs):
        """Post-sample hook: no-op (transition collection is done inside _store_transition)."""
        pass

    def before_sample(self, *args, **kwargs):
        """Pre-sample hook: no-op."""
        pass

    # ----------------------------------------------------------
    # PPO Training
    # ----------------------------------------------------------

    def train(self, *args, **kwargs):
        """Train all three Actor-Critic pairs using PPO.

        Triggered every rollout_steps environment steps.
        Performs multiple epochs of PPO updates over the collected rollout buffer.
        """
        if self.task_mode != TRAINING:
            return

        # Only update when rollout buffer is full enough
        if len(self.rollout_buffer) < self.rollout_steps:
            return

        # Get bootstrap value for the last state
        if self._prev_global_state is not None:
            with torch.no_grad():
                gs_t = torch.tensor(self._prev_global_state, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
                sf_t = torch.tensor(self._prev_seq_features, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
                # Use average of all three critic values as bootstrap
                v_seq = self.seq_critic(gs_t, sf_t).item()
                # For route and agv, use the first stored features
                if self.rollout_buffer.route_features:
                    rf_last = np.array(self.rollout_buffer.route_features[-1], dtype=np.float32)
                    rf_t = torch.tensor(rf_last, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
                    v_route = self.route_critic(gs_t, rf_t).item()
                else:
                    v_route = 0.0
                if self.rollout_buffer.agv_features:
                    af_last = np.array(self.rollout_buffer.agv_features[-1], dtype=np.float32)
                    af_t = torch.tensor(af_last, dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
                    v_agv = self.agv_critic(gs_t, af_t).item()
                else:
                    v_agv = 0.0
                last_value = (v_seq + v_route + v_agv) / 3.0
        else:
            last_value = 0.0

        # Compute GAE-based returns and advantages
        self.rollout_buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda,
        )

        # PPO update for all three sub-agents
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.ppo_epochs):
            seq_loss_dict = self._ppo_update_sub_agent('seq')
            route_loss_dict = self._ppo_update_sub_agent('route')
            agv_loss_dict = self._ppo_update_sub_agent('agv')

            for loss_dict in [seq_loss_dict, route_loss_dict, agv_loss_dict]:
                if loss_dict is not None:
                    total_actor_loss += loss_dict['actor_loss']
                    total_critic_loss += loss_dict['critic_loss']
                    total_entropy += loss_dict['entropy']
                    n_updates += 1

            # 主动释放 GIL，防止 CPU 密集的 PPO 训练阻塞 asyncio 事件循环
            time.sleep(0)

        self._train_step += 1

        # Record training stats
        if n_updates > 0:
            avg_actor = total_actor_loss / n_updates
            avg_critic = total_critic_loss / n_updates
            avg_entropy = total_entropy / n_updates
            avg_total = avg_actor + self.value_loss_coeff * avg_critic - self.entropy_coeff * avg_entropy

            self.training_history['actor_loss'].append(avg_actor)
            self.training_history['critic_loss'].append(avg_critic)
            self.training_history['entropy'].append(avg_entropy)
            self.training_history['total_loss'].append(avg_total)

        # Clear rollout buffer after update
        self.rollout_buffer.reset()
        self._rollout_step_count = 0

    def _ppo_update_sub_agent(self, sub_agent: str) -> Optional[Dict[str, float]]:
        """Perform one PPO update epoch for a sub-agent.

        Args:
            sub_agent: 'seq', 'route', or 'agv'

        Returns:
            Dictionary with actor_loss, critic_loss, entropy, or None if insufficient data
        """
        data = self.rollout_buffer.get_tensors(sub_agent)
        n_samples = data['states'].shape[0]

        if n_samples < self.ppo_batch_size:
            # Use all data if not enough for a mini-batch
            mini_batches = [None]
        else:
            n_batches = max(1, n_samples // self.ppo_batch_size)
            indices = np.random.permutation(n_samples)
            mini_batches = [indices[i * self.ppo_batch_size:(i + 1) * self.ppo_batch_size]
                           for i in range(n_batches)]

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_batches_actual = 0

        # Select the correct networks and optimizer
        if sub_agent == 'seq':
            actor = self.seq_actor
            critic = self.seq_critic
            optimizer = self.optimizer_seq
            feat_key = 'seq_features'
            max_output_dim = self.max_seq_output_dim
        elif sub_agent == 'route':
            actor = self.route_actor
            critic = self.route_critic
            optimizer = self.optimizer_route
            feat_key = 'route_features'
            max_output_dim = self.max_route_output_dim
        elif sub_agent == 'agv':
            actor = self.agv_actor
            critic = self.agv_critic
            optimizer = self.optimizer_agv
            feat_key = 'agv_features'
            max_output_dim = self.max_agv_output_dim

        # Set networks to training mode
        self.gnn_encoder.train()
        actor.train()
        critic.train()

        for batch_indices in mini_batches:
            if batch_indices is None:
                # Use all data
                states = data['states']
                features = data[feat_key]
                actions = data['actions']
                old_log_probs = data['old_log_probs']
                advantages = data['advantages']
                returns = data['returns']
            else:
                states = data['states'][batch_indices]
                features = data[feat_key][batch_indices]
                actions = data['actions'][batch_indices]
                old_log_probs = data['old_log_probs'][batch_indices]
                advantages = data['advantages'][batch_indices]
                returns = data['returns'][batch_indices]

            # Clamp actions to valid range
            actions = actions.clamp(0, max_output_dim - 1)

            # Re-encode states through GNN for gradient flow
            # Note: We store raw global_state in the buffer; to get GNN gradients,
            # we recompute from the stored states (treated as fixed features here)
            # This matches the DQN approach where states are treated as fixed inputs

            # Evaluate current policy
            logits = actor(states, features)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            values = critic(states, features)

            # PPO clipped surrogate loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            critic_loss = F.mse_loss(values, returns)

            # Total loss
            loss = actor_loss + self.value_loss_coeff * critic_loss - self.entropy_coeff * entropy

            # Optimizer step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.gnn_encoder.parameters()) + list(actor.parameters()) + list(critic.parameters()),
                self.max_grad_norm,
            )
            optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            n_batches_actual += 1

        if n_batches_actual == 0:
            return None

        return {
            'actor_loss': total_actor_loss / n_batches_actual,
            'critic_loss': total_critic_loss / n_batches_actual,
            'entropy': total_entropy / n_batches_actual,
        }

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _get_factory_graph(self, agvs):
        """Get the factory floor graph from AGVs."""
        if agvs and hasattr(agvs[0], 'graph'):
            return agvs[0].graph
        if self.context and hasattr(self.context, 'graph'):
            return self.context.graph
        return None

    def _get_current_time(self) -> float:
        """Get current simulation time."""
        if self.context and hasattr(self.context, 'env_timeline'):
            return self.context.env_timeline
        return 0.0

    def _remaining_processing_time(self, op: Operation) -> float:
        """Compute total remaining processing time for an operation's job."""
        remaining = 0.0
        nxt = op
        while nxt is not None:
            if nxt.durations:
                avg_dur = sum(d for _, d in nxt.durations) / len(nxt.durations)
                remaining += avg_dur
            nxt = nxt.get_next_operation()
        return remaining

    # ----------------------------------------------------------
    # Model save / load
    # ----------------------------------------------------------

    def save_model(self, path: Optional[str] = None) -> bool:
        """Save all network state_dicts and hyperparameters to a .pt file."""
        try:
            if path is None:
                agent_name = self.name or "GraphPPOAgent"
                agent_dir = f"training_logs/models/{agent_name}"
                os.makedirs(agent_dir, exist_ok=True)
                path = f"{agent_dir}/agent_model.pt"

            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            checkpoint = {
                'gnn_encoder': self.gnn_encoder.state_dict(),
                'seq_actor': self.seq_actor.state_dict(),
                'seq_critic': self.seq_critic.state_dict(),
                'route_actor': self.route_actor.state_dict(),
                'route_critic': self.route_critic.state_dict(),
                'agv_actor': self.agv_actor.state_dict(),
                'agv_critic': self.agv_critic.state_dict(),
                'optimizer_gnn': self.optimizer_gnn.state_dict(),
                'optimizer_seq': self.optimizer_seq.state_dict(),
                'optimizer_route': self.optimizer_route.state_dict(),
                'optimizer_agv': self.optimizer_agv.state_dict(),
                'hyperparams': {
                    'hidden_dim': self.hidden_dim,
                    'gamma': self.gamma,
                    'clip_epsilon': self.clip_epsilon,
                    'gae_lambda': self.gae_lambda,
                    'ppo_epochs': self.ppo_epochs,
                    'ppo_batch_size': self.ppo_batch_size,
                    'entropy_coeff': self.entropy_coeff,
                    'value_loss_coeff': self.value_loss_coeff,
                    'max_grad_norm': self.max_grad_norm,
                    'rollout_steps': self.rollout_steps,
                    'max_seq_output_dim': self.max_seq_output_dim,
                    'max_route_output_dim': self.max_route_output_dim,
                    'max_agv_output_dim': self.max_agv_output_dim,
                    'seq_feature_dim': self.seq_feature_dim,
                    'route_feature_dim': self.route_feature_dim,
                    'agv_feature_dim': self.agv_feature_dim,
                    'allow_agv_reassignment': self.allow_agv_reassignment,
                },
                'reward_normalizer': {
                    'mean': self.reward_normalizer.mean,
                    'var': self.reward_normalizer.var,
                    'count': self.reward_normalizer.count,
                },
                'training_history': self.training_history,
                'train_step': self._train_step,
                'mode': self.mode,
            }
            torch.save(checkpoint, path)
            LOGGER.info(f"[GraphPPOAgent] Model saved to {path}")
            return True
        except Exception as e:
            LOGGER.error(f"[GraphPPOAgent] Save failed: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load all network state_dicts and hyperparameters from a .pt file."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])
            self.seq_actor.load_state_dict(checkpoint['seq_actor'])
            self.seq_critic.load_state_dict(checkpoint['seq_critic'])
            self.route_actor.load_state_dict(checkpoint['route_actor'])
            self.route_critic.load_state_dict(checkpoint['route_critic'])
            self.agv_actor.load_state_dict(checkpoint['agv_actor'])
            self.agv_critic.load_state_dict(checkpoint['agv_critic'])

            for opt_key, optimizer in [
                ('optimizer_gnn', self.optimizer_gnn),
                ('optimizer_seq', self.optimizer_seq),
                ('optimizer_route', self.optimizer_route),
                ('optimizer_agv', self.optimizer_agv),
            ]:
                if opt_key in checkpoint:
                    optimizer.load_state_dict(checkpoint[opt_key])

            hp = checkpoint.get('hyperparams', {})
            self.gamma = hp.get('gamma', self.gamma)
            self.clip_epsilon = hp.get('clip_epsilon', self.clip_epsilon)
            self.gae_lambda = hp.get('gae_lambda', self.gae_lambda)
            self.ppo_epochs = hp.get('ppo_epochs', self.ppo_epochs)
            self.ppo_batch_size = hp.get('ppo_batch_size', self.ppo_batch_size)
            self.entropy_coeff = hp.get('entropy_coeff', self.entropy_coeff)
            self.value_loss_coeff = hp.get('value_loss_coeff', self.value_loss_coeff)

            if 'reward_normalizer' in checkpoint:
                rn = checkpoint['reward_normalizer']
                self.reward_normalizer.mean = rn['mean']
                self.reward_normalizer.var = rn['var']
                self.reward_normalizer.count = rn['count']

            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']

            self._train_step = checkpoint.get('train_step', 0)

            LOGGER.info(f"[GraphPPOAgent] Model loaded from {path}")
            return True
        except Exception as e:
            LOGGER.error(f"[GraphPPOAgent] Load failed: {e}")
            return False

    # ----------------------------------------------------------
    # Training Metrics
    # ----------------------------------------------------------

    def get_training_metrics(self) -> Dict[str, Any]:
        """Return training metrics for convergence detection."""
        metrics = {
            'episode_reward': self._episode_reward,
            'training_history': self.training_history,
        }
        if self.training_history.get('actor_loss'):
            metrics['actor_loss'] = self.training_history['actor_loss'][-1]
        if self.training_history.get('critic_loss'):
            metrics['critic_loss'] = self.training_history['critic_loss'][-1]
        if self.training_history.get('entropy'):
            metrics['entropy'] = self.training_history['entropy'][-1]
        if self.training_history.get('total_loss'):
            metrics['total_loss'] = self.training_history['total_loss'][-1]
        return metrics

    def new_episode(self):
        """Reset episode-level state. Call at the start of each episode."""
        self._episode_reward = 0.0
        self._prev_global_state = None
        self._prev_seq_features = None
        self._prev_decisions_info = []

        # Record episode stats
        if self.training_history['episodes']:
            last_ep = self.training_history['episodes'][-1] + 1
        else:
            last_ep = 1
        self.training_history['episodes'].append(last_ep)

        makespan = 0.0
        if self.context and hasattr(self.context, 'env_timeline'):
            makespan = self.context.env_timeline
        self.training_history['makespans'].append(makespan)

    def __repr__(self):
        return (f"<GraphPPOAgent id={self.agent_id} name={self.name} "
                f"mode={self.mode} device={self.device}>")
