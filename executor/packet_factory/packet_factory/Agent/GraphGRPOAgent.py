"""
GraphGRPOAgent - GNN-enhanced GRPO Agent for FJSP

Combines GNN-based state encoding (from GraphPPOAgent) with GRPO
(Group Relative Policy Optimization) for critic-free training:

Architecture:
  FactoryGraphBuilder -> GNNStateEncoder -> global_state [256] + node_embs [N, 64]
                                                |
                     +--------------------------+--------------------------+
                     |                          |                          |
              SequencingActor           RoutingActor           AGVSelectionActor
           (operation prioritization)  (machine assignment)       (AGV pick)

Key differences from GraphPPOAgent (PPO):
- No critic networks: GRPO uses group-relative advantage normalization instead of GAE
- GRPORolloutBuffer with group_id tracking and group-relative advantage computation
- Clipped surrogate + optional KL penalty, no value loss
- Store full candidate feature sets for route/AGV decisions to ensure
  sampling distribution matches training distribution
- Reward design references DualDRLAgent (load balance, makespan, utilization,
  queue reduction, bottleneck relief) plus graph-agent dense shaping
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
# Factory Graph Builder (reused from GraphDualAgent / GraphPPOAgent)
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
# GNN State Encoder (reused from GraphDualAgent / GraphPPOAgent)
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
# GRPO Rollout Buffer (group-relative advantage, no critic/GAE)
# ============================================================

class GRPORolloutBuffer:
    """On-policy rollout buffer for GRPO with group-relative advantage normalization.

    Unlike PPO's RolloutBuffer:
    - No value/GAE tracking (GRPO is critic-free)
    - Stores group_id per transition for group-relative normalization
    - Stores full candidate feature arrays for route/AGV decisions
      so training can reconstruct the same categorical distribution over candidates
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
        self.rewards = []
        self.group_ids = []
        self.trajectory_ids = []
        # Graph snapshots are kept on CPU and re-encoded during policy updates.
        # This preserves a gradient path to the shared GNN encoder without
        # retaining the original rollout computation graph.
        self.graph_data = []
        self.graph_node_type_offsets = []

        # Sequencing
        self.seq_actions = []
        self.seq_log_probs = []
        self.seq_candidate_counts = []

        # Routing — store ALL candidate features per step
        self.route_actions = []
        self.route_log_probs = []
        self.route_all_candidate_features = []  # list of np.ndarray [n_candidates, feat_dim]

        # AGV — store ALL candidate features per step
        self.agv_actions = []
        self.agv_log_probs = []
        self.agv_all_candidate_features = []  # list of np.ndarray [n_candidates, feat_dim]

        # Computed after rollout
        self.advantages = None
        self.returns = None
        self.advantage_diagnostics = {}

    def __len__(self):
        return len(self.rewards)

    def push(self, state: np.ndarray, seq_features: np.ndarray,
             reward: float, done: bool, group_id: int,
             trajectory_id: Optional[int] = None,
             graph_data: Optional[Data] = None,
             graph_node_type_offsets: Optional[Dict[str, Tuple[int, int]]] = None,
             # Sequencing
             seq_action: int = -1, seq_log_prob: float = 0.0,
             seq_candidate_count: int = 0,
             # Routing
             route_action: int = -1, route_log_prob: float = 0.0,
             route_all_candidate_feats: Optional[np.ndarray] = None,
             # AGV
             agv_action: int = -1, agv_log_prob: float = 0.0,
             agv_all_candidate_feats: Optional[np.ndarray] = None):
        """Store one transition step."""
        self.states.append(state)
        self.seq_features.append(seq_features)
        self.dones.append(done)
        self.rewards.append(reward)
        self.group_ids.append(group_id)
        self.trajectory_ids.append(trajectory_id)
        self.graph_data.append(graph_data)
        self.graph_node_type_offsets.append(graph_node_type_offsets)

        self.seq_actions.append(seq_action)
        self.seq_log_probs.append(seq_log_prob)
        self.seq_candidate_counts.append(seq_candidate_count)

        self.route_actions.append(route_action)
        self.route_log_probs.append(route_log_prob)
        self.route_all_candidate_features.append(
            route_all_candidate_feats if route_all_candidate_feats is not None
            else np.zeros((1, 1), dtype=np.float32)
        )

        self.agv_actions.append(agv_action)
        self.agv_log_probs.append(agv_log_prob)
        self.agv_all_candidate_features.append(
            agv_all_candidate_feats if agv_all_candidate_feats is not None
            else np.zeros((1, 1), dtype=np.float32)
        )

    def compute_group_relative_advantages(self, gamma: float, min_group_size: int = 2):
        """Compute discounted returns and group-relative advantages (no critic).

        Steps:
        1. Compute discounted returns backwards within episode boundaries.
        2. Partition samples by group_id.
        3. For each group, normalize: advantage = (return - group_mean) / (group_std + eps).
        4. If a group has fewer than min_group_size, fall back to global normalization.

        Args:
            gamma: discount factor
            min_group_size: minimum samples per group for group normalization
        """
        n = len(self.rewards)
        if n == 0:
            return

        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # Compute discounted returns at the environment-decision level.
        # Several transitions can be emitted by one environment state (the
        # sequence, routing and AGV decisions). They share one reward and do
        # not represent consecutive simulator time steps. Aggregating by
        # group before discounting prevents those transitions from receiving
        # different artificial discount factors.
        returns = np.zeros(n, dtype=np.float32)
        group_to_indices = {}
        ordered_groups = []
        for idx, gid in enumerate(self.group_ids):
            if gid not in group_to_indices:
                group_to_indices[gid] = []
                ordered_groups.append(gid)
            group_to_indices[gid].append(idx)

        group_rewards = []
        group_dones = []
        for gid in ordered_groups:
            idx = group_to_indices[gid]
            group_rewards.append(float(rewards[idx].sum()))
            group_dones.append(bool(np.any(dones[idx])))

        group_returns = np.zeros(len(ordered_groups), dtype=np.float32)
        running = 0.0
        for group_pos in reversed(range(len(ordered_groups))):
            if group_dones[group_pos]:
                running = 0.0
            running = group_rewards[group_pos] + gamma * running
            group_returns[group_pos] = running

        for group_pos, gid in enumerate(ordered_groups):
            returns[group_to_indices[gid]] = group_returns[group_pos]

        # Compute group-relative advantages
        advantages = np.zeros(n, dtype=np.float32)
        global_mean = returns.mean()
        global_std_raw = returns.std()
        global_std = global_std_raw + 1e-8

        # A decision batch may contain several three-stage actions that all
        # receive the same shared reward. In that case the within-group
        # return standard deviation is exactly zero, and normalizing by the
        # group would silently turn every advantage into zero. Fall back to
        # buffer-wide normalization for such degenerate groups.
        normalized_group_count = 0
        fallback_group_count = 0
        degenerate_group_count = 0

        group_ids_arr = np.array(self.group_ids)
        unique_groups = np.unique(group_ids_arr)

        for gid in unique_groups:
            idx = np.where(group_ids_arr == gid)[0]
            group_returns = returns[idx]

            group_std_raw = group_returns.std()
            if len(idx) >= min_group_size and group_std_raw > 1e-6:
                mean = group_returns.mean()
                std = group_std_raw + 1e-8
                normalized_group_count += 1
            else:
                # This covers singleton groups and shared-reward groups with
                # no within-group variation.
                mean = global_mean
                std = global_std
                fallback_group_count += 1
                if len(idx) >= min_group_size and group_std_raw <= 1e-6:
                    degenerate_group_count += 1

            advantages[idx] = (group_returns - mean) / std

        self.returns = returns
        self.advantages = advantages
        self.advantage_diagnostics = {
            'group_count': int(len(unique_groups)),
            'normalized_group_count': int(normalized_group_count),
            'fallback_group_count': int(fallback_group_count),
            'degenerate_group_count': int(degenerate_group_count),
            'global_return_std': float(global_std_raw),
            'nonzero_advantage_count': int(np.count_nonzero(np.abs(advantages) > 1e-8)),
        }

    def compute_strict_group_relative_advantages(self, gamma: float):
        """Normalize complete trajectory returns within each rollout group.

        Strict groups are formed by collecting multiple independent episodes
        from the same initial state and event condition. ``group_id`` is
        shared by those episodes while ``trajectory_id`` identifies one
        complete rollout. The normalized trajectory return is copied to all
        transitions in that trajectory, which matches a terminal-return
        GRPO/REINFORCE-style update and keeps the semantics separate from the
        decision-batch engineering variant.
        """
        n = len(self.rewards)
        if n == 0:
            return
        if any(tid is None for tid in self.trajectory_ids):
            raise ValueError("strict group advantages require trajectory_id for every transition")

        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        trajectory_ids = np.asarray(self.trajectory_ids)
        group_ids = np.asarray(self.group_ids)
        unique_trajectories = list(dict.fromkeys(self.trajectory_ids))
        trajectory_returns = {}

        # Compute each trajectory's discounted return-to-go and use the first
        # state return as its complete rollout score.
        for tid in unique_trajectories:
            idx = np.where(trajectory_ids == tid)[0]
            running = 0.0
            first_return = 0.0
            for pos in reversed(idx.tolist()):
                if dones[pos]:
                    running = 0.0
                running = float(rewards[pos]) + gamma * running
                first_return = running
            trajectory_returns[tid] = first_return

        all_scores = np.asarray(list(trajectory_returns.values()), dtype=np.float32)
        global_mean = float(all_scores.mean())
        global_std_raw = float(all_scores.std())
        global_std = global_std_raw + 1e-8
        advantages = np.zeros(n, dtype=np.float32)
        normalized_group_count = 0
        fallback_group_count = 0
        degenerate_group_count = 0

        for gid in np.unique(group_ids):
            group_tids = list(dict.fromkeys(
                tid for tid, sample_gid in zip(self.trajectory_ids, self.group_ids)
                if sample_gid == gid
            ))
            scores = np.asarray([trajectory_returns[tid] for tid in group_tids], dtype=np.float32)
            score_std = float(scores.std())
            if len(scores) >= 2 and score_std > 1e-6:
                mean, std = float(scores.mean()), score_std + 1e-8
                normalized_group_count += 1
            else:
                mean, std = global_mean, global_std
                fallback_group_count += 1
                if len(scores) >= 2 and score_std <= 1e-6:
                    degenerate_group_count += 1
            for tid in group_tids:
                advantages[trajectory_ids == tid] = (trajectory_returns[tid] - mean) / std

        self.returns = np.asarray(
            [trajectory_returns[tid] for tid in self.trajectory_ids], dtype=np.float32
        )
        self.advantages = advantages
        self.advantage_diagnostics = {
            'strict': True,
            'group_count': int(len(np.unique(group_ids))),
            'trajectory_count': int(len(unique_trajectories)),
            'normalized_group_count': int(normalized_group_count),
            'fallback_group_count': int(fallback_group_count),
            'degenerate_group_count': int(degenerate_group_count),
            'global_return_std': global_std_raw,
            'nonzero_advantage_count': int(np.count_nonzero(np.abs(advantages) > 1e-8)),
        }

    def get_tensors(self, sub_agent: str) -> Dict[str, Any]:
        """Convert stored data to tensors for GRPO update.

        Args:
            sub_agent: 'seq', 'route', or 'agv'

        Returns:
            Dictionary of torch tensors and candidate feature lists
        """
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        seq_feats = torch.tensor(np.array(self.seq_features), dtype=torch.float32, device=self.device)
        advantages = torch.tensor(self.advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(self.returns, dtype=torch.float32, device=self.device)

        if sub_agent == 'seq':
            actions = torch.tensor(np.array(self.seq_actions), dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor(np.array(self.seq_log_probs), dtype=torch.float32, device=self.device)
            candidate_counts = torch.tensor(
                np.array(self.seq_candidate_counts), dtype=torch.long, device=self.device
            )
            return {
                'states': states, 'seq_features': seq_feats,
                'actions': actions, 'old_log_probs': old_log_probs,
                'candidate_counts': candidate_counts,
                'advantages': advantages, 'returns': returns,
                'graph_data': self.graph_data,
                'graph_node_type_offsets': self.graph_node_type_offsets,
            }
        elif sub_agent == 'route':
            actions = torch.tensor(np.array(self.route_actions), dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor(np.array(self.route_log_probs), dtype=torch.float32, device=self.device)
            # Pass raw candidate feature list for per-sample reconstruction
            return {
                'states': states,
                'actions': actions, 'old_log_probs': old_log_probs,
                'advantages': advantages, 'returns': returns,
                'route_all_candidate_features': self.route_all_candidate_features,
                'graph_data': self.graph_data,
                'graph_node_type_offsets': self.graph_node_type_offsets,
            }
        elif sub_agent == 'agv':
            actions = torch.tensor(np.array(self.agv_actions), dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor(np.array(self.agv_log_probs), dtype=torch.float32, device=self.device)
            return {
                'states': states,
                'actions': actions, 'old_log_probs': old_log_probs,
                'advantages': advantages, 'returns': returns,
                'agv_all_candidate_features': self.agv_all_candidate_features,
                'graph_data': self.graph_data,
                'graph_node_type_offsets': self.graph_node_type_offsets,
            }
        else:
            raise ValueError(f"Unknown sub_agent: {sub_agent}")


# ============================================================
# Actor Networks (Policy Networks — same as GraphPPOAgent)
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
# GraphGRPOAgent
# ============================================================

@register_component("packet_factory.GraphGRPOAgent")
class GraphGRPOAgent(BaseAgent):
    """GNN-enhanced GRPO Agent for FJSP.

    Combines GNN state encoding with GRPO (Group Relative Policy Optimization):
    1. GNN encodes factory graph -> global_state [256] + node_embs
    2. SequencingActor ranks READY operations by priority
    3. For each operation in priority order:
       a. RoutingActor selects the best capable machine
       b. AGVSelectionActor selects the best available AGV
    4. Training: critic-free GRPO with group-relative advantage normalization,
       clipped surrogate, optional KL penalty, entropy bonus

    Reward design references DualDRLAgent (load balance, makespan, utilization,
    queue reduction, bottleneck relief) combined with graph-agent dense shaping.
    """

    def __init__(self, name=None, agent_id=None, context=None,
                 ui_mode: str = BACKEND, task_mode: str = TRAINING,
                 model_path: Optional[str] = None,
                 # Graph encoder
                 hidden_dim: int = 64,
                 # GRPO common
                 gamma: float = 0.99,
                 # Learning rates
                 lr_gnn: float = 1e-4,
                 lr_actor: float = 3e-4,
                 # GRPO hyperparameters
                 clip_epsilon: float = 0.2,
                 grpo_epochs: int = 4,
                 grpo_batch_size: int = 64,
                 grpo_group_size: int = 8,
                 min_group_size: int = 2,
                 entropy_coeff: float = 0.01,
                 kl_coeff: float = 0.0,
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
                 train_gnn: bool = True,
                 strict_group_sampling: bool = False,
                 state_encoder_mode: str = 'graph',
                 reward_mode: str = 'composite',
                 agv_selection_mode: str = 'policy',
                 device: Optional[str] = None,
                 **kwargs):
        super().__init__(name, agent_id, context, ui_mode, task_mode, model_path)

        self.hidden_dim = hidden_dim
        self.state_dim = hidden_dim * 4
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.grpo_epochs = grpo_epochs
        self.grpo_batch_size = grpo_batch_size
        self.grpo_group_size = grpo_group_size
        self.min_group_size = min_group_size
        self.entropy_coeff = entropy_coeff
        self.kl_coeff = kl_coeff
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.max_seq_output_dim = max_seq_output_dim
        self.max_route_output_dim = max_route_output_dim
        self.max_agv_output_dim = max_agv_output_dim
        self.seq_feature_dim = seq_feature_dim
        self.route_feature_dim = route_feature_dim
        self.agv_feature_dim = agv_feature_dim
        self.allow_agv_reassignment = allow_agv_reassignment
        self.train_gnn = train_gnn
        self.strict_group_sampling = strict_group_sampling
        if state_encoder_mode not in {'graph', 'flat'}:
            raise ValueError("state_encoder_mode must be 'graph' or 'flat'")
        if reward_mode not in {'composite', 'time_only'}:
            raise ValueError("reward_mode must be 'composite' or 'time_only'")
        if agv_selection_mode not in {'policy', 'first_available'}:
            raise ValueError("agv_selection_mode must be 'policy' or 'first_available'")
        self.state_encoder_mode = state_encoder_mode
        self.reward_mode = reward_mode
        self.agv_selection_mode = agv_selection_mode
        self._strict_group_id: Optional[int] = None
        self._strict_trajectory_id: Optional[int] = None

        # Device
        self.device = resolve_device(device, tag="GraphGRPOAgent")
        log_device(self.device, tag="GraphGRPOAgent")

        # Graph builder
        self.graph_builder = FactoryGraphBuilder(device=str(self.device))

        # GNN encoder (shared across all Actors)
        self.gnn_encoder = GNNStateEncoder(
            hidden_dim=hidden_dim, num_layers=2,
        ).to(self.device)

        # Initialize Actors only (no critics — GRPO is critic-free)
        self._initialize_networks(lr_gnn, lr_actor)

        # GRPO rollout buffer
        self.rollout_buffer = GRPORolloutBuffer(self.device)

        # Reward normalization
        self.reward_normalizer = RunningMeanStd()

        # Rollout step counter
        self._rollout_step_count = 0
        # Strict group collection can defer policy updates until all
        # independent rollouts in a group have finished.
        self.defer_updates = False

        # Group tracking.  A group now corresponds to one environment
        # decision batch (all three-stage actions emitted from one state),
        # rather than an arbitrary slice of consecutive transitions.
        self._grpo_group_id = 0

        # State tracking for transition collection
        self._prev_global_state: Optional[np.ndarray] = None
        self._prev_seq_features: Optional[np.ndarray] = None
        self._prev_graph_data: Optional[Data] = None
        self._prev_graph_node_type_offsets: Optional[Dict[str, Tuple[int, int]]] = None
        self._prev_decisions_info: List[Dict] = []

        # DualDRLAgent-style queue tracking for reward
        self._prev_queue_len = 0

        # Training statistics
        self.training_history: Dict[str, list] = {
            'episodes': [], 'policy_loss': [],
            'entropy': [], 'total_loss': [], 'makespans': [],
            'kl_approx': [],
            'advantage_diagnostics': [],
        }
        self._episode_reward = 0.0
        self._train_step = 0

        # Load model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            LOGGER.info(f"[GraphGRPOAgent] Loaded model from {model_path}")
        else:
            LOGGER.info(f"[GraphGRPOAgent] Initialized on {self.device}, "
                        f"hidden_dim={hidden_dim}, state_dim={self.state_dim}, "
                        f"clip_epsilon={clip_epsilon}, grpo_group_size={grpo_group_size}")

    def _initialize_networks(self, lr_gnn, lr_actor):
        """Initialize all Actors and optimizers (no critics)."""
        # Sequencing Actor
        self.seq_actor = SequencingActor(
            self.state_dim, self.seq_feature_dim, self.max_seq_output_dim,
        ).to(self.device)

        # Routing Actor
        self.route_actor = RoutingActor(
            self.state_dim, self.route_feature_dim, self.max_route_output_dim,
        ).to(self.device)

        # AGV Selection Actor
        self.agv_actor = AGVSelectionActor(
            self.state_dim, self.agv_feature_dim, self.max_agv_output_dim,
        ).to(self.device)

        # Optimizers: actor + GNN for each sub-agent
        self.optimizer_seq = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) +
            list(self.seq_actor.parameters()),
            lr=lr_actor,
        )
        self.optimizer_route = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) +
            list(self.route_actor.parameters()),
            lr=lr_actor,
        )
        self.optimizer_agv = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) +
            list(self.agv_actor.parameters()),
            lr=lr_actor,
        )

        # Separate GNN optimizer for save/load state and optional standalone use.
        self.optimizer_gnn = torch.optim.Adam(
            self.gnn_encoder.parameters(), lr=lr_gnn,
        )

    # ----------------------------------------------------------
    # Feature Extraction (unchanged from GraphPPOAgent)
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
    # Core: Decision Flow (GRPO policy sampling)
    # ----------------------------------------------------------

    @staticmethod
    def _snapshot_graph(data: Data) -> Data:
        """Copy rollout graph tensors to CPU for later re-encoding."""
        return Data(
            x=data.x.detach().cpu().clone(),
            node_type=data.node_type.detach().cpu().clone(),
            edge_index=data.edge_index.detach().cpu().clone(),
        )

    def _encode_flat_state(self, graph_data: Data) -> torch.Tensor:
        """Encode aggregate node statistics without graph message passing.

        The flat ablation keeps the actor input dimensionality unchanged while
        removing edge-dependent message passing. It uses only permutation-
        invariant feature means/stds and node-type proportions, then pads or
        repeats this vector to ``state_dim``. No second learned encoder is
        introduced, so the comparison isolates relational graph encoding.
        """
        x = graph_data.x.to(self.device).float()
        node_type = graph_data.node_type.to(self.device).long()
        if x.ndim != 2 or x.shape[0] == 0:
            return torch.zeros(self.state_dim, device=self.device)
        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False)
        counts = torch.stack([(node_type == kind).float().mean() for kind in range(4)])
        flat = torch.cat([mean, std, counts], dim=0)
        repeats = (self.state_dim + flat.numel() - 1) // flat.numel()
        return flat.repeat(repeats)[:self.state_dim]

    def _encode_state_graph(self, graph_data: Data,
                            node_type_offsets: Dict[str, Tuple[int, int]]):
        if self.state_encoder_mode == 'flat':
            return self._encode_flat_state(graph_data), None
        return self.gnn_encoder(graph_data, node_type_offsets)

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

        with torch.no_grad():
            global_state, node_embs = self._encode_state_graph(
                graph_result.data, graph_result.node_type_offsets,
            )

        global_state_np = global_state.cpu().numpy()
        graph_snapshot = self._snapshot_graph(graph_result.data)
        graph_node_type_offsets = dict(graph_result.node_type_offsets)

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
            self._store_transition(
                global_state_np, None, None, None, None, None,
                graph_data=graph_snapshot,
                graph_node_type_offsets=graph_node_type_offsets,
            )
            return [], DEFAULT_STEP_TIME

        # 3. Sequencing: rank operations by policy
        seq_features = self._extract_seq_features(machines, jobs, current_time)

        with torch.no_grad():
            gs_t = torch.tensor(global_state_np, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            sf_t = torch.tensor(seq_features, dtype=torch.float32,
                                device=self.device).unsqueeze(0)

            # Get action logits (no critic in GRPO)
            seq_logits = self.seq_actor(gs_t, sf_t).squeeze(0)

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

        # Build a priority order that is consistent with the sequence action
        # whose log-probability was recorded above.  Previously training
        # sampled ``seq_action`` and then discarded it in favour of an
        # independent random permutation, so the replayed old log-prob did not
        # correspond to the executed action.  The sequence head now chooses
        # the first operation; the remaining operations are ordered by their
        # current logits.  Only that first choice is stored as a sequence
        # transition below, while route/AGV heads still receive every O-M-V
        # decision in the batch.
        with torch.no_grad():
            all_logits = seq_logits[:len(ready_ops)]
            first_index = int(seq_action.item())
            remaining = [i for i in range(len(ready_ops)) if i != first_index]
            remaining.sort(key=lambda i: float(all_logits[i].item()), reverse=True)
            priority_order = [first_index] + remaining

        # 4. Pre-compute shared data structures once (avoids O(n^2))
        op_to_job, competing_ops, avg_queue, job_progress = self._precompute_context(machines, jobs)

        # 5. For each op in priority order: Route + AGV Select
        assigned_agvs = set()
        current_decisions_info = []

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

                valid_agv_logits = agv_logits[:, 0]  # first logit per candidate

                agv_policy_enabled = self.agv_selection_mode == 'policy'
                if self.task_mode == TRAINING and agv_policy_enabled:
                    dist = Categorical(logits=valid_agv_logits)
                    agv_action = dist.sample()
                    agv_log_prob = dist.log_prob(agv_action).item()
                else:
                    agv_action = torch.argmax(valid_agv_logits)
                    if self.agv_selection_mode == 'first_available':
                        agv_action = torch.tensor(0, device=self.device)
                    agv_log_prob = 0.0

            selected_agv = available_agvs[agv_action.item()]

            # In the fixed first-available ablation, the AGV is still used by
            # the environment, but this is a deterministic rule rather than
            # a sampled policy action.  Do not put a fabricated log-probability
            # into the rollout buffer: doing so would update the AGV actor
            # against an action that the actor did not choose.
            stored_agv_action = agv_action.item() if agv_policy_enabled else -1
            stored_agv_log_prob = agv_log_prob if agv_policy_enabled else 0.0

            decisions.append((op, selected_agv, selected_machine))
            assigned_agvs.add(selected_agv.id)

            current_decisions_info.append({
                # The sequencing policy made one categorical choice for the
                # first priority position.  Subsequent positions are a
                # deterministic completion of that choice and must not be
                # replayed as if they had independently been sampled.
                'seq_action': rank_idx if rank_idx == priority_order[0] else -1,
                'seq_candidate_count': len(ready_ops),
                'route_action': route_action.item(),
                'agv_action': stored_agv_action,
                'seq_features': seq_features,
                'route_features': route_feats_np[route_action.item()],
                'route_all_candidate_features': route_feats_np,  # full candidate set
                'agv_features': agv_feats_np[agv_action.item()],
                'agv_all_candidate_features': agv_feats_np,      # full candidate set
                'seq_log_prob': seq_log_prob if rank_idx == priority_order[0] else 0.0,
                'route_log_prob': route_log_prob,
                'agv_log_prob': stored_agv_log_prob,
            })

        # Store transition data for training
        self._store_transition(
            global_state_np, seq_features, current_decisions_info,
            None, None, None,
            graph_data=graph_snapshot,
            graph_node_type_offsets=graph_node_type_offsets,
        )

        return decisions, DEFAULT_STEP_TIME

    # ----------------------------------------------------------
    # Reward (references DualDRLAgent design + graph-agent dense shaping)
    # ----------------------------------------------------------

    def reward(self, *args, **kwargs) -> float:
        """Compute composite reward referencing DualDRLAgent design.

        Compatible with both calling patterns:
        - agent.reward({}) or agent.reward() — uses self.context
        - agent.reward(env_info) — uses env_info with makespan/machines/jobs/agvs
        """
        env_info = None
        if args and isinstance(args[0], dict):
            env_info = args[0]
        elif 'env_info' in kwargs:
            env_info = kwargs['env_info']
        elif 'env' in kwargs:
            env_info = kwargs['env']

        r = self._compute_reward(env_info)
        self._episode_reward += r
        return r

    def _compute_reward(self, env_info: Optional[Dict] = None) -> float:
        """Pure reward computation without side effects.

        Reward design references DualDRLAgent:
        - Makespan/time penalty (DualDRLAgent routing reward)
        - Machine load balance / CV penalty (DualDRLAgent routing reward)
        - Machine utilization (DualDRLAgent routing reward)
        - Queue reduction / bottleneck relief (DualDRLAgent sequencing reward)
        Plus graph-agent dense shaping:
        - Completed job ratio
        - Waiting operations penalty
        - Active AGV ratio
        """
        reward = 0.0

        # Resolve state from env_info or self.context
        makespan = 0.0
        machines = None
        jobs = None
        agvs = None

        if env_info and isinstance(env_info, dict):
            makespan = env_info.get('makespan', 0.0)
            machines = env_info.get('machines', None)
            jobs = env_info.get('jobs', None)
            agvs = env_info.get('agvs', None)

        # Fallback to self.context if env_info is insufficient
        if machines is None and self.context and hasattr(self.context, 'machines'):
            machines = self.context.machines
        if jobs is None and self.context and hasattr(self.context, 'jobs'):
            jobs = self.context.jobs
        if agvs is None and self.context and hasattr(self.context, 'agvs'):
            agvs = self.context.agvs
        if makespan == 0.0 and self.context and hasattr(self.context, 'env_timeline'):
            makespan = self.context.env_timeline

        # Controlled reward ablation. This is a dense elapsed-time penalty;
        # call it ``time-only`` in the paper rather than exact terminal
        # C_max optimization.
        if self.reward_mode == 'time_only':
            return -0.005 * makespan

        # ---- DualDRLAgent-style routing reward ----
        # Makespan penalty
        reward -= 0.005 * makespan

        # Machine utilization
        if machines:
            total_m = len(machines) if machines else 1
            working = sum(1 for m in machines if m.status == MachineStatus.WORKING)
            reward += 0.5 * (working / total_m)

            # Load balance — penalize high CV of machine queues (DualDRLAgent: -0.3 * load_std)
            loads = [len(m.input_queue) for m in machines]
            if len(loads) > 1 and np.mean(loads) > 0:
                cv = float(np.std(loads)) / (float(np.mean(loads)) + 1e-5)
                reward -= 3.0 * cv
        else:
            total_m = 1

        # ---- DualDRLAgent-style sequencing reward ----
        # Queue reduction / bottleneck relief
        total_queue_len = sum(len(m.input_queue) for m in machines) if machines else 0
        if hasattr(self, '_prev_queue_len'):
            queue_reduction = -(total_queue_len - self._prev_queue_len)
            reward += 0.5 * queue_reduction
            # Bottleneck relief bonus (DualDRLAgent: +0.1 if queue shrank)
            if queue_reduction > 0:
                reward += 0.1
        self._prev_queue_len = total_queue_len

        # ---- Graph-agent dense shaping ----
        # Completed job ratio
        if jobs:
            total = len(jobs) if jobs else 1
            completed = sum(1 for j in jobs if j.is_finished())
            reward += 50.0 * (completed / total)

            # Waiting operations penalty
            waiting = sum(
                1 for j in jobs
                for i in range(j.get_operation_count())
                if j.get_operation(i).get_status() == OperationStatus.WAITING
            )
            reward -= 0.5 * waiting

        # Active AGV ratio
        if agvs:
            total_a = len(agvs) if agvs else 1
            active = sum(1 for a in agvs if a.get_status() != AGVStatus.READY)
            reward += 1.5 * (active / total_a)

        return reward

    # ----------------------------------------------------------
    # Transition Collection
    # ----------------------------------------------------------

    def _store_transition(self, global_state_np, seq_features, decisions_info,
                          next_global_state_np, next_seq_features, reward,
                          graph_data: Optional[Data] = None,
                          graph_node_type_offsets: Optional[Dict[str, Tuple[int, int]]] = None,
                          done_override: Optional[bool] = None):
        """Store current step data and collect transition from previous step into rollout buffer.

        For GRPO, we push ALL decisions (not just the first) per step,
        and track group_id for group-relative advantage normalization.
        """
        if self.task_mode != TRAINING:
            self._prev_global_state = global_state_np
            self._prev_seq_features = seq_features
            self._prev_graph_data = graph_data
            self._prev_graph_node_type_offsets = graph_node_type_offsets
            self._prev_decisions_info = []
            return

        # Count calls from the environment, rather than action transitions:
        # waiting steps may legitimately produce no transition while still
        # advancing the rollout horizon.
        self._rollout_step_count += 1

        # Collect transitions from previous step into rollout buffer
        if self._prev_global_state is not None and self._prev_decisions_info:
            step_reward = self._compute_reward()

            # Normalize reward
            self.reward_normalizer.update(np.array([step_reward]))
            norm_reward = self.reward_normalizer.normalize(np.array([step_reward]))[0]

            done = bool(done_override) if done_override is not None else False
            if done_override is None and self.context and hasattr(self.context, 'jobs'):
                done = all(j.is_finished() for j in self.context.jobs)

            n_decisions = len(self._prev_decisions_info)
            shared_reward = norm_reward / max(n_decisions, 1)

            # Push every decision from this environment step into the same
            # decision-batch group.  The next environment state starts a new
            # group, so a group never crosses state boundaries.
            transition_group_id = (
                self._strict_group_id
                if self.strict_group_sampling and self._strict_group_id is not None
                else self._grpo_group_id
            )
            for info in self._prev_decisions_info:
                route_candidate_feats = info.get('route_all_candidate_features')
                agv_candidate_feats = info.get('agv_all_candidate_features')

                self.rollout_buffer.push(
                    state=self._prev_global_state,
                    seq_features=self._prev_seq_features,
                    reward=shared_reward,
                    done=float(done),
                    group_id=transition_group_id,
                    trajectory_id=(
                        self._strict_trajectory_id
                        if self.strict_group_sampling else None
                    ),
                    graph_data=self._prev_graph_data,
                    graph_node_type_offsets=self._prev_graph_node_type_offsets,
                    # Sequencing
                    seq_action=info.get('seq_action', -1),
                    seq_log_prob=info.get('seq_log_prob', 0.0),
                    seq_candidate_count=info.get('seq_candidate_count', 0),
                    # Routing
                    route_action=info.get('route_action', -1),
                    route_log_prob=info.get('route_log_prob', 0.0),
                    route_all_candidate_feats=route_candidate_feats,
                    # AGV
                    agv_action=info.get('agv_action', -1),
                    agv_log_prob=info.get('agv_log_prob', 0.0),
                    agv_all_candidate_feats=agv_candidate_feats,
                )

            if not self.strict_group_sampling:
                self._grpo_group_id += 1

        # Update previous state
        self._prev_global_state = global_state_np
        self._prev_seq_features = seq_features
        self._prev_graph_data = graph_data
        self._prev_graph_node_type_offsets = graph_node_type_offsets
        self._prev_decisions_info = decisions_info if decisions_info else []

    def after_sample(self, *args, **kwargs):
        """Post-sample hook: no-op (transition collection is done inside _store_transition)."""
        pass

    def before_sample(self, *args, **kwargs):
        """Pre-sample hook: no-op."""
        pass

    # ----------------------------------------------------------
    # GRPO Training
    # ----------------------------------------------------------

    def train(self, *args, **kwargs):
        """Train all three Actors using GRPO (critic-free, group-relative advantage).

        Triggered every rollout_steps environment steps.
        1. Compute group-relative advantages (no critic/GAE).
        2. Perform grpo_epochs of clipped surrogate updates for each sub-agent.
        """
        if self.task_mode != TRAINING or self.defer_updates:
            return

        # ``rollout_steps`` is an environment-decision horizon. A single
        # environment step can emit zero, one, or several action transitions,
        # so using ``len(rollout_buffer)`` here would make training frequency
        # depend on the number of AGVs/READY operations and can prevent any
        # update on instances with sparse decisions.
        if self._rollout_step_count < self.rollout_steps:
            return
        if len(self.rollout_buffer) == 0:
            return

        # Compute group-relative advantages (no critic needed)
        if self.strict_group_sampling:
            self.rollout_buffer.compute_strict_group_relative_advantages(
                gamma=self.gamma,
            )
        else:
            self.rollout_buffer.compute_group_relative_advantages(
                gamma=self.gamma, min_group_size=self.min_group_size,
            )
        self.training_history['advantage_diagnostics'].append(
            dict(self.rollout_buffer.advantage_diagnostics)
        )

        # GRPO update for all three sub-agents
        total_policy_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        for epoch in range(self.grpo_epochs):
            seq_stats = self._grpo_update_sub_agent('seq')
            route_stats = self._grpo_update_sub_agent('route')
            agv_stats = self._grpo_update_sub_agent('agv')

            for stats in [seq_stats, route_stats, agv_stats]:
                if stats is not None:
                    total_policy_loss += stats['policy_loss']
                    total_entropy += stats['entropy']
                    total_kl += stats['kl_approx']
                    n_updates += 1

            # 主动释放 GIL，防止 CPU 密集的 GRPO 训练阻塞 asyncio 事件循环
            time.sleep(0)

        self._train_step += 1

        # Record training stats
        if n_updates > 0:
            avg_policy_loss = total_policy_loss / n_updates
            avg_entropy = total_entropy / n_updates
            avg_kl = total_kl / n_updates
            avg_total = avg_policy_loss + self.kl_coeff * avg_kl - self.entropy_coeff * avg_entropy

            self.training_history['policy_loss'].append(avg_policy_loss)
            self.training_history['entropy'].append(avg_entropy)
            self.training_history['kl_approx'].append(avg_kl)
            self.training_history['total_loss'].append(avg_total)

        # Clear rollout buffer after update
        self.rollout_buffer.reset()
        self._rollout_step_count = 0

    def flush_pending_transition(self, done: bool = True):
        """Commit the final action of an episode to the rollout buffer.

        ``env.step`` terminates immediately after the final action, so there
        may be no subsequent ``sample`` call to collect that action's reward.
        Strict multi-rollout collection calls this method before resetting the
        environment, ensuring every trajectory has an explicit terminal
        transition.
        """
        if self.task_mode != TRAINING or self._prev_global_state is None:
            return
        self._store_transition(
            self._prev_global_state,
            self._prev_seq_features,
            [], None, None, None,
            graph_data=self._prev_graph_data,
            graph_node_type_offsets=self._prev_graph_node_type_offsets,
            done_override=done,
        )

    def _reencode_rollout_states(self, data: Dict[str, Any], indices: List[int]) -> torch.Tensor:
        """Re-encode stored graph snapshots for a policy-update mini-batch.

        Rollout action selection intentionally runs under ``no_grad``.  During
        the update, graph snapshots are replayed through the current encoder so
        the policy loss can provide gradients to GNN parameters.  Missing
        snapshots are supported for backward compatibility with synthetic or
        legacy buffers and fall back to the stored global vectors.
        """
        if not self.train_gnn:
            return data['states'][indices]

        graph_data = data.get('graph_data', [])
        graph_offsets = data.get('graph_node_type_offsets', [])
        encoded = []
        for idx in indices:
            graph = graph_data[idx] if idx < len(graph_data) else None
            offsets = graph_offsets[idx] if idx < len(graph_offsets) else None
            if graph is None or offsets is None:
                encoded.append(data['states'][idx])
                continue
            graph = graph.to(self.device)
            global_state, _ = self._encode_state_graph(graph, offsets)
            encoded.append(global_state)
        return torch.stack(encoded, dim=0)

    def _grpo_update_sub_agent(self, sub_agent: str) -> Optional[Dict[str, float]]:
        """Perform one GRPO update epoch for a sub-agent.

        Uses group-relative advantages instead of critic-based GAE.

        For route/AGV sub-agents, reconstructs the categorical distribution
        over the stored candidate set to ensure sampling/training consistency.

        Args:
            sub_agent: 'seq', 'route', or 'agv'

        Returns:
            Dictionary with policy_loss, entropy, kl_approx, or None if insufficient data
        """
        data = self.rollout_buffer.get_tensors(sub_agent)
        n_samples = data['states'].shape[0]

        if n_samples < self.grpo_batch_size:
            # Use all data if not enough for a mini-batch
            mini_batches = [None]
        else:
            n_batches = max(1, n_samples // self.grpo_batch_size)
            indices = np.random.permutation(n_samples)
            mini_batches = [indices[i * self.grpo_batch_size:(i + 1) * self.grpo_batch_size]
                           for i in range(n_batches)]

        total_policy_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_batches_actual = 0

        # Select the correct networks and optimizer
        if sub_agent == 'seq':
            actor = self.seq_actor
            optimizer = self.optimizer_seq
            feat_key = 'seq_features'
            max_output_dim = self.max_seq_output_dim
        elif sub_agent == 'route':
            actor = self.route_actor
            optimizer = self.optimizer_route
            max_output_dim = self.max_route_output_dim
        elif sub_agent == 'agv':
            actor = self.agv_actor
            optimizer = self.optimizer_agv
            max_output_dim = self.max_agv_output_dim

        # Set networks to training mode
        self.gnn_encoder.train()
        actor.train()

        for batch_indices in mini_batches:
            if sub_agent == 'seq':
                if batch_indices is None:
                    sample_indices = list(range(n_samples))
                else:
                    sample_indices = batch_indices.tolist()
                encoded_states = self._reencode_rollout_states(data, sample_indices)

                # Sequencing: standard categorical distribution over fixed output dims
                if batch_indices is None:
                    states = encoded_states
                    features = data[feat_key]
                    actions = data['actions']
                    old_log_probs = data['old_log_probs']
                    candidate_counts = data['candidate_counts']
                    advantages = data['advantages']
                else:
                    states = encoded_states
                    features = data[feat_key][batch_indices]
                    actions = data['actions'][batch_indices]
                    old_log_probs = data['old_log_probs'][batch_indices]
                    candidate_counts = data['candidate_counts'][batch_indices]
                    advantages = data['advantages'][batch_indices]

                # The sequencing policy is sampled only over the currently
                # executable operations.  Reconstruct that same masked
                # categorical distribution during the update; treating the
                # fixed output tail as valid would assign probability mass to
                # operations that were not candidates at sampling time.
                candidate_counts = candidate_counts.clamp(min=1, max=max_output_dim)
                valid_mask = (actions >= 0) & (actions < candidate_counts)
                if valid_mask.sum() == 0:
                    continue
                states = states[valid_mask]
                features = features[valid_mask]
                actions = actions[valid_mask]
                old_log_probs = old_log_probs[valid_mask]
                candidate_counts = candidate_counts[valid_mask]
                advantages = advantages[valid_mask]

                # Clamp actions to valid range
                actions = actions.clamp(0, max_output_dim - 1)

                # Evaluate current policy
                logits = actor(states, features)
                output_indices = torch.arange(
                    logits.shape[-1], device=self.device
                ).unsqueeze(0)
                masked_logits = logits.masked_fill(
                    output_indices >= candidate_counts.unsqueeze(1),
                    float('-inf'),
                )
                dist = Categorical(logits=masked_logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

            else:
                # Route / AGV: reconstruct distribution over stored candidate sets
                all_candidate_features = data[
                    f'{sub_agent}_all_candidate_features'
                ]

                if batch_indices is None:
                    batch_idx_list = list(range(n_samples))
                else:
                    batch_idx_list = batch_indices.tolist()

                encoded_states = self._reencode_rollout_states(data, batch_idx_list)
                encoded_state_by_index = {
                    idx: encoded_states[pos] for pos, idx in enumerate(batch_idx_list)
                }

                new_log_probs_list = []
                old_log_probs_list = []
                advantages_list = []
                entropy_values = []
                n_valid = 0

                for i in batch_idx_list:
                    action_i = data['actions'][i].item()
                    if action_i < 0:
                        continue

                    candidate_feats = all_candidate_features[i]
                    if candidate_feats.shape[0] <= 1 and candidate_feats.shape[1] <= 1:
                        continue

                    state_i = encoded_state_by_index[i]
                    advantage_i = data['advantages'][i]
                    old_lp_i = data['old_log_probs'][i].item()

                    # Reconstruct categorical distribution over candidates
                    n_cand = candidate_feats.shape[0]
                    cand_t = torch.tensor(candidate_feats, dtype=torch.float32,
                                          device=self.device)
                    gs_batch = state_i.unsqueeze(0).expand(n_cand, -1)

                    logits = actor(gs_batch, cand_t)
                    valid_logits = logits[:, 0]  # first logit per candidate
                    dist = Categorical(logits=valid_logits)

                    action_clamped = min(action_i, n_cand - 1)
                    new_lp = dist.log_prob(torch.tensor(action_clamped, device=self.device))
                    ent = dist.entropy()

                    new_log_probs_list.append(new_lp.unsqueeze(0))
                    old_log_probs_list.append(old_lp_i)
                    advantages_list.append(advantage_i.item())
                    # Keep entropy as a tensor so the entropy coefficient
                    # contributes gradients to the actor and shared encoder.
                    entropy_values.append(ent)
                    n_valid += 1

                if n_valid == 0:
                    continue

                new_log_probs = torch.cat(new_log_probs_list, dim=0)
                old_log_probs = torch.tensor(old_log_probs_list, dtype=torch.float32,
                                            device=self.device)
                advantages = torch.tensor(advantages_list, dtype=torch.float32,
                                         device=self.device)
                entropy = torch.stack(entropy_values).mean()

            # GRPO clipped surrogate loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Approximate KL divergence (for monitoring / optional penalty)
            approx_kl = (old_log_probs - new_log_probs).mean()

            # Total loss: policy + optional KL penalty - entropy bonus
            loss = policy_loss + self.kl_coeff * approx_kl - self.entropy_coeff * entropy

            # Optimizer step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.gnn_encoder.parameters()) + list(actor.parameters()),
                self.max_grad_norm,
            )
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item() if isinstance(entropy, torch.Tensor) else entropy
            total_kl += approx_kl.item() if isinstance(approx_kl, torch.Tensor) else approx_kl
            n_batches_actual += 1

        if n_batches_actual == 0:
            return None

        return {
            'policy_loss': total_policy_loss / n_batches_actual,
            'entropy': total_entropy / n_batches_actual,
            'kl_approx': total_kl / n_batches_actual,
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
                if self.model_path:
                    path = self.model_path
                else:
                    agent_name = self.name or "GraphGRPOAgent"
                    agent_dir = f"training_logs/models/{agent_name}"
                    os.makedirs(agent_dir, exist_ok=True)
                    path = f"{agent_dir}/agent_model.pt"

            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            checkpoint = {
                'gnn_encoder': self.gnn_encoder.state_dict(),
                'seq_actor': self.seq_actor.state_dict(),
                'route_actor': self.route_actor.state_dict(),
                'agv_actor': self.agv_actor.state_dict(),
                'optimizer_gnn': self.optimizer_gnn.state_dict(),
                'optimizer_seq': self.optimizer_seq.state_dict(),
                'optimizer_route': self.optimizer_route.state_dict(),
                'optimizer_agv': self.optimizer_agv.state_dict(),
                'hyperparams': {
                    'hidden_dim': self.hidden_dim,
                    'gamma': self.gamma,
                    'clip_epsilon': self.clip_epsilon,
                    'grpo_epochs': self.grpo_epochs,
                    'grpo_batch_size': self.grpo_batch_size,
                    'grpo_group_size': self.grpo_group_size,
                    'min_group_size': self.min_group_size,
                    'entropy_coeff': self.entropy_coeff,
                    'kl_coeff': self.kl_coeff,
                    'max_grad_norm': self.max_grad_norm,
                    'rollout_steps': self.rollout_steps,
                    'max_seq_output_dim': self.max_seq_output_dim,
                    'max_route_output_dim': self.max_route_output_dim,
                    'max_agv_output_dim': self.max_agv_output_dim,
                    'seq_feature_dim': self.seq_feature_dim,
                    'route_feature_dim': self.route_feature_dim,
                    'agv_feature_dim': self.agv_feature_dim,
                    'allow_agv_reassignment': self.allow_agv_reassignment,
                    'train_gnn': self.train_gnn,
                    'strict_group_sampling': self.strict_group_sampling,
                    'state_encoder_mode': self.state_encoder_mode,
                    'reward_mode': self.reward_mode,
                    'agv_selection_mode': self.agv_selection_mode,
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
            LOGGER.info(f"[GraphGRPOAgent] Model saved to {path}")
            return True
        except Exception as e:
            LOGGER.error(f"[GraphGRPOAgent] Save failed: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load all network state_dicts and hyperparameters from a .pt file.

        Tolerant of missing critic keys (for loading from PPO checkpoints,
        only GNN/actor weights are loaded).
        """
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])
            self.seq_actor.load_state_dict(checkpoint['seq_actor'])
            self.route_actor.load_state_dict(checkpoint['route_actor'])
            self.agv_actor.load_state_dict(checkpoint['agv_actor'])

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
            self.grpo_epochs = hp.get('grpo_epochs', self.grpo_epochs)
            self.grpo_batch_size = hp.get('grpo_batch_size', self.grpo_batch_size)
            self.grpo_group_size = hp.get('grpo_group_size', self.grpo_group_size)
            self.min_group_size = hp.get('min_group_size', self.min_group_size)
            self.entropy_coeff = hp.get('entropy_coeff', self.entropy_coeff)
            self.kl_coeff = hp.get('kl_coeff', self.kl_coeff)
            self.train_gnn = hp.get('train_gnn', self.train_gnn)
            self.strict_group_sampling = hp.get(
                'strict_group_sampling', self.strict_group_sampling
            )
            self.state_encoder_mode = hp.get('state_encoder_mode', self.state_encoder_mode)
            self.reward_mode = hp.get('reward_mode', self.reward_mode)
            self.agv_selection_mode = hp.get('agv_selection_mode', self.agv_selection_mode)

            if 'reward_normalizer' in checkpoint:
                rn = checkpoint['reward_normalizer']
                self.reward_normalizer.mean = rn['mean']
                self.reward_normalizer.var = rn['var']
                self.reward_normalizer.count = rn['count']

            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            # Checkpoints created before advantage diagnostics were added do
            # not contain the new history key. Keep loading backward
            # compatible so a legacy checkpoint can be fine-tuned or used in
            # the short verification run without failing on the first update.
            self.training_history.setdefault('advantage_diagnostics', [])

            self._train_step = checkpoint.get('train_step', 0)

            LOGGER.info(f"[GraphGRPOAgent] Model loaded from {path}")
            return True
        except Exception as e:
            LOGGER.error(f"[GraphGRPOAgent] Load failed: {e}")
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
        if self.training_history.get('policy_loss'):
            metrics['policy_loss'] = self.training_history['policy_loss'][-1]
        if self.training_history.get('entropy'):
            metrics['entropy'] = self.training_history['entropy'][-1]
        if self.training_history.get('total_loss'):
            metrics['total_loss'] = self.training_history['total_loss'][-1]
        if self.training_history.get('kl_approx'):
            metrics['kl_approx'] = self.training_history['kl_approx'][-1]
        return metrics

    def new_episode(self):
        """Reset episode-level state. Call at the start of each episode."""
        self._episode_reward = 0.0
        self._prev_global_state = None
        self._prev_seq_features = None
        self._prev_graph_data = None
        self._prev_graph_node_type_offsets = None
        self._prev_decisions_info = []
        self._prev_queue_len = 0

        # Record episode stats.
        if self.training_history['episodes']:
            last_ep = self.training_history['episodes'][-1] + 1
        else:
            last_ep = 1
        self.training_history['episodes'].append(last_ep)

        makespan = 0.0
        if self.context and hasattr(self.context, 'env_timeline'):
            makespan = self.context.env_timeline
        self.training_history['makespans'].append(makespan)

    def begin_strict_rollout(self, group_id: int, trajectory_id: int):
        """Attach subsequent transitions to one strict GRPO rollout.

        The caller must create independent environments from the same initial
        instance/state, keep the policy fixed while collecting all members of
        a group, and call :meth:`train` only after the group is complete.
        """
        if not self.strict_group_sampling:
            raise RuntimeError("set strict_group_sampling=true before collecting strict rollouts")
        self._strict_group_id = int(group_id)
        self._strict_trajectory_id = int(trajectory_id)

    def end_strict_rollout(self):
        """Stop attaching transitions to the current strict rollout."""
        self._strict_group_id = None
        self._strict_trajectory_id = None

    def __repr__(self):
        return (f"<GraphGRPOAgent id={self.agent_id} name={self.name} "
                f"device={self.device} group_size={self.grpo_group_size}>")
