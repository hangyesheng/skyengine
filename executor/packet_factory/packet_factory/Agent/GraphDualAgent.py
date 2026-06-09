"""
GraphDual Agent - GNN-enhanced Dual DQN Agent for FJSP

Combines design ideas from GraphDPAgent and DualDRLAgent:
- GNN-based state encoding (from GraphDPAgent)
- Dual decomposition: Sequencing + Routing + AGV Selection (from DualDRLAgent)
- Three cooperating DQN sub-agents, each with online/target nets
- Fixes: sequencing actually drives operation ordering, AGV assignment is learned,
  current_time from env, reward normalization, all hyperparams from YAML

Architecture:
  FactoryGraphBuilder -> GNNStateEncoder -> global_state [256] + node_embs [N, 64]
                                                |
                     +--------------------------+--------------------------+
                     |                          |                          |
              SequencingDQN              RoutingDQN              AGVSelectionDQN
           (operation priority)      (machine assignment)          (AGV pick)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from .BaseAgent import BaseAgent, DEFAULT_STEP_TIME, FRONTEND, BACKEND, TRAINING, INFERENCE
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
# Factory Graph Builder (reused from GraphDPAgent)
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
# GNN State Encoder (reused from GraphDPAgent)
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
# Replay Buffer (from DualDRLAgent pattern)
# ============================================================

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer: List[Dict[str, np.ndarray]] = []

    def push(self, experience: Dict[str, np.ndarray]):
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        return {
            key: torch.tensor(np.array([exp[key] for exp in batch]),
                              dtype=torch.float32, device=self.device)
            for key in batch[0]
        }

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQN Networks
# ============================================================

class SequencingDQN(nn.Module):
    """Sequencing Q-network: ranks ready operations by priority.

    Input: global_state [state_dim] + seq_features [seq_feature_dim]
    Output: Q-values [max_seq_output_dim] over operation slots
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
        # InstanceNorm on each input
        gs = self.state_norm(global_state.unsqueeze(1)).squeeze(1)
        sf = self.feat_norm(seq_features.unsqueeze(1)).squeeze(1)
        x = torch.cat([gs, sf], dim=-1)

        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc_out(x)


class RoutingDQN(nn.Module):
    """Routing Q-network: selects the best machine for an operation.

    Input: global_state [state_dim] + route_features [route_feature_dim]
    Output: Q-values [max_route_output_dim] over machine slots
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


class AGVSelectionDQN(nn.Module):
    """AGV Selection Q-network: selects the best AGV for a given (op, machine) assignment.

    Input: global_state [state_dim] + agv_features [agv_feature_dim]
    Output: Q-values [max_agv_output_dim] over AGV slots
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
# GraphDualAgent
# ============================================================

@register_component("packet_factory.GraphDualAgent")
class GraphDualAgent(BaseAgent):
    """GNN-enhanced Dual DQN Agent for FJSP.

    Combines GNN state encoding (GraphDPAgent) with dual decomposition (DualDRLAgent):
    1. GNN encodes factory graph -> global_state [256] + node_embs
    2. SequencingDQN ranks READY operations by priority
    3. For each operation in priority order:
       - RoutingDQN selects the best capable machine
       - AGVSelectionDQN selects the best available AGV
    4. Training: three independent DQNs with replay buffers
    """

    def __init__(self, name=None, agent_id=None, context=None,
                 ui_mode: str = BACKEND, task_mode: str = TRAINING,
                 model_path: Optional[str] = None,
                 # Graph encoder
                 hidden_dim: int = 64,
                 # DQN common
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.998,
                 epsilon_min: float = 0.01,
                 # Learning rates
                 lr_gnn: float = 1e-4,
                 lr_seq: float = 1e-3,
                 lr_route: float = 1e-3,
                 lr_agv: float = 1e-3,
                 # Training
                 buffer_capacity: int = 5000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
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
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.max_seq_output_dim = max_seq_output_dim
        self.max_route_output_dim = max_route_output_dim
        self.max_agv_output_dim = max_agv_output_dim
        self.seq_feature_dim = seq_feature_dim
        self.route_feature_dim = route_feature_dim
        self.agv_feature_dim = agv_feature_dim
        self.allow_agv_reassignment = allow_agv_reassignment

        # Device
        if device is None or device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if self.device.type == 'cuda':
            LOGGER.info(f"[GraphDualAgent] Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            LOGGER.info(f"[GraphDualAgent] Using CPU")

        # Graph builder
        self.graph_builder = FactoryGraphBuilder(device=str(self.device))

        # GNN encoder (shared across all DQNs)
        self.gnn_encoder = GNNStateEncoder(
            hidden_dim=hidden_dim, num_layers=2,
        ).to(self.device)

        # Initialize all DQN sub-agents
        self._initialize_networks(
            lr_gnn, lr_seq, lr_route, lr_agv,
        )

        # Reward normalization
        self.reward_normalizer = RunningMeanStd()

        # State tracking for transition collection
        self._prev_global_state: Optional[np.ndarray] = None
        self._prev_seq_features: Optional[np.ndarray] = None
        self._prev_decisions_info: List[Dict] = []

        # Training statistics
        self.training_history: Dict[str, list] = {
            'episodes': [], 'seq_loss': [], 'route_loss': [],
            'agv_loss': [], 'total_loss': [], 'makespans': [], 'epsilon': [],
        }
        self._episode_reward = 0.0
        self._train_step = 0

        # Load model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            LOGGER.info(f"[GraphDualAgent] Loaded model from {model_path}")
        else:
            LOGGER.info(f"[GraphDualAgent] Initialized on {self.device}, "
                        f"hidden_dim={hidden_dim}, state_dim={self.state_dim}")

    def _initialize_networks(self, lr_gnn, lr_seq, lr_route, lr_agv):
        """Initialize all DQN sub-agents with online + target nets."""
        # Sequencing DQN
        self.seq_net = SequencingDQN(
            self.state_dim, self.seq_feature_dim, self.max_seq_output_dim,
        ).to(self.device)
        self.seq_target = SequencingDQN(
            self.state_dim, self.seq_feature_dim, self.max_seq_output_dim,
        ).to(self.device)
        self.seq_target.load_state_dict(self.seq_net.state_dict())
        self.seq_target.eval()

        # Routing DQN
        self.route_net = RoutingDQN(
            self.state_dim, self.route_feature_dim, self.max_route_output_dim,
        ).to(self.device)
        self.route_target = RoutingDQN(
            self.state_dim, self.route_feature_dim, self.max_route_output_dim,
        ).to(self.device)
        self.route_target.load_state_dict(self.route_net.state_dict())
        self.route_target.eval()

        # AGV Selection DQN
        self.agv_net = AGVSelectionDQN(
            self.state_dim, self.agv_feature_dim, self.max_agv_output_dim,
        ).to(self.device)
        self.agv_target = AGVSelectionDQN(
            self.state_dim, self.agv_feature_dim, self.max_agv_output_dim,
        ).to(self.device)
        self.agv_target.load_state_dict(self.agv_net.state_dict())
        self.agv_target.eval()

        # Optimizers: GNN encoder shares lr_gnn with all sub-agents
        self.optimizer_gnn = torch.optim.Adam(
            self.gnn_encoder.parameters(), lr=lr_gnn,
        )
        self.optimizer_seq = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) + list(self.seq_net.parameters()),
            lr=lr_seq,
        )
        self.optimizer_route = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) + list(self.route_net.parameters()),
            lr=lr_route,
        )
        self.optimizer_agv = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) + list(self.agv_net.parameters()),
            lr=lr_agv,
        )

        # Replay buffers
        self.seq_replay = ReplayBuffer(self.buffer_capacity, self.device)
        self.route_replay = ReplayBuffer(self.buffer_capacity, self.device)
        self.agv_replay = ReplayBuffer(self.buffer_capacity, self.device)

    # ----------------------------------------------------------
    # Feature Extraction
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
        # Average duration of waiting ops
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
    # Core: Decision Flow
    # ----------------------------------------------------------

    def sample(self, agvs: List[AGV], machines: List[Machine],
               jobs: List[Job]) -> Tuple[List[Tuple[Operation, AGV, Machine]], float]:
        """Three-phase decision: Sequence -> Route -> AGV Select.

        1. GNN encodes factory state
        2. SequencingDQN ranks ready operations by priority
        3. For each op in priority order:
           a. RoutingDQN selects machine
           b. AGVSelectionDQN selects AGV
        """
        decisions = []

        # 1. Build and encode current state
        factory_graph = self._get_factory_graph(agvs)
        current_time = self._get_current_time()

        graph_result = self.graph_builder.build(
            agvs, machines, jobs, factory_graph, current_time,
        )

        self.gnn_encoder.eval()
        self.seq_net.eval()
        self.route_net.eval()
        self.agv_net.eval()

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

        # 3. Sequencing: rank operations by Q-value
        seq_features = self._extract_seq_features(machines, jobs, current_time)

        with torch.no_grad():
            gs_t = torch.tensor(global_state_np, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            sf_t = torch.tensor(seq_features, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            seq_q = self.seq_net(gs_t, sf_t).squeeze(0).cpu().numpy()

        # Rank by Q-value (higher = higher priority)
        valid_q = seq_q[:len(ready_ops)]
        if self.task_mode == TRAINING and random.random() < self.epsilon:
            # Random permutation for exploration
            priority_order = list(range(len(ready_ops)))
            random.shuffle(priority_order)
        else:
            priority_order = list(np.argsort(-valid_q))  # descending Q

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
                gs_t = torch.tensor(global_state_np, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
                rf_t = torch.tensor(route_feats_np, dtype=torch.float32,
                                    device=self.device)
                # Batch: repeat global state for each candidate
                gs_batch = gs_t.expand(len(capable_machines), -1)
                route_q = self.route_net(gs_batch, rf_t).cpu().numpy()

            valid_route_q = route_q[:, :self.max_route_output_dim]
            route_q_per_machine = [valid_route_q[i, 0] for i in range(len(capable_machines))]

            if self.task_mode == TRAINING and random.random() < self.epsilon:
                route_idx = random.randint(0, len(capable_machines) - 1)
            else:
                route_idx = int(np.argmax(route_q_per_machine))

            selected_machine = capable_machines[route_idx]

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
                agv_q = self.agv_net(gs_batch, af_t).cpu().numpy()

            valid_agv_q = agv_q[:, :self.max_agv_output_dim]
            agv_q_per_agv = [valid_agv_q[i, 0] for i in range(len(available_agvs))]

            if self.task_mode == TRAINING and random.random() < self.epsilon:
                agv_idx = random.randint(0, len(available_agvs) - 1)
            else:
                agv_idx = int(np.argmax(agv_q_per_agv))

            selected_agv = available_agvs[agv_idx]

            decisions.append((op, selected_agv, selected_machine))
            assigned_agvs.add(selected_agv.id)

            current_decisions_info.append({
                'seq_action': rank_idx,
                'route_action': route_idx,
                'agv_action': agv_idx,
                'seq_features': seq_features,
                'route_features': route_feats_np[route_idx],
                'agv_features': agv_feats_np[agv_idx],
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
        """Store current step data and collect transition from previous step."""
        if self.task_mode != TRAINING:
            self._prev_global_state = global_state_np
            self._prev_seq_features = seq_features
            self._prev_decisions_info = []
            return

        # Collect transitions from previous step
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
            for info in self._prev_decisions_info:
                shared_reward = norm_reward / max(n_decisions, 1)

                # Sequencing transition
                self.seq_replay.push({
                    'state': self._prev_global_state,
                    'feat': self._prev_seq_features,
                    'action': np.array([info['seq_action']], dtype=np.float32),
                    'reward': np.array([shared_reward], dtype=np.float32),
                    'next_state': next_gs,
                    'next_feat': next_sf if next_sf is not None else self._prev_seq_features,
                    'done': np.array([float(done)], dtype=np.float32),
                })

                # Routing transition
                self.route_replay.push({
                    'state': self._prev_global_state,
                    'feat': info['route_features'],
                    'action': np.array([info['route_action']], dtype=np.float32),
                    'reward': np.array([shared_reward], dtype=np.float32),
                    'next_state': next_gs,
                    'next_feat': info['route_features'],  # same since env obs is empty
                    'done': np.array([float(done)], dtype=np.float32),
                })

                # AGV transition
                self.agv_replay.push({
                    'state': self._prev_global_state,
                    'feat': info['agv_features'],
                    'action': np.array([info['agv_action']], dtype=np.float32),
                    'reward': np.array([shared_reward], dtype=np.float32),
                    'next_state': next_gs,
                    'next_feat': info['agv_features'],
                    'done': np.array([float(done)], dtype=np.float32),
                })

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
    # Training
    # ----------------------------------------------------------

    def train(self, *args, **kwargs):
        """Train all three DQN sub-agents from their replay buffers."""
        if self.task_mode != TRAINING:
            return

        seq_loss = self._train_sequencing_dqn()
        route_loss = self._train_routing_dqn()
        agv_loss = self._train_agv_dqn()

        self._train_step += 1
        if self._train_step % self.target_update_freq == 0:
            self.seq_target.load_state_dict(self.seq_net.state_dict())
            self.route_target.load_state_dict(self.route_net.state_dict())
            self.agv_target.load_state_dict(self.agv_net.state_dict())
            LOGGER.info(f"[GraphDualAgent] Target nets updated at step {self._train_step}")

        # Record training stats
        total = 0.0
        if seq_loss is not None:
            self.training_history['seq_loss'].append(seq_loss)
            total += seq_loss
        if route_loss is not None:
            self.training_history['route_loss'].append(route_loss)
            total += route_loss
        if agv_loss is not None:
            self.training_history['agv_loss'].append(agv_loss)
            total += agv_loss
        if seq_loss is not None or route_loss is not None or agv_loss is not None:
            self.training_history['total_loss'].append(total)

    def _train_sequencing_dqn(self):
        """Train Sequencing DQN from replay buffer."""
        batch = self.seq_replay.sample(self.batch_size)
        if batch is None:
            return None

        state = batch['state']
        feat = batch['feat']
        action = batch['action'].long()
        reward = batch['reward'].squeeze(-1)
        next_state = batch['next_state']
        next_feat = batch['next_feat']
        done = batch['done'].squeeze(-1)

        # Current Q
        q_values = self.seq_net(state, feat)
        q_value = q_values.gather(1, action.clamp(0, self.max_seq_output_dim - 1)).squeeze(1)

        # Target Q
        with torch.no_grad():
            next_q = self.seq_target(next_state, next_feat)
            max_next_q = next_q.max(1)[0]
            target_q = reward + self.gamma * max_next_q * (1 - done)

        loss = F.mse_loss(q_value, target_q)

        self.optimizer_seq.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.seq_net.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.gnn_encoder.parameters(), max_norm=1.0)
        self.optimizer_seq.step()

        return loss.item()

    def _train_routing_dqn(self):
        """Train Routing DQN from replay buffer."""
        batch = self.route_replay.sample(self.batch_size)
        if batch is None:
            return None

        state = batch['state']
        feat = batch['feat']
        action = batch['action'].long()
        reward = batch['reward'].squeeze(-1)
        next_state = batch['next_state']
        next_feat = batch['next_feat']
        done = batch['done'].squeeze(-1)

        q_values = self.route_net(state, feat)
        q_value = q_values.gather(1, action.clamp(0, self.max_route_output_dim - 1)).squeeze(1)

        with torch.no_grad():
            next_q = self.route_target(next_state, next_feat)
            max_next_q = next_q.max(1)[0]
            target_q = reward + self.gamma * max_next_q * (1 - done)

        loss = F.mse_loss(q_value, target_q)

        self.optimizer_route.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.route_net.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.gnn_encoder.parameters(), max_norm=1.0)
        self.optimizer_route.step()

        return loss.item()

    def _train_agv_dqn(self):
        """Train AGV Selection DQN from replay buffer."""
        batch = self.agv_replay.sample(self.batch_size)
        if batch is None:
            return None

        state = batch['state']
        feat = batch['feat']
        action = batch['action'].long()
        reward = batch['reward'].squeeze(-1)
        next_state = batch['next_state']
        next_feat = batch['next_feat']
        done = batch['done'].squeeze(-1)

        q_values = self.agv_net(state, feat)
        q_value = q_values.gather(1, action.clamp(0, self.max_agv_output_dim - 1)).squeeze(1)

        with torch.no_grad():
            next_q = self.agv_target(next_state, next_feat)
            max_next_q = next_q.max(1)[0]
            target_q = reward + self.gamma * max_next_q * (1 - done)

        loss = F.mse_loss(q_value, target_q)

        self.optimizer_agv.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agv_net.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.gnn_encoder.parameters(), max_norm=1.0)
        self.optimizer_agv.step()

        return loss.item()

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
                agent_name = self.name or "GraphDualAgent"
                agent_dir = f"training_logs/models/{agent_name}"
                os.makedirs(agent_dir, exist_ok=True)
                path = f"{agent_dir}/agent_model.pt"

            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            checkpoint = {
                'gnn_encoder': self.gnn_encoder.state_dict(),
                'seq_net': self.seq_net.state_dict(),
                'seq_target': self.seq_target.state_dict(),
                'route_net': self.route_net.state_dict(),
                'route_target': self.route_target.state_dict(),
                'agv_net': self.agv_net.state_dict(),
                'agv_target': self.agv_target.state_dict(),
                'optimizer_gnn': self.optimizer_gnn.state_dict(),
                'optimizer_seq': self.optimizer_seq.state_dict(),
                'optimizer_route': self.optimizer_route.state_dict(),
                'optimizer_agv': self.optimizer_agv.state_dict(),
                'hyperparams': {
                    'hidden_dim': self.hidden_dim,
                    'gamma': self.gamma,
                    'epsilon': self.epsilon,
                    'epsilon_decay': self.epsilon_decay,
                    'epsilon_min': self.epsilon_min,
                    'batch_size': self.batch_size,
                    'buffer_capacity': self.buffer_capacity,
                    'target_update_freq': self.target_update_freq,
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
            LOGGER.info(f"[GraphDualAgent] Model saved to {path}")
            return True
        except Exception as e:
            LOGGER.error(f"[GraphDualAgent] Save failed: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load all network state_dicts and hyperparameters from a .pt file."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])
            self.seq_net.load_state_dict(checkpoint['seq_net'])
            self.seq_target.load_state_dict(checkpoint['seq_target'])
            self.route_net.load_state_dict(checkpoint['route_net'])
            self.route_target.load_state_dict(checkpoint['route_target'])
            self.agv_net.load_state_dict(checkpoint['agv_net'])
            self.agv_target.load_state_dict(checkpoint['agv_target'])

            for opt_key, optimizer in [
                ('optimizer_gnn', self.optimizer_gnn),
                ('optimizer_seq', self.optimizer_seq),
                ('optimizer_route', self.optimizer_route),
                ('optimizer_agv', self.optimizer_agv),
            ]:
                if opt_key in checkpoint:
                    optimizer.load_state_dict(checkpoint[opt_key])

            hp = checkpoint.get('hyperparams', {})
            self.epsilon = hp.get('epsilon', self.epsilon)
            self.gamma = hp.get('gamma', self.gamma)

            if 'reward_normalizer' in checkpoint:
                rn = checkpoint['reward_normalizer']
                self.reward_normalizer.mean = rn['mean']
                self.reward_normalizer.var = rn['var']
                self.reward_normalizer.count = rn['count']

            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']

            self._train_step = checkpoint.get('train_step', 0)

            LOGGER.info(f"[GraphDualAgent] Model loaded from {path}")
            return True
        except Exception as e:
            LOGGER.error(f"[GraphDualAgent] Load failed: {e}")
            return False

    # ----------------------------------------------------------
    # Training Metrics
    # ----------------------------------------------------------

    def get_training_metrics(self) -> Dict[str, Any]:
        """Return training metrics for convergence detection."""
        metrics = {
            'episode_reward': self._episode_reward,
            'epsilon': self.epsilon,
            'training_history': self.training_history,
        }
        if self.training_history.get('seq_loss'):
            metrics['seq_loss'] = self.training_history['seq_loss'][-1]
        if self.training_history.get('route_loss'):
            metrics['route_loss'] = self.training_history['route_loss'][-1]
        if self.training_history.get('agv_loss'):
            metrics['agv_loss'] = self.training_history['agv_loss'][-1]
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
        self.training_history['epsilon'].append(self.epsilon)

        # Epsilon decay per episode
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        makespan = 0.0
        if self.context and hasattr(self.context, 'env_timeline'):
            makespan = self.context.env_timeline
        self.training_history['makespans'].append(makespan)

    def __repr__(self):
        return (f"<GraphDualAgent id={self.agent_id} name={self.name} "
                f"mode={self.mode} device={self.device}>")
