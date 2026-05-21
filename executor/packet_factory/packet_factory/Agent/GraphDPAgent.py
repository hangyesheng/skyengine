"""
GraphDP Agent - GNN-based Dynamic Programming DRL Agent for FJSP

Core idea:
- State = GNN-compressed encoding of the factory graph
- Action = strictly defined (Operation, AGV, Machine) assignment
- TransitionNet T(s,a)->s': learns state encoding transitions
- ValueNet V(s)->R: learns state value function
- DP Rollout: enumerate valid actions, predict future via T, evaluate via V
- Re-walk: after network updates, re-encode stored graphs and retrain

This is a model-based RL approach where:
- The GNN encoder compresses the factory graph into a latent state
- The transition model allows look-ahead planning without simulation
- The value model provides the objective for planning
- The DP-style rollout combines both for multi-step decision-making
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
import os
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass, field
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


@dataclass
class EpisodeStep:
    """One step within an episode's DP rollout."""
    graph_result: 'GraphBuildResult'
    global_state: torch.Tensor
    node_embs: torch.Tensor
    action_indices: List[Tuple[int, int, int]]
    action_embs: torch.Tensor
    reward: float = 0.0


@dataclass
class Transition:
    """A complete (s, a, r, s') transition for training."""
    graph_result_s: 'GraphBuildResult'
    graph_result_s_next: 'GraphBuildResult'
    action_info: Tuple[int, int, int]
    reward: float
    done: bool


# ============================================================
# Factory Graph Builder
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
        # Build node features and ID mappings
        machine_feats, machine_id_map = self._machine_features(machines, current_time)
        op_feats, op_id_map = self._operation_features(jobs)
        agv_feats, agv_id_map = self._agv_features(agvs)
        job_feats, job_id_map = self._job_features(jobs)

        n_m = len(machine_feats)
        n_o = len(op_feats)
        n_a = len(agv_feats)
        n_j = len(job_feats)

        # Pad to MAX_DIM and concatenate
        all_feats = np.concatenate([
            self._pad(machine_feats),
            self._pad(op_feats),
            self._pad(agv_feats),
            self._pad(job_feats),
        ], axis=0)

        # Node type indices: 0=machine, 1=operation, 2=agv, 3=job
        node_type = np.concatenate([
            np.zeros(n_m, dtype=np.int64),
            np.ones(n_o, dtype=np.int64),
            np.full(n_a, 2, dtype=np.int64),
            np.full(n_j, 3, dtype=np.int64),
        ])

        data = Data(
            x=torch.tensor(all_feats, dtype=torch.float32),
            node_type=torch.tensor(node_type, dtype=torch.int64),
        )

        # Build edges (all bidirectional)
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

        # Build id_to_node_idx mapping: (type_str, original_id) -> graph_node_idx
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
# GNN State Encoder
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

        # Type embeddings
        self.type_embedding = nn.Embedding(4, hidden_dim)

        # Type-specific input projections
        self.machine_proj = nn.Linear(machine_dim, hidden_dim)
        self.operation_proj = nn.Linear(operation_dim, hidden_dim)
        self.agv_proj = nn.Linear(agv_dim, hidden_dim)
        self.job_proj = nn.Linear(job_dim, hidden_dim)

        # SAGEConv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, data: Data,
                node_type_offsets: Dict[str, Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data: PyG Data with x [N, feature_dim] and node_type [N]
            node_type_offsets: {type_name: (start_idx, count)}

        Returns:
            global_state: [hidden_dim * 4] global state vector
            node_embs: [N, hidden_dim] node embeddings
        """
        x = data.x
        node_type = data.node_type

        # Project features by node type
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

        for type_idx, proj in enumerate([self.machine_proj, self.operation_proj,
                                          self.agv_proj, self.job_proj]):
            mask = (node_type == type_idx)
            if mask.any():
                h[mask] = F.leaky_relu(proj(x[mask]))

        # Add type embeddings
        type_embs = self.type_embedding(node_type)
        h = h + type_embs
        h = F.leaky_relu(h)

        # Message passing with residual connections
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, data.edge_index)
            h = norm(h + h_new)
            h = F.leaky_relu(h)

        # Per-type mean pooling -> concatenate
        parts = []
        for type_name in ['machine', 'operation', 'agv', 'job']:
            start, count = node_type_offsets[type_name]
            if count > 0:
                pooled = h[start:start + count].mean(dim=0)
            else:
                pooled = torch.zeros(self.hidden_dim, device=h.device)
            parts.append(pooled)

        global_state = torch.cat(parts, dim=-1)  # [hidden_dim * 4]
        return global_state, h


# ============================================================
# Action Encoder
# ============================================================

class ActionEncoder(nn.Module):
    """Encodes an action (op, agv, machine) using their GNN node embeddings."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, op_emb: torch.Tensor, agv_emb: torch.Tensor,
                machine_emb: torch.Tensor) -> torch.Tensor:
        """Single action encoding.

        Args:
            op_emb: [hidden_dim]
            agv_emb: [hidden_dim]
            machine_emb: [hidden_dim]

        Returns:
            action_emb: [hidden_dim]
        """
        x = torch.cat([op_emb, agv_emb, machine_emb], dim=-1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x

    def forward_batch(self, node_embs: torch.Tensor,
                      indices: torch.Tensor) -> torch.Tensor:
        """Batch action encoding.

        Args:
            node_embs: [N, hidden_dim] all node embeddings
            indices: [K, 3] graph node indices of (op, agv, machine)

        Returns:
            action_embs: [K, hidden_dim]
        """
        op_embs = node_embs[indices[:, 0]]    # [K, hidden_dim]
        agv_embs = node_embs[indices[:, 1]]   # [K, hidden_dim]
        mach_embs = node_embs[indices[:, 2]]  # [K, hidden_dim]
        x = torch.cat([op_embs, agv_embs, mach_embs], dim=-1)  # [K, hidden_dim*3]
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x


# ============================================================
# Transition Network
# ============================================================

class TransitionNet(nn.Module):
    """Predicts next global state encoding given current state and action.

    Deterministic model: s' = T(s, a)
    """

    def __init__(self, state_dim: int = 256, action_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

    def forward(self, global_state: torch.Tensor,
                action_emb: torch.Tensor) -> torch.Tensor:
        """Predict next state.

        Args:
            global_state: [state_dim] or [B, state_dim]
            action_emb: [action_dim] or [B, action_dim]

        Returns:
            predicted_next_state: [state_dim] or [B, state_dim]
        """
        x = torch.cat([global_state, action_emb], dim=-1)
        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================
# Value Network
# ============================================================

class ValueNet(nn.Module):
    """Predicts the value V(s) of a global state encoding."""

    def __init__(self, state_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        self.ln1 = nn.LayerNorm(hidden_dim)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_state: [state_dim] or [B, state_dim]

        Returns:
            value: [1] or [B, 1]
        """
        x = F.leaky_relu(self.ln1(self.fc1(global_state)))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================
# GraphDPAgent
# ============================================================

@register_component("packet_factory.GraphDPAgent")
class GraphDPAgent(BaseAgent):
    """GNN-based Dynamic Programming DRL Agent for FJSP.

    Decision-making follows a DP rollout:
    1. Encode current factory state via GNN -> global_state[256], node_embs[N,64]
    2. For each READY operation, evaluate all (machine, AGV) candidates
       by predicting next state T(s,a) and evaluating V(s')
    3. Pick the best assignment, advance the predicted state
    4. Return all assignments as the decision path

    Training updates the transition model T and value model V,
    then re-walks stored trajectories for policy improvement.
    """

    def __init__(self, name=None, agent_id=None, context=None,
                 ui_mode: str = BACKEND, task_mode: str = TRAINING,
                 model_path: Optional[str] = None,
                 hidden_dim: int = 64,
                 gamma: float = 0.99,
                 lr_gnn: float = 1e-4,
                 lr_trans: float = 1e-3,
                 lr_value: float = 1e-3,
                 alpha: float = 1.0,
                 rewalk_iters: int = 3,
                 buffer_capacity: int = 5000,
                 batch_size: int = 64,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.998,
                 epsilon_min: float = 0.01,
                 allow_agv_reassignment: bool = False,
                 device: Optional[str] = None,
                 **kwargs):
        super().__init__(name, agent_id, context, ui_mode, task_mode, model_path)

        self.hidden_dim = hidden_dim
        self.state_dim = hidden_dim * 4  # global state = per-type pooled + concat
        self.action_dim = hidden_dim
        self.gamma = gamma
        self.alpha = alpha
        self.rewalk_iters = rewalk_iters
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.allow_agv_reassignment = allow_agv_reassignment

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Graph builder
        self.graph_builder = FactoryGraphBuilder(device=str(self.device))

        # Neural networks
        self.gnn_encoder = GNNStateEncoder(
            hidden_dim=hidden_dim, num_layers=2,
        ).to(self.device)
        self.action_encoder = ActionEncoder(hidden_dim).to(self.device)
        self.transition_net = TransitionNet(
            state_dim=self.state_dim, action_dim=self.action_dim,
        ).to(self.device)
        self.value_net = ValueNet(state_dim=self.state_dim).to(self.device)

        # Optimizers (separate lr for GNN vs T/V)
        self.optimizer_gnn = torch.optim.Adam(
            list(self.gnn_encoder.parameters()) + list(self.action_encoder.parameters()),
            lr=lr_gnn,
        )
        self.optimizer_trans = torch.optim.Adam(
            self.transition_net.parameters(), lr=lr_trans,
        )
        self.optimizer_value = torch.optim.Adam(
            self.value_net.parameters(), lr=lr_value,
        )

        # Training buffers
        self.replay_buffer: List[Transition] = []
        self._episode_buffer: List[EpisodeStep] = []

        # State tracking for transition collection
        self._prev_graph_result: Optional[GraphBuildResult] = None
        self._prev_global_state: Optional[torch.Tensor] = None
        self._prev_node_embs: Optional[torch.Tensor] = None

        # Training statistics
        self.training_history: Dict[str, list] = {
            'episodes': [], 'trans_loss': [], 'value_loss': [],
            'total_loss': [], 'makespans': [], 'epsilon': [],
        }
        self._episode_reward = 0.0
        self._step_count = 0

        # Load model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            LOGGER.info(f"[GraphDPAgent] Loaded model from {model_path}")
        else:
            LOGGER.info(f"[GraphDPAgent] Initialized on {self.device}, "
                        f"hidden_dim={hidden_dim}, state_dim={self.state_dim}")

    # ----------------------------------------------------------
    # Core: DP Rollout
    # ----------------------------------------------------------

    def sample(self, agvs: List[AGV], machines: List[Machine],
               jobs: List[Job]) -> Tuple[List[Tuple[Operation, AGV, Machine]], float]:
        """DP-style rollout: plan a path of decisions using T and V.

        For each READY operation, evaluate all (machine, AGV) candidates
        by predicting the next state and its value. Pick the best.
        After each decision, advance the predicted state via T.
        """
        decisions = []

        # 1. Build and encode current state
        factory_graph = self._get_factory_graph(agvs)
        current_time = self._get_current_time()

        graph_result = self.graph_builder.build(
            agvs, machines, jobs, factory_graph, current_time,
        )
        self.gnn_encoder.eval()
        with torch.no_grad():
            global_state, node_embs = self.gnn_encoder(
                graph_result.data, graph_result.node_type_offsets,
            )

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
            self._store_step(graph_result, global_state, node_embs, [], None, 0.0)
            # Check completion
            if all(j.is_finished() for j in jobs):
                self.alive = False
                return [], 0
            return [], DEFAULT_STEP_TIME

        # Sort by urgency: more remaining work -> more urgent
        ready_ops.sort(key=lambda op: -self._remaining_processing_time(op))

        # 3. DP rollout: for each READY op, find best (machine, agv)
        current_state = global_state.detach()
        current_embs = node_embs.detach()
        assigned_agvs = set()

        all_action_indices = []
        all_action_embs = []

        for op in ready_ops:
            capable_machines = [m for m in machines
                                if op.is_machine_capable(m.id) and m.is_available()]
            available_agvs = [a for a in agvs
                              if a.get_status() == AGVStatus.READY
                              and (self.allow_agv_reassignment or a.id not in assigned_agvs)]

            if not capable_machines or not available_agvs:
                continue

            # Build candidate actions
            candidates = []
            for m in capable_machines:
                for a in available_agvs:
                    op_idx = graph_result.id_to_node_idx.get(("operation", op.id))
                    agv_idx = graph_result.id_to_node_idx.get(("agv", a.id))
                    m_idx = graph_result.id_to_node_idx.get(("machine", m.id))
                    if op_idx is not None and agv_idx is not None and m_idx is not None:
                        candidates.append((op_idx, agv_idx, m_idx, op, a, m))

            if not candidates:
                continue

            # Batch evaluation of all candidates
            K = len(candidates)
            candidate_indices = torch.tensor(
                [[c[0], c[1], c[2]] for c in candidates],
                dtype=torch.long, device=self.device,
            )

            with torch.no_grad():
                action_embs = self.action_encoder.forward_batch(
                    current_embs, candidate_indices,
                )  # [K, action_dim]

                state_repeated = current_state.unsqueeze(0).expand(K, -1)  # [K, state_dim]
                next_states = self.transition_net(state_repeated, action_embs)  # [K, state_dim]
                values = self.value_net(next_states).squeeze(-1)  # [K]

            # Select best action (epsilon-greedy during training)
            if self.task_mode == TRAINING and random.random() < self.epsilon:
                best_idx = random.randint(0, K - 1)
            else:
                best_idx = int(values.argmax())

            best = candidates[best_idx]
            decisions.append((best[3], best[4], best[5]))  # (op, agv, machine)
            all_action_indices.append((best[0], best[1], best[2]))
            if all_action_embs is not None or len(all_action_embs) == 0:
                all_action_embs.append(action_embs[best_idx])

            # Advance predicted state for next op's evaluation
            with torch.no_grad():
                current_state = next_states[best_idx].detach()
            assigned_agvs.add(best[4].id)

        # Store step for training
        step_reward = 0.0  # will be computed in reward()
        if all_action_embs:
            action_emb_tensor = torch.stack(all_action_embs)
        else:
            action_emb_tensor = torch.zeros(0, self.action_dim, device=self.device)

        self._store_step(graph_result, global_state, node_embs,
                         all_action_indices, action_emb_tensor, step_reward)

        return decisions, DEFAULT_STEP_TIME

    # ----------------------------------------------------------
    # Reward
    # ----------------------------------------------------------

    def reward(self, *args, **kwargs) -> float:
        """Compute composite reward: makespan penalty + utilization + balance + waiting."""
        reward = 0.0

        if self.context and hasattr(self.context, 'env_timeline'):
            # Makespan penalty
            reward -= 0.005 * self.context.env_timeline

        if self.context and hasattr(self.context, 'jobs'):
            jobs = self.context.jobs
            total = len(jobs) if jobs else 1
            completed = sum(1 for j in jobs if j.is_finished())
            # Completion bonus
            reward += 50.0 * (completed / total)

            # Waiting penalty
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

            # Load balance
            loads = [len(m.input_queue) for m in machines]
            if len(loads) > 1 and np.mean(loads) > 0:
                cv = float(np.std(loads)) / (float(np.mean(loads)) + 1e-5)
                reward -= 3.0 * cv

        if self.context and hasattr(self.context, 'agvs'):
            agvs = self.context.agvs
            total_a = len(agvs) if agvs else 1
            active = sum(1 for a in agvs if a.get_status() != AGVStatus.READY)
            reward += 1.5 * (active / total_a)

        self._episode_reward += reward
        return reward

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def before_sample(self, *args, **kwargs):
        """Collect transition from previous step if available."""
        if self._prev_graph_result is None:
            return
        if self.task_mode != TRAINING:
            return

        # Get current state from args or kwargs
        agvs = kwargs.get('agvs', args[0] if len(args) > 0 else [])
        machines = kwargs.get('machines', args[1] if len(args) > 1 else [])
        jobs = kwargs.get('jobs', args[2] if len(args) > 2 else [])

        if not agvs and not machines and not jobs:
            return

        factory_graph = self._get_factory_graph(agvs)
        current_time = self._get_current_time()
        next_graph_result = self.graph_builder.build(
            agvs, machines, jobs, factory_graph, current_time,
        )

        # Compute reward for the transition
        step_reward = self.reward()

        # Check if done
        done = all(j.is_finished() for j in jobs) if jobs else False

        # Store transition for each action taken in previous step
        prev_step = self._episode_buffer[-1] if self._episode_buffer else None
        if prev_step and prev_step.action_indices:
            for i, action_idx in enumerate(prev_step.action_indices):
                trans = Transition(
                    graph_result_s=self._prev_graph_result,
                    graph_result_s_next=next_graph_result,
                    action_info=action_idx,
                    reward=step_reward / max(len(prev_step.action_indices), 1),
                    done=done,
                )
                self.replay_buffer.append(trans)
                if len(self.replay_buffer) > self.buffer_capacity:
                    self.replay_buffer.pop(0)

        self._prev_graph_result = next_graph_result

    def train(self, *args, **kwargs):
        """Train transition and value networks from replay buffer, then re-walk."""
        if self.task_mode != TRAINING:
            return

        if len(self.replay_buffer) < self.batch_size:
            return

        # Phase 1: Standard training from replay buffer
        trans_loss, value_loss = self._train_from_buffer()

        # Phase 2: Re-walk
        if self.rewalk_iters > 0 and len(self._episode_buffer) > 1:
            self._rewalk()

        # Record training stats
        total_loss = trans_loss + self.alpha * value_loss
        self.training_history['trans_loss'].append(trans_loss)
        self.training_history['value_loss'].append(value_loss)
        self.training_history['total_loss'].append(total_loss)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _train_from_buffer(self) -> Tuple[float, float]:
        """Train from replay buffer. Returns (trans_loss, value_loss)."""
        batch = random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))

        self.gnn_encoder.train()
        self.transition_net.train()
        self.value_net.train()

        total_trans_loss = 0.0
        total_value_loss = 0.0

        for trans in batch:
            # Re-encode s and s'
            s_global, s_node_embs = self.gnn_encoder(
                trans.graph_result_s.data, trans.graph_result_s.node_type_offsets,
            )
            s_next_global, _ = self.gnn_encoder(
                trans.graph_result_s_next.data, trans.graph_result_s_next.node_type_offsets,
            )

            # Action embedding
            op_idx, agv_idx, m_idx = trans.action_info
            op_emb = s_node_embs[op_idx]
            agv_emb = s_node_embs[agv_idx]
            mach_emb = s_node_embs[m_idx]
            action_emb = self.action_encoder(op_emb, agv_emb, mach_emb)

            # Transition loss
            s_pred = self.transition_net(s_global, action_emb)
            trans_loss = F.mse_loss(s_pred, s_next_global.detach())

            # Value loss (TD)
            v_s = self.value_net(s_global)
            v_s_next = self.value_net(s_next_global.detach())
            target = trans.reward + self.gamma * v_s_next * (1.0 - float(trans.done))
            value_loss = F.mse_loss(v_s, target.detach())

            total_trans_loss += trans_loss.item()
            total_value_loss += value_loss.item()

            # Backward
            self.optimizer_gnn.zero_grad()
            self.optimizer_trans.zero_grad()
            self.optimizer_value.zero_grad()

            loss = trans_loss + self.alpha * value_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.gnn_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.transition_net.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)

            self.optimizer_gnn.step()
            self.optimizer_trans.step()
            self.optimizer_value.step()

        n = len(batch)
        return total_trans_loss / n, total_value_loss / n

    def _rewalk(self):
        """Re-walk stored episode: re-encode graphs with updated GNN, retrain."""
        self.gnn_encoder.train()
        self.transition_net.train()
        self.value_net.train()

        for _ in range(self.rewalk_iters):
            # Compute Monte Carlo returns from episode
            steps = self._episode_buffer
            if len(steps) < 2:
                break

            # Compute returns G_t for each step
            returns = []
            G = 0.0
            for step in reversed(steps):
                G = step.reward + self.gamma * G
                returns.insert(0, G)

            for i, step in enumerate(steps):
                if not step.action_indices:
                    continue

                # Re-encode current and next state
                s_global, s_node_embs = self.gnn_encoder(
                    step.graph_result.data, step.graph_result.node_type_offsets,
                )

                if i + 1 < len(steps):
                    next_step = steps[i + 1]
                    s_next_global, _ = self.gnn_encoder(
                        next_step.graph_result.data, next_step.graph_result.node_type_offsets,
                    )
                    done = False
                else:
                    s_next_global = torch.zeros_like(s_global)
                    done = True

                for j, action_idx in enumerate(step.action_indices):
                    op_idx, agv_idx, m_idx = action_idx
                    op_emb = s_node_embs[op_idx]
                    agv_emb = s_node_embs[agv_idx]
                    mach_emb = s_node_embs[m_idx]
                    action_emb = self.action_encoder(op_emb, agv_emb, mach_emb)

                    # Transition loss
                    s_pred = self.transition_net(s_global, action_emb)
                    trans_loss = F.mse_loss(s_pred, s_next_global.detach())

                    # Value loss (Monte Carlo)
                    v_s = self.value_net(s_global)
                    G_t = torch.tensor([returns[i]], dtype=torch.float32, device=self.device)
                    value_loss = F.mse_loss(v_s, G_t)

                    loss = trans_loss + self.alpha * value_loss

                    self.optimizer_gnn.zero_grad()
                    self.optimizer_trans.zero_grad()
                    self.optimizer_value.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.gnn_encoder.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.transition_net.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)

                    self.optimizer_gnn.step()
                    self.optimizer_trans.step()
                    self.optimizer_value.step()

    def after_sample(self, *args, **kwargs):
        """Post-sample hook: store current graph for next transition."""
        if self._episode_buffer:
            self._prev_graph_result = self._episode_buffer[-1].graph_result

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _store_step(self, graph_result, global_state, node_embs,
                    action_indices, action_embs, reward):
        """Store an episode step."""
        step = EpisodeStep(
            graph_result=graph_result,
            global_state=global_state.detach(),
            node_embs=node_embs.detach(),
            action_indices=action_indices,
            action_embs=action_embs.detach() if action_embs is not None else torch.zeros(0),
            reward=reward,
        )
        self._episode_buffer.append(step)
        self._step_count += 1

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

    def _get_valid_actions(self, agvs, machines, jobs):
        """Get all valid (op, agv, machine) actions."""
        valid = []
        for job in jobs:
            if job.is_finished():
                continue
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() != OperationStatus.READY:
                    continue
                capable = [m for m in machines if op.is_machine_capable(m.id) and m.is_available()]
                available = [a for a in agvs if a.get_status() == AGVStatus.READY]
                for a in available:
                    for m in capable:
                        valid.append((op, a, m))
        return valid

    # ----------------------------------------------------------
    # Model save / load
    # ----------------------------------------------------------

    def save_model(self, path: Optional[str] = None) -> bool:
        """Save all network state_dicts and hyperparameters to a .pt file."""
        try:
            if path is None:
                agent_name = self.name or "GraphDPAgent"
                agent_dir = f"training_logs/models/{agent_name}"
                os.makedirs(agent_dir, exist_ok=True)
                path = f"{agent_dir}/agent_model.pt"

            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            checkpoint = {
                'gnn_encoder': self.gnn_encoder.state_dict(),
                'action_encoder': self.action_encoder.state_dict(),
                'transition_net': self.transition_net.state_dict(),
                'value_net': self.value_net.state_dict(),
                'optimizer_gnn': self.optimizer_gnn.state_dict(),
                'optimizer_trans': self.optimizer_trans.state_dict(),
                'optimizer_value': self.optimizer_value.state_dict(),
                'hyperparams': {
                    'hidden_dim': self.hidden_dim,
                    'gamma': self.gamma,
                    'alpha': self.alpha,
                    'epsilon': self.epsilon,
                    'epsilon_decay': self.epsilon_decay,
                    'epsilon_min': self.epsilon_min,
                    'rewalk_iters': self.rewalk_iters,
                    'batch_size': self.batch_size,
                    'buffer_capacity': self.buffer_capacity,
                    'allow_agv_reassignment': self.allow_agv_reassignment,
                },
                'training_history': self.training_history,
                'mode': self.mode,
            }
            torch.save(checkpoint, path)
            LOGGER.info(f"[GraphDPAgent] Model saved to {path}")
            return True
        except Exception as e:
            LOGGER.error(f"[GraphDPAgent] Save failed: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load all network state_dicts and hyperparameters from a .pt file."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder'])
            self.action_encoder.load_state_dict(checkpoint['action_encoder'])
            self.transition_net.load_state_dict(checkpoint['transition_net'])
            self.value_net.load_state_dict(checkpoint['value_net'])

            if 'optimizer_gnn' in checkpoint:
                self.optimizer_gnn.load_state_dict(checkpoint['optimizer_gnn'])
            if 'optimizer_trans' in checkpoint:
                self.optimizer_trans.load_state_dict(checkpoint['optimizer_trans'])
            if 'optimizer_value' in checkpoint:
                self.optimizer_value.load_state_dict(checkpoint['optimizer_value'])

            hp = checkpoint.get('hyperparams', {})
            self.epsilon = hp.get('epsilon', self.epsilon)
            self.gamma = hp.get('gamma', self.gamma)
            self.alpha = hp.get('alpha', self.alpha)

            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']

            LOGGER.info(f"[GraphDPAgent] Model loaded from {path}")
            return True
        except Exception as e:
            LOGGER.error(f"[GraphDPAgent] Load failed: {e}")
            return False

    def new_episode(self):
        """Reset episode-level state. Call at the start of each episode."""
        self._episode_buffer.clear()
        self._episode_reward = 0.0
        self._step_count = 0
        self._prev_graph_result = None
        self._prev_global_state = None
        self._prev_node_embs = None

        # Record episode stats
        if self.training_history['episodes']:
            last_ep = self.training_history['episodes'][-1] + 1
        else:
            last_ep = 1
        self.training_history['episodes'].append(last_ep)
        self.training_history['epsilon'].append(self.epsilon)

        makespan = 0.0
        if self.context and hasattr(self.context, 'env_timeline'):
            makespan = self.context.env_timeline
        self.training_history['makespans'].append(makespan)

    def __repr__(self):
        return (f"<GraphDPAgent id={self.agent_id} name={self.name} "
                f"mode={self.mode} device={self.device}>")
