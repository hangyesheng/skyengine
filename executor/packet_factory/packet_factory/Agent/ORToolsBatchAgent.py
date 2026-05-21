"""
ORTools Batch Agent for Flexible Job Shop Scheduling.
Uses OR-Tools CP-SAT solver to generate complete scheduling plans.

Optimized for 'optimization' mode: the agent outputs ALL operations' scheduling
arrangements at once (both READY and WAITING), sorted by scheduled start time
per AGV. The environment's optimization loop then progressively executes these
decisions without re-invoking the agent, only re-planning when a disruptive
event occurs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import os
import json

from .BaseAgent import BaseAgent, DEFAULT_STEP_TIME, BACKEND, TRAINING
from executor.packet_factory.packet_factory.Agent.GreedyAgent import GreedyAgent
from executor.packet_factory.registry import register_component
from executor.packet_factory.logger.logger import LOGGER

from executor.packet_factory.packet_factory.optimizer.ORToolsOptimizer import ORToolsOptimizer


@dataclass
class ScheduleResult:
    """Result of the optimization solver."""
    success: bool
    decisions: List[Tuple[Any, Any, Any]]  # (Operation, AGV, Machine)
    makespan: float
    solver_status: str
    solve_time: float


@dataclass
class BatchScheduleResult:
    """Complete batch scheduling result with all operations."""
    success: bool
    schedule: List[Tuple[Any, Any, Any, float]]  # (Operation, AGV, Machine, scheduled_start_time)
    makespan: float
    solver_status: str
    solve_time: float
    immediate_decisions: List[Tuple[Any, Any, Any]]  # Only ready operations
    future_decisions: List[Tuple[Any, Any, Any, float]]  # Waiting operations with start times


@register_component("packet_factory.ORToolsBatchAgent")
class ORToolsBatchAgent(BaseAgent):
    """
    Agent that generates complete batch schedules using OR-Tools CP-SAT solver.

    In optimization mode, sample() returns ALL operations (READY + WAITING)
    sorted by scheduled start time per AGV. The environment progressively
    executes these decisions; re-planning only occurs on disruptive events.
    """

    def __init__(self, name=None, agent_id=None, context=None,
                 ui_mode: str = BACKEND, task_mode: str = TRAINING,
                 model_path: str = None,
                 time_limit_seconds: int = 30,
                 fallback_enabled: bool = True,
                 **kwargs):
        """
        Initialize ORToolsBatchAgent.

        Args:
            name: Agent name
            agent_id: Unique agent identifier
            context: Reference to environment (PacketFactoryEnv)
            ui_mode: frontend | backend (visualization control)
            task_mode: training | inference (learning vs. model usage)
            model_path: Model file path (not used for OR-Tools)
            time_limit_seconds: Maximum solving time per decision
            fallback_enabled: Whether to fall back to greedy on failure
            **kwargs: Additional arguments
        """
        super().__init__(name, agent_id, context, ui_mode, task_mode, model_path)

        self.optimizer = ORToolsOptimizer(time_limit_seconds=time_limit_seconds)
        self.fallback_enabled = fallback_enabled
        self.fallback_agent = GreedyAgent(name=f"{name}_Fallback" if name else "GreedyFallback")
        self.last_result: Optional[BatchScheduleResult] = None

        # 结果跟踪
        self.solve_history: List[Dict] = []
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.result_dir = None

        LOGGER.info(f"ORToolsBatchAgent initialized: time_limit={time_limit_seconds}s, "
                   f"fallback={fallback_enabled}")

    def _get_result_dir(self) -> str:
        """Get or create the result directory path."""
        if self.result_dir is None:
            agent_name = getattr(self, 'name', 'ORToolsBatchAgent')
            self.result_dir = f"training_logs/results/{agent_name}_{self.run_timestamp}"
            os.makedirs(self.result_dir, exist_ok=True)
            LOGGER.info(f"Results will be saved to: {self.result_dir}")
        return self.result_dir

    def _serialize_decision(self, decision: Tuple[Any, Any, Any]) -> Dict:
        """Serialize a single decision to dict."""
        op, agv, machine = decision
        return {
            "operation_id": op.id if hasattr(op, 'id') else None,
            "job_id": op.job_id if hasattr(op, 'job_id') else None,
            "machine_id": machine.id if hasattr(machine, 'id') else None,
            "agv_id": agv.id if hasattr(agv, 'id') else None,
            "status": str(op.status) if hasattr(op, 'status') else None,
        }

    def _solve_batch(self, jobs, machines, agvs, graph, current_time: float = 0.0) -> BatchScheduleResult:
        """
        Solve the complete FJSP problem for all operations.

        This includes:
        - All READY operations
        - All WAITING operations (whose predecessors are not yet complete)
        """
        start_time = datetime.now()

        from executor.packet_factory.packet_factory.packet_factory_env.Utils.util import OperationStatus

        # Collect ALL operations (not just READY)
        all_ops = []
        for job in jobs:
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                all_ops.append((job, i, op))

        # Filter out FINISHED operations
        active_ops = [(job, i, op) for job, i, op in all_ops
                      if op.get_status() != OperationStatus.FINISHED]

        if not active_ops:
            return BatchScheduleResult(
                success=True,
                schedule=[],
                makespan=current_time,
                solver_status="NO_ACTIVE_OPERATIONS",
                solve_time=0.0,
                immediate_decisions=[],
                future_decisions=[]
            )

        num_ops = len(active_ops)
        num_machines = len(machines)
        num_agvs = len(agvs)

        LOGGER.info(f"ORToolsBatchAgent: Solving batch FJSP with {num_ops} ops, {num_machines} machines, {num_agvs} AGVs")
        print(f"[ORToolsBatchAgent] Solving: {num_ops} ops, {num_machines} machines, {num_agvs} AGVs")

        # Build the CP-SAT model
        from ortools.sat.python.cp_model import CpModel, CpSolver, OPTIMAL, FEASIBLE, INFEASIBLE, MODEL_INVALID, UNKNOWN

        model = CpModel()

        # Estimate horizon
        horizon = self._estimate_horizon(active_ops, machines, current_time)
        big_m = horizon * 3

        # Variables
        start_vars = [model.NewIntVar(0, horizon, f"s_{j}") for j in range(num_ops)]
        end_vars = [model.NewIntVar(0, horizon * 2, f"e_{j}") for j in range(num_ops)]
        machine_vars = []

        for j, (job, op_idx, op) in enumerate(active_ops):
            capable = [m_idx for m_idx, m in enumerate(machines) if op.is_machine_capable(m.get_id())]
            if not capable:
                LOGGER.warning(f"Operation {op.id} has no capable machine!")
                return BatchScheduleResult(
                    success=False, schedule=[], makespan=current_time,
                    solver_status="NO_CAPABLE_MACHINE", solve_time=0.0,
                    immediate_decisions=[], future_decisions=[]
                )
            machine_vars.append(model.NewIntVar(min(capable), max(capable), f"m_{j}"))

        # Constraint 1: Job sequencing (all operations, not just ready)
        job_ops: Dict[int, List[int]] = {}
        for j, (job, op_idx, op) in enumerate(active_ops):
            job_ops.setdefault(job.get_id(), []).append(j)

        for op_list in job_ops.values():
            for k in range(len(op_list) - 1):
                j_curr, j_next = op_list[k], op_list[k + 1]
                model.Add(start_vars[j_next] >= end_vars[j_curr])

        # Constraint 2: Machine capacity
        on_machine_vars = {}
        for j in range(num_ops):
            for m_idx in range(num_machines):
                _, _, op = active_ops[j]
                if op.is_machine_capable(machines[m_idx].get_id()):
                    on_var = model.NewBoolVar(f"on_{j}_{m_idx}")
                    on_machine_vars[(j, m_idx)] = on_var
                    model.Add(machine_vars[j] == m_idx).OnlyEnforceIf(on_var)
                    model.Add(machine_vars[j] != m_idx).OnlyEnforceIf(on_var.Not())

        for m_idx, machine in enumerate(machines):
            ops_on_m = [j for j in range(num_ops) if (j, m_idx) in on_machine_vars]
            for i in range(len(ops_on_m)):
                for k in range(i + 1, len(ops_on_m)):
                    j1, j2 = ops_on_m[i], ops_on_m[k]
                    is_before = model.NewBoolVar(f"before_{j1}_{j2}_{m_idx}")

                    _, _, op1 = active_ops[j1]
                    _, _, op2 = active_ops[j2]
                    dur1 = int(op1.get_duration(machine.get_id()))
                    dur2 = int(op2.get_duration(machine.get_id()))

                    both_on = model.NewBoolVar(f"both_{j1}_{j2}_{m_idx}")
                    model.AddBoolAnd([on_machine_vars[(j1, m_idx)], on_machine_vars[(j2, m_idx)]]).OnlyEnforceIf(both_on)
                    model.AddBoolOr([on_machine_vars[(j1, m_idx)].Not(), on_machine_vars[(j2, m_idx)].Not()]).OnlyEnforceIf(both_on.Not())

                    model.Add(end_vars[j1] <= start_vars[j2] + big_m * (1 - is_before)).OnlyEnforceIf(both_on)
                    model.Add(end_vars[j2] <= start_vars[j1] + big_m * is_before).OnlyEnforceIf(both_on)

        # Constraint 3: End time = start + duration
        for j, (job, op_idx, op) in enumerate(active_ops):
            capable = [(m_idx, int(op.get_duration(m.get_id())))
                      for m_idx, m in enumerate(machines) if op.is_machine_capable(m.get_id())]
            dur_var = model.NewIntVar(min(d for _, d in capable), max(d for _, d in capable), f"dur_{j}")
            for m_idx, dur in capable:
                model.Add(dur_var == dur).OnlyEnforceIf(on_machine_vars[(j, m_idx)])
            model.Add(end_vars[j] == start_vars[j] + dur_var)

        # Each operation on exactly one machine
        for j in range(num_ops):
            on_vars_j = [on_machine_vars[(j, m_idx)] for m_idx in range(num_machines) if (j, m_idx) in on_machine_vars]
            model.AddBoolOr(on_vars_j)

        # Objective: minimize makespan
        makespan = model.NewIntVar(0, horizon * 2, "makespan")
        model.AddMaxEquality(makespan, end_vars)
        model.Minimize(makespan)

        # Solve
        solver = CpSolver()
        solver.parameters.max_time_in_seconds = self.optimizer.time_limit_seconds
        solver.parameters.num_workers = self.optimizer.num_workers

        status = solver.Solve(model)
        solve_time = (datetime.now() - start_time).total_seconds()

        solver_status_map = {
            OPTIMAL: "OPTIMAL",
            FEASIBLE: "FEASIBLE",
            INFEASIBLE: "INFEASIBLE",
            MODEL_INVALID: "MODEL_INVALID",
            UNKNOWN: "UNKNOWN",
        }
        solver_status = solver_status_map.get(status, f"UNKNOWN({status})")

        if status == OPTIMAL or status == FEASIBLE:
            # Build complete schedule with start times
            schedule = []
            immediate = []
            future = []

            for j, (job, op_idx, op) in enumerate(active_ops):
                m_idx = solver.Value(machine_vars[j])
                start_time_val = float(solver.Value(start_vars[j]))
                a_idx = j % num_agvs

                machine = machines[m_idx]
                agv = agvs[a_idx]

                decision = (op, agv, machine)
                schedule.append((op, agv, machine, start_time_val))

                # Categorize as immediate or future based on current status
                if op.get_status() == OperationStatus.READY:
                    immediate.append(decision)
                else:
                    future.append((op, agv, machine, start_time_val))

            final_makespan = float(solver.Value(makespan))

            LOGGER.info(f"ORToolsBatchAgent solved: status={solver_status}, makespan={final_makespan}, "
                       f"ops={num_ops}, time={solve_time:.2f}s")
            print(f"[ORToolsBatchAgent] Solved: {solver_status}, makespan={final_makespan}, "
                  f"immediate={len(immediate)}, future={len(future)}")

            return BatchScheduleResult(
                success=True,
                schedule=schedule,
                makespan=final_makespan,
                solver_status=solver_status,
                solve_time=solve_time,
                immediate_decisions=immediate,
                future_decisions=future
            )
        else:
            LOGGER.warning(f"ORToolsBatchAgent failed: status={solver_status}, time={solve_time:.2f}s")
            print(f"[ORToolsBatchAgent] Failed: {solver_status}")
            return BatchScheduleResult(
                success=False, schedule=[], makespan=current_time,
                solver_status=solver_status, solve_time=solve_time,
                immediate_decisions=[], future_decisions=[]
            )

    def _estimate_horizon(self, active_ops, machines, current_time: float) -> int:
        """Estimate an upper bound for the scheduling horizon."""
        total_time = float(current_time)
        for _, _, op in active_ops:
            min_duration = min(
                op.get_duration(m.get_id())
                for m in machines
                if op.is_machine_capable(m.get_id())
            )
            if min_duration < float('inf'):
                total_time += min_duration
        return max(int(total_time) + 100, 1000)

    def sample(self, agvs, machines, jobs) -> Tuple[List[Tuple[Any, Any, Any]], float]:
        """
        Generate scheduling decisions using batch optimization.

        In optimization mode, returns ALL operations' scheduling arrangements
        at once (both READY and WAITING), sorted by scheduled start time per AGV.
        The environment progressively executes these decisions without re-invoking
        the agent, only re-planning when a disruptive event occurs.

        Args:
            agvs: List of AGV objects
            machines: List of Machine objects
            jobs: List of Job objects with operations

        Returns:
            Tuple of (decisions, step_time)
            - decisions: List of (Operation, AGV, Machine) tuples for ALL operations,
              sorted by scheduled start time within each AGV
            - step_time: Recommended simulation step time
        """
        # Extract graph and current time from context
        graph = None
        current_time = 0.0

        if self.context is not None:
            if hasattr(self.context, 'getGraph'):
                graph = self.context.getGraph()
            if hasattr(self.context, 'get_env_timeline'):
                current_time = self.context.get_env_timeline()

        # Solve the complete batch scheduling problem
        result = self._solve_batch(jobs, machines, agvs, graph, current_time)
        self.last_result = result

        # Save solve history
        step_idx = len(self.solve_history)
        serialized = {
            "step": step_idx,
            "timestamp": datetime.now().isoformat(),
            "current_time": current_time,
            "success": result.success,
            "solver_status": result.solver_status,
            "makespan": result.makespan,
            "solve_time": result.solve_time,
            "immediate_count": len(result.immediate_decisions),
            "future_count": len(result.future_decisions),
            "total_count": len(result.schedule),
        }
        self.solve_history.append(serialized)
        self._save_step(step_idx, serialized)

        if result.success:
            all_decisions = self._build_execution_order(result)

            LOGGER.info(f"ORToolsBatchAgent: {len(result.immediate_decisions)} immediate, "
                       f"{len(result.future_decisions)} future, "
                       f"{len(all_decisions)} total dispatched")
            return all_decisions, DEFAULT_STEP_TIME
        elif self.fallback_enabled:
            LOGGER.warning(f"ORToolsBatchAgent: Batch solve failed, falling back to GreedyAgent")
            return self.fallback_agent.sample(agvs, machines, jobs)
        else:
            LOGGER.error(f"ORToolsBatchAgent: Solve failed and fallback disabled")
            return [], DEFAULT_STEP_TIME

    def _build_execution_order(self, result: BatchScheduleResult) -> List[Tuple[Any, Any, Any]]:
        """
        Build the execution order from the batch schedule result.

        Sorts operations by scheduled start time within each AGV,
        ensuring the AGV processes them in the optimal sequence.
        Skips operations already in progress (MOVING/WORKING/FINISHED).

        Args:
            result: Batch scheduling result from the solver

        Returns:
            List of (Operation, AGV, Machine) tuples sorted by execution order
        """
        from executor.packet_factory.packet_factory.packet_factory_env.Utils.util import OperationStatus

        # Group by AGV and sort by start time
        agv_schedules: Dict[Any, List[Tuple[float, Any, Any, Any]]] = {}

        for op, agv, machine, start_time in result.schedule:
            status = op.get_status()
            # Skip operations already being processed or completed
            if status in (OperationStatus.MOVING, OperationStatus.WORKING, OperationStatus.FINISHED):
                continue
            agv_schedules.setdefault(agv, []).append((start_time, op, agv, machine))

        # Sort each AGV's operations by start time and flatten
        all_decisions = []
        for agv, ops in agv_schedules.items():
            ops.sort(key=lambda x: x[0])
            for _, op, agv, machine in ops:
                all_decisions.append((op, agv, machine))

        return all_decisions

    def _save_step(self, step_idx: int, data: Dict):
        """Save a single solve step to JSON file."""
        result_dir = self._get_result_dir()
        step_file = os.path.join(result_dir, f"batch_step_{step_idx:04d}.json")
        with open(step_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_future_decisions(self) -> List[Tuple[Any, Any, Any, float]]:
        """Get the cached future decisions (operations scheduled but not yet ready)."""
        if self.last_result:
            return self.last_result.future_decisions
        return []

    def get_complete_schedule(self) -> List[Tuple[Any, Any, Any, float]]:
        """Get the complete schedule from last solve."""
        if self.last_result:
            return self.last_result.schedule
        return []

    def save_training_result(self) -> str:
        """Save training/batch solve results to training_logs/results/."""
        result_dir = self._get_result_dir()

        total_solve_time = sum(s['solve_time'] for s in self.solve_history)
        successful_solves = sum(1 for s in self.solve_history if s['success'])

        final_makespan = 0.0
        for s in reversed(self.solve_history):
            if s['success']:
                final_makespan = s['makespan']
                break

        report = {
            'makespan': final_makespan,
            'total_solve_time': total_solve_time,
            'decision_stats': {
                'total_decision_time': total_solve_time,
                'decision_count': len(self.solve_history),
                'average_decision_time': total_solve_time / len(self.solve_history) if self.solve_history else 0,
            },
            'batch_solve_summary': {
                'total_solves': len(self.solve_history),
                'successful_solves': successful_solves,
            },
            'metadata': {
                'agent_name': getattr(self, 'name', 'ORToolsBatchAgent'),
                'agent_id': getattr(self, 'agent_id', None),
                'run_timestamp': self.run_timestamp,
                'save_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'time_limit_seconds': self.optimizer.time_limit_seconds,
            }
        }

        report_file = os.path.join(result_dir, 'training_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        history_file = os.path.join(result_dir, 'batch_history.jsonl')
        with open(history_file, 'w', encoding='utf-8') as f:
            for entry in self.solve_history:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        LOGGER.info(f"[ORToolsBatchAgent] Results saved to {result_dir}")
        return result_dir

    def reward(self, *args, **kwargs) -> float:
        """Calculate reward (for compatibility)."""
        return 0.0

    def train(self, *args, **kwargs):
        """Training method (no-op for optimization agent)."""
        LOGGER.debug("ORToolsBatchAgent.train() called - no-op")

    def get_last_result(self) -> Optional[BatchScheduleResult]:
        """Get the last optimization result."""
        return self.last_result

    def set_time_limit(self, seconds: int):
        """Update the solver time limit."""
        self.optimizer = ORToolsOptimizer(time_limit_seconds=seconds)
        LOGGER.info(f"ORToolsBatchAgent: Time limit updated to {seconds}s")