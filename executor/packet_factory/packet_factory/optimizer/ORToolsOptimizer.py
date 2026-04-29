"""
ORTools Optimizer for Flexible Job Shop Scheduling with AGV transportation.
Uses OR-Tools CP-SAT solver for global optimization.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import time as time_module

from ortools.sat.python.cp_model import CpModel, CpSolver, OPTIMAL, FEASIBLE, INFEASIBLE, MODEL_INVALID, UNKNOWN

from executor.packet_factory.logger.logger import LOGGER


@dataclass
class ScheduleResult:
    """Result of the optimization solver."""
    success: bool
    decisions: List[Tuple[Any, Any, Any]]  # (Operation, AGV, Machine)
    makespan: float
    solver_status: str
    solve_time: float


class ORToolsOptimizer:
    """
    OR-Tools based optimizer for Flexible Job Shop Scheduling (FJSP) with AGV transport.

    This optimizer formulates the scheduling problem as a CP-SAT model and solves it
    to find (near) optimal schedules minimizing makespan.
    """

    def __init__(self, time_limit_seconds: int = 30, num_workers: int = 4):
        """
        Initialize the OR-Tools optimizer.

        Args:
            time_limit_seconds: Maximum time to spend solving (per call)
            num_workers: Number of parallel workers for the solver
        """
        self.time_limit_seconds = time_limit_seconds
        self.num_workers = num_workers

    def solve(self, jobs, machines, agvs, graph, current_time: float = 0.0) -> ScheduleResult:
        """
        Solve the FJSP scheduling problem.

        Args:
            jobs: List of Job objects with operations
            machines: List of Machine objects
            agvs: List of AGV objects
            graph: Graph object for distance calculations
            current_time: Current simulation time

        Returns:
            ScheduleResult with decisions and makespan
        """
        start_time = time_module.time()

        from executor.packet_factory.packet_factory.packet_factory_env.Utils.util import OperationStatus

        # Collect all pending operations
        pending_ops = []
        for job in jobs:
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                if op.get_status() == OperationStatus.READY:
                    pending_ops.append((job, i, op))

        if not pending_ops:
            return ScheduleResult(
                success=True,
                decisions=[],
                makespan=current_time,
                solver_status="NO_PENDING_OPERATIONS",
                solve_time=time_module.time() - start_time
            )

        num_ops = len(pending_ops)
        num_machines = len(machines)
        num_agvs = len(agvs)

        LOGGER.info(f"ORToolsOptimizer: Solving FJSP with {num_ops} ops, {num_machines} machines, {num_agvs} AGVs")
        print(f"[ORToolsOptimizer] Solving FJSP: {num_ops} ops, {num_machines} machines, {num_agvs} AGVs")

        # Build the model
        model = CpModel()

        horizon = self._estimate_horizon(jobs, machines, current_time)
        big_m = horizon * 3  # Large enough to effectively disable constraints

        # Variables: start time, end time, and machine assignment for each operation
        start_vars = [model.NewIntVar(0, horizon, f"s_{j}") for j in range(num_ops)]
        end_vars = [model.NewIntVar(0, horizon * 2, f"e_{j}") for j in range(num_ops)]
        machine_vars = []

        for j, (job, op_idx, op) in enumerate(pending_ops):
            capable = [m_idx for m_idx, m in enumerate(machines) if op.is_machine_capable(m.get_id())]
            if not capable:
                LOGGER.warning(f"Operation {op.id} has no capable machine!")
                return ScheduleResult(
                    success=False, decisions=[], makespan=current_time,
                    solver_status="NO_CAPABLE_MACHINE", solve_time=time_module.time() - start_time
                )
            machine_vars.append(model.NewIntVar(min(capable), max(capable), f"m_{j}"))

        # Constraint 1: Job sequencing - operations within same job are sequential
        job_ops: Dict[int, List[int]] = {}
        for j, (job, op_idx, op) in enumerate(pending_ops):
            job_ops.setdefault(job.get_id(), []).append(j)

        for op_list in job_ops.values():
            for k in range(len(op_list) - 1):
                j_curr, j_next = op_list[k], op_list[k + 1]
                model.Add(start_vars[j_next] >= end_vars[j_curr])

        # Constraint 2: Machine capacity - no overlapping on same machine
        # Create boolean variables to track which machine each operation is on
        on_machine_vars = {}  # (j, m_idx) -> bool var

        for j in range(num_ops):
            for m_idx in range(num_machines):
                _, _, op = pending_ops[j]
                if op.is_machine_capable(machines[m_idx].get_id()):
                    on_var = model.NewBoolVar(f"on_{j}_{m_idx}")
                    on_machine_vars[(j, m_idx)] = on_var
                    # Link to machine_vars[j]
                    model.Add(machine_vars[j] == m_idx).OnlyEnforceIf(on_var)
                    model.Add(machine_vars[j] != m_idx).OnlyEnforceIf(on_var.Not())

        # For each machine, enforce no-overlap between pairs of operations
        for m_idx, machine in enumerate(machines):
            ops_on_m = [j for j in range(num_ops) if (j, m_idx) in on_machine_vars]

            for i in range(len(ops_on_m)):
                for k in range(i + 1, len(ops_on_m)):
                    j1, j2 = ops_on_m[i], ops_on_m[k]

                    # Boolean: is j1 before j2?
                    is_before = model.NewBoolVar(f"before_{j1}_{j2}_{m_idx}")

                    # Get durations for this machine
                    _, _, op1 = pending_ops[j1]
                    _, _, op2 = pending_ops[j2]
                    dur1 = int(op1.get_duration(machine.get_id()))
                    dur2 = int(op2.get_duration(machine.get_id()))

                    # Both on machine m_idx -> enforce ordering
                    both_on = model.NewBoolVar(f"both_{j1}_{j2}_{m_idx}")
                    model.AddBoolAnd([on_machine_vars[(j1, m_idx)], on_machine_vars[(j2, m_idx)]]).OnlyEnforceIf(both_on)
                    model.AddBoolOr([on_machine_vars[(j1, m_idx)].Not(), on_machine_vars[(j2, m_idx)].Not()]).OnlyEnforceIf(both_on.Not())

                    # If both on this machine, enforce ordering via is_before
                    model.Add(end_vars[j1] <= start_vars[j2] + big_m * (1 - is_before)).OnlyEnforceIf(both_on)
                    model.Add(end_vars[j2] <= start_vars[j1] + big_m * is_before).OnlyEnforceIf(both_on)

        # Constraint 3: End time = start + duration (based on machine assignment)
        for j, (job, op_idx, op) in enumerate(pending_ops):
            capable = [(m_idx, int(op.get_duration(m.get_id())))
                     for m_idx, m in enumerate(machines) if op.is_machine_capable(m.get_id())]

            dur_var = model.NewIntVar(min(d for _, d in capable), max(d for _, d in capable), f"dur_{j}")
            for m_idx, dur in capable:
                model.Add(dur_var == dur).OnlyEnforceIf(on_machine_vars[(j, m_idx)])

            model.Add(end_vars[j] == start_vars[j] + dur_var)

        # Ensure each operation is on exactly one machine
        for j in range(num_ops):
            on_vars_j = [on_machine_vars[(j, m_idx)] for m_idx in range(num_machines) if (j, m_idx) in on_machine_vars]
            model.AddBoolOr(on_vars_j)

        # Objective: minimize makespan
        makespan = model.NewIntVar(0, horizon * 2, "makespan")
        model.AddMaxEquality(makespan, end_vars)
        model.Minimize(makespan)

                # Solve
        solver = CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_seconds
        solver.parameters.num_workers = self.num_workers

        LOGGER.info(f"ORToolsOptimizer: Starting solve with time limit {self.time_limit_seconds}s")

        status = solver.Solve(model)
        solve_time = time_module.time() - start_time

        solver_status_map = {
            OPTIMAL: "OPTIMAL",
            FEASIBLE: "FEASIBLE",
            INFEASIBLE: "INFEASIBLE",
            MODEL_INVALID: "MODEL_INVALID",
            UNKNOWN: "UNKNOWN",
        }
        solver_status = solver_status_map.get(status, f"UNKNOWN({status})")

        if status == OPTIMAL or status == FEASIBLE:
            decisions = []
            for j, (job, op_idx, op) in enumerate(pending_ops):
                m_idx = solver.Value(machine_vars[j])
                a_idx = j % num_agvs
                machine = machines[m_idx]
                agv = agvs[a_idx]
                decisions.append((op, agv, machine))

            final_makespan = float(solver.Value(makespan))

            LOGGER.info(f"ORTools solved: status={solver_status}, makespan={final_makespan}, "
                       f"ops={num_ops}, time={solve_time:.2f}s")
            print(f"[ORTools] Solved: {solver_status}, makespan={final_makespan}, time={solve_time:.3f}s")

            return ScheduleResult(
                success=True,
                decisions=decisions,
                makespan=final_makespan,
                solver_status=solver_status,
                solve_time=solve_time
            )
        else:
            LOGGER.warning(f"ORTools failed: status={solver_status}, time={solve_time:.2f}s")
            print(f"[ORTools] Failed: {solver_status}, time={solve_time:.3f}s")
            return ScheduleResult(
                success=False, decisions=[], makespan=current_time,
                solver_status=solver_status, solve_time=solve_time
            )

    def _estimate_horizon(self, jobs, machines, current_time: float) -> int:
        """Estimate an upper bound for the scheduling horizon."""
        total_time = float(current_time)

        for job in jobs:
            for i in range(job.get_operation_count()):
                op = job.get_operation(i)
                min_duration = min(
                    op.get_duration(m.get_id())
                    for m in machines
                    if op.is_machine_capable(m.get_id())
                )
                if min_duration < float('inf'):
                    total_time += min_duration

        return max(int(total_time) + 100, 1000)