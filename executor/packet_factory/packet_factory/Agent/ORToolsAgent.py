"""
ORTools Agent for Flexible Job Shop Scheduling.
Uses OR-Tools CP-SAT solver to find globally optimal or near-optimal schedules.
"""

from typing import List, Tuple, Any, Optional

from .BaseAgent import BaseAgent, DEFAULT_STEP_TIME, BACKEND, TRAINING
from executor.packet_factory.packet_factory.Agent.GreedyAgent import GreedyAgent
from executor.packet_factory.registry import register_component
from executor.packet_factory.logger.logger import LOGGER

from executor.packet_factory.packet_factory.optimizer.ORToolsOptimizer import ORToolsOptimizer, ScheduleResult


@register_component("packet_factory.ORToolsAgent")
class ORToolsAgent(BaseAgent):
    """
    Agent that uses OR-Tools CP-SAT solver for FJSP scheduling.

    This agent:
    1. Extracts current state (pending operations, machine/AGV status)
    2. Calls ORToolsOptimizer to solve the full scheduling problem
    3. Returns decisions in the standard (Operation, AGV, Machine) format
    4. Falls back to GreedyAgent if optimization fails or times out
    """

    def __init__(self, name=None, agent_id=None, context=None,
                 ui_mode: str = BACKEND, task_mode: str = TRAINING,
                 model_path: str = None,
                 time_limit_seconds: int = 30,
                 fallback_enabled: bool = True,
                 **kwargs):
        """
        Initialize ORToolsAgent.

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
        self.last_result: Optional[ScheduleResult] = None

        LOGGER.info(f"ORToolsAgent initialized: time_limit={time_limit_seconds}s, "
                   f"fallback={fallback_enabled}")

    def sample(self, agvs, machines, jobs) -> Tuple[List[Tuple[Any, Any, Any]], float]:
        """
        Generate scheduling decisions using OR-Tools optimization.

        Args:
            agvs: List of AGV objects
            machines: List of Machine objects
            jobs: List of Job objects with operations

        Returns:
            Tuple of (decisions, step_time)
            - decisions: List of (Operation, AGV, Machine) tuples
            - step_time: Recommended simulation step time
        """
        # Extract graph from context
        graph = None
        current_time = 0.0

        if self.context is not None:
            if hasattr(self.context, 'getGraph'):
                graph = self.context.getGraph()
            if hasattr(self.context, 'get_env_timeline'):
                current_time = self.context.get_env_timeline()

        # Solve the scheduling problem
        result = self.optimizer.solve(
            jobs=jobs,
            machines=machines,
            agvs=agvs,
            graph=graph,
            current_time=current_time
        )

        self.last_result = result

        if result.success:
            LOGGER.info(f"ORToolsAgent: Solved successfully, {len(result.decisions)} decisions, "
                       f"makespan={result.makespan:.1f}, time={result.solve_time:.2f}s")
            return result.decisions, DEFAULT_STEP_TIME
        elif self.fallback_enabled:
            LOGGER.warning(f"ORToolsAgent: Optimization failed ({result.solver_status}), "
                          f"falling back to GreedyAgent")
            return self.fallback_agent.sample(agvs, machines, jobs)
        else:
            LOGGER.error(f"ORToolsAgent: Optimization failed and fallback disabled, "
                        f"returning empty decisions")
            return [], DEFAULT_STEP_TIME

    def reward(self, *args, **kwargs) -> float:
        """
        Calculate reward (for compatibility with environment).
        In optimization mode, reward is computed externally based on makespan.
        """
        return 0.0

    def train(self, *args, **kwargs):
        """
        Training method (no-op for optimization agent).
        OR-Tools finds optimal solutions analytically, no learning required.
        """
        LOGGER.debug("ORToolsAgent.train() called - no-op (optimization doesn't learn)")

    def get_last_result(self) -> Optional[ScheduleResult]:
        """Get the last optimization result for analysis."""
        return self.last_result

    def set_time_limit(self, seconds: int):
        """Update the solver time limit."""
        self.optimizer = ORToolsOptimizer(time_limit_seconds=seconds)
        LOGGER.info(f"ORToolsAgent: Time limit updated to {seconds}s")