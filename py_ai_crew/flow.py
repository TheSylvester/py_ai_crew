from typing import List, Dict, Optional, Callable, Any, Union
import asyncio
from datetime import datetime
import json
import hashlib
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent
from pydantic_ai.result import RunResult
from pydantic_ai.dependencies import RunContext
from pydantic_ai.logging import LogConfig

from .models import Task, TaskStatus, TaskOutput
from .executor import TaskExecutor
from .goals import Goal, GoalSet


class ProcessType(str, Enum):
    """Type of process execution."""

    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"


class NextAction(str, Enum):
    """Defines the next action after a task execution."""

    CONTINUE = "continue"  # Continue to next task
    RETRY = "retry"  # Retry current task
    BRANCH = "branch"  # Branch to different task
    END = "end"  # End flow execution


class FlowCondition(BaseModel):
    """Defines a condition for task execution."""

    task_id: str
    condition: Callable[[RunResult], bool]
    next_task_id: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class FlowState(BaseModel):
    """Maintains flow execution state."""

    current_task_id: str
    retry_counts: Dict[str, int] = Field(default_factory=dict)
    max_retries: int = Field(default=3, ge=0)
    visited_tasks: List[str] = Field(default_factory=list)
    execution_start: datetime = Field(default_factory=datetime.utcnow)
    execution_end: Optional[datetime] = None
    pending_tasks: List[Task] = Field(default_factory=list)
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def can_retry(self, task_id: str) -> bool:
        """Check if a task can be retried."""
        return self.retry_counts.get(task_id, 0) < self.max_retries

    def increment_retry(self, task_id: str) -> None:
        """Increment retry count for a task."""
        self.retry_counts[task_id] = self.retry_counts.get(task_id, 0) + 1

    def mark_visited(self, task_id: str) -> None:
        """Mark a task as visited."""
        if task_id not in self.visited_tasks:
            self.visited_tasks.append(task_id)

    def complete_execution(self) -> None:
        """Mark execution as complete."""
        self.execution_end = datetime.utcnow()

    @property
    def execution_duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.execution_end:
            return (self.execution_end - self.execution_start).total_seconds()
        return None


class Flow:
    """Manages sequences of tasks with conditional branching, loops, and goal-based execution."""

    def __init__(
        self,
        tasks: List[Task],
        memory: bool = False,
        cache: bool = True,
        callbacks: Optional[Dict[str, Callable]] = None,
        conditions: Optional[List[FlowCondition]] = None,
        max_retries: int = 3,
        goals: Optional[GoalSet] = None,
        process_type: ProcessType = ProcessType.SEQUENTIAL,
        manager_agent: Optional[Agent] = None,
        log_config: Optional[LogConfig] = None,
    ):
        # Validate and store tasks with IDs
        if not tasks:
            raise ValueError("At least one task is required")
        self.tasks = {str(i): task for i, task in enumerate(tasks)}

        # Initialize configuration
        self.memory = memory
        self.cache = cache
        self.callbacks = callbacks or {}
        self.conditions = conditions or []
        self.max_retries = max_retries
        self.goals = goals or GoalSet()
        self.process_type = process_type
        self.manager_agent = manager_agent
        self.log_config = log_config or LogConfig()

        # Initialize execution components
        self.executor = TaskExecutor()
        self.context = RunContext() if memory else None
        self._cache_store: Dict[str, RunResult] = {}

        # Validate manager for hierarchical process
        if self.process_type == ProcessType.HIERARCHICAL and not self.manager_agent:
            self.manager_agent = self._create_default_manager()

    def _create_default_manager(self) -> Agent:
        """Create a default manager agent for hierarchical process."""
        return Agent(
            model="gpt-4-1106-preview",
            system_prompt=(
                "You are a project manager coordinating a team of AI agents. "
                "Your role is to oversee task execution, delegate tasks effectively, "
                "and ensure high-quality results through proper validation. "
                "You should coordinate the team's efforts, provide clear direction, "
                "and maintain high standards for all deliverables."
            ),
        )

    async def _execute_hierarchical(self, state: FlowState) -> List[RunResult]:
        """Execute tasks in hierarchical mode with manager oversight."""
        results = []

        while state.current_task_id is not None:
            task = self.tasks[state.current_task_id]

            # Get manager's assessment and delegation
            assessment = await self.manager_agent.execute(
                f"Assess and delegate task: {task.description}\n"
                f"Current context: {self.context.dict() if self.context else 'None'}"
            )

            # Execute task with manager oversight
            try:
                if task.async_execution:
                    # Handle async execution
                    future = asyncio.create_task(
                        self._execute_task_with_manager(task, state)
                    )
                    state.pending_tasks.append((task, future))
                else:
                    # Handle sync execution
                    result = await self._execute_task_with_manager(task, state)
                    results.append(result)

                    # Process any completed async tasks
                    completed = []
                    for pending_task, future in state.pending_tasks:
                        if future.done():
                            results.append(future.result())
                            completed.append((pending_task, future))

                    # Remove completed tasks
                    for completed_task in completed:
                        state.pending_tasks.remove(completed_task)

                # Determine next action
                action, next_task_id = await self._get_manager_next_action(
                    state.current_task_id, results[-1] if results else None, state
                )

                # Handle action
                if action == NextAction.RETRY:
                    state.increment_retry(state.current_task_id)
                    task.status = TaskStatus.PENDING
                elif action == NextAction.BRANCH:
                    state.mark_visited(state.current_task_id)
                    state.current_task_id = next_task_id
                elif action == NextAction.CONTINUE:
                    state.mark_visited(state.current_task_id)
                    state.current_task_id = next_task_id
                else:  # END
                    state.mark_visited(state.current_task_id)
                    state.current_task_id = None

            except Exception as e:
                task.status = TaskStatus.FAILED
                raise

        # Wait for any remaining async tasks
        if state.pending_tasks:
            pending_futures = [future for _, future in state.pending_tasks]
            additional_results = await asyncio.gather(
                *pending_futures, return_exceptions=True
            )
            results.extend(
                r for r in additional_results if not isinstance(r, Exception)
            )

        return results

    async def _execute_task_with_manager(
        self, task: Task, state: FlowState
    ) -> RunResult:
        """Execute a task with manager oversight."""
        # Get manager's guidance
        guidance = await self.manager_agent.execute(
            f"Provide guidance for task: {task.description}\n"
            f"Current context: {self.context.dict() if self.context else 'None'}"
        )

        # Execute task
        result = await task.execute(self.context.dict() if self.context else None)

        # Get manager's validation
        validation = await self.manager_agent.execute(
            f"Validate result for task: {task.description}\n"
            f"Result: {result.data}\n"
            "Provide feedback and determine if the result meets quality standards."
        )

        # Update context if using memory
        if self.memory and self.context is not None:
            self.context.update({task.description: result.data})

        return result

    async def _get_manager_next_action(
        self, task_id: str, result: Optional[RunResult], state: FlowState
    ) -> tuple[NextAction, Optional[str]]:
        """Get manager's decision on next action."""
        if not result:
            return NextAction.CONTINUE, self._get_next_task_id(task_id)

        decision = await self.manager_agent.execute(
            f"Determine next action for completed task: {self.tasks[task_id].description}\n"
            f"Result: {result.data}\n"
            f"Options:\n"
            f"1. Continue to next task\n"
            f"2. Retry current task\n"
            f"3. Branch to a different task\n"
            f"4. End execution\n"
            f"Consider task dependencies, quality requirements, and overall flow goals."
        )

        # Parse manager's decision
        if "retry" in decision.data.lower():
            if state.can_retry(task_id):
                return NextAction.RETRY, task_id
        elif "branch" in decision.data.lower():
            # Extract target task ID from decision
            for condition in self.conditions:
                if condition.task_id == task_id and condition.condition(result):
                    return NextAction.BRANCH, condition.next_task_id
        elif "end" in decision.data.lower():
            return NextAction.END, None

        # Default to continue
        return NextAction.CONTINUE, self._get_next_task_id(task_id)

    async def kickoff(
        self,
        start_task_id: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> List[RunResult]:
        """Start executing the flow with support for branching, loops, and goals."""
        # Initialize state
        state = FlowState(
            current_task_id=start_task_id or "0", max_retries=self.max_retries
        )

        # Reset goals at the start of execution
        self.goals.reset_all()

        # Interpolate task inputs if provided
        if inputs:
            for task in self.tasks.values():
                task.interpolate_inputs(inputs)

        try:
            # Execute based on process type
            if self.process_type == ProcessType.HIERARCHICAL:
                results = await self._execute_hierarchical(state)
            else:
                results = await self._execute_sequential(state)

            state.complete_execution()
            return results

        except Exception as e:
            state.complete_execution()
            raise

    async def replay_from_task(
        self, task_id: str, inputs: Optional[Dict[str, Any]] = None
    ) -> List[RunResult]:
        """Replay flow execution from a specific task."""
        return await self.kickoff(start_task_id=task_id, inputs=inputs)

    def save_execution_log(
        self, task: Task, result: RunResult, state: FlowState
    ) -> None:
        """Save task execution details for replay."""
        log_entry = {
            "task_id": task.description,
            "timestamp": datetime.utcnow(),
            "context": self.context.dict() if self.context else None,
            "result": result.dict(),
            "status": task.status,
            "output": task.output.dict() if task.output else None,
        }
        state.execution_log.append(log_entry)

    async def _execute_task(
        self, task: Task, context: Optional[Dict[str, Any]] = None
    ) -> RunResult:
        """Execute a task with proper logging and context."""
        if self._is_cached(task):
            return self._get_cached_result(task)

        # Create run context with dependencies
        run_context = RunContext() if not self.context else self.context

        # Execute task
        result = await task.execute(context, run_context)

        # Cache result if enabled
        if self.cache:
            self._cache_result(task, result)

        return result

    async def _execute_sequential(self, state: FlowState) -> List[RunResult]:
        """Execute tasks sequentially."""
        results = []

        while state.current_task_id is not None:
            task = self.tasks[state.current_task_id]
            task.status = TaskStatus.RUNNING

            try:
                # Execute task
                if self._is_cached(task):
                    result = self._get_cached_result(task)
                else:
                    result = await task.execute(
                        self.context.dict() if self.context else None
                    )
                    self._cache_result(task, result)

                # Update task status
                task.status = TaskStatus.COMPLETED

                # Handle post-execution operations
                await self._handle_callbacks(task, result)
                self._update_context(task, result)

                # Determine next action based on goals and conditions
                action, next_task_id = self._get_next_action(
                    state.current_task_id, result, state
                )

                # Handle action
                if action == NextAction.RETRY:
                    state.increment_retry(state.current_task_id)
                    task.status = TaskStatus.PENDING  # Reset for retry
                elif action == NextAction.BRANCH:
                    results.append(result)
                    state.mark_visited(state.current_task_id)
                    state.current_task_id = next_task_id
                elif action == NextAction.CONTINUE:
                    results.append(result)
                    state.mark_visited(state.current_task_id)
                    state.current_task_id = next_task_id
                else:  # END
                    results.append(result)
                    state.mark_visited(state.current_task_id)
                    state.current_task_id = None

            except Exception as e:
                task.status = TaskStatus.FAILED
                raise

        return results

    def _generate_cache_key(self, task: Task) -> str:
        """Generate a unique cache key for a task."""
        task_data = {
            "description": task.description,
            "context": self.context.dict() if self.context else None,
        }
        task_json = json.dumps(task_data, sort_keys=True)
        return hashlib.sha256(task_json.encode()).hexdigest()

    def _is_cached(self, task: Task) -> bool:
        """Check if task result is cached."""
        if not self.cache:
            return False
        cache_key = self._generate_cache_key(task)
        return cache_key in self._cache_store

    def _get_cached_result(self, task: Task) -> RunResult:
        """Retrieve cached task result."""
        cache_key = self._generate_cache_key(task)
        return self._cache_store[cache_key]

    def _cache_result(self, task: Task, result: RunResult) -> None:
        """Cache task result."""
        if self.cache:
            cache_key = self._generate_cache_key(task)
            self._cache_store[cache_key] = result

    async def _handle_callbacks(self, task: Task, result: RunResult) -> None:
        """Execute callbacks after task completion."""
        for callback_name, callback in self.callbacks.items():
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task, result)
                else:
                    callback(task, result)
            except Exception as e:
                # Log error but don't fail execution
                print(f"Error in callback {callback_name}: {e}")

    def _update_context(self, task: Task, result: RunResult) -> None:
        """Update context with task results."""
        if self.memory and self.context is not None:
            # Update context with task result
            self.context[task.description] = result.data
            # Update context with any additional metadata
            if isinstance(result.data, dict):
                self.context.update(result.data)

    def _get_next_action(
        self, task_id: str, result: RunResult, state: FlowState
    ) -> tuple[NextAction, Optional[str]]:
        """Determine the next action based on conditions, goals, and state."""
        # Check if goals are met
        if not self.goals.check_goals(result, self.context):
            # If we have failed goals, end the flow
            if self.goals.get_failed_goals():
                return NextAction.END, None
            # If we have pending goals and can retry, do so
            if state.can_retry(task_id):
                return NextAction.RETRY, task_id

        # Check conditions for branching
        for condition in self.conditions:
            if condition.task_id == task_id and condition.condition(result):
                return NextAction.BRANCH, condition.next_task_id

        # Check if we need to retry
        if isinstance(result.data, str) and "retry" in result.data.lower():
            if state.can_retry(task_id):
                return NextAction.RETRY, task_id
            return NextAction.CONTINUE, self._get_next_task_id(task_id)

        # Continue to next task
        next_task_id = self._get_next_task_id(task_id)
        return (
            (NextAction.END, None)
            if next_task_id is None
            else (NextAction.CONTINUE, next_task_id)
        )

    def _get_next_task_id(self, current_task_id: str) -> Optional[str]:
        """Get the next task ID in sequence."""
        try:
            current_idx = int(current_task_id)
            next_idx = str(current_idx + 1)
            return next_idx if next_idx in self.tasks else None
        except ValueError:
            return None
