import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, List
from datetime import datetime

from pydantic_ai.result import RunResult
from .models import Task, TaskRun, TaskStatus


class TaskExecutor:
    """Handles task execution with built-in evaluation capabilities."""

    def __init__(self):
        self.context: Dict[str, Any] = {}

    @asynccontextmanager
    async def track_run(self, task: Task) -> TaskRun:
        """Context manager to track task execution metrics."""
        run = task.create_run()
        task.status = TaskStatus.RUNNING
        try:
            yield run
            run.metrics["duration"] = (
                datetime.utcnow() - run.start_time
            ).total_seconds()
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            run.error = str(e)
            task.status = TaskStatus.FAILED
            raise
        finally:
            run.end_time = datetime.utcnow()

    async def execute(self, task: Task, context: Optional[Dict] = None) -> RunResult:
        """Execute a task and track metrics."""
        if context:
            self.context.update(context)

        async with self.track_run(task) as run:
            try:
                # Verify result type matches agent's expected type
                if task.result_type != task.agent._result_schema.result_type:
                    raise ValueError(
                        f"Task result type {task.result_type} does not match "
                        f"agent's result type {task.agent._result_schema.result_type}"
                    )

                # Execute task with pydantic-ai
                result = await task.agent.run(task.description, deps=self.context)

                # Store result in run
                run.result = result

                # Update metrics
                run.metrics.update(self._calculate_metrics(result))
                return result
            except Exception as e:
                run.error = str(e)
                task.status = TaskStatus.FAILED
                raise

    def _calculate_metrics(self, result: RunResult) -> Dict[str, float]:
        """Calculate standard metrics for task evaluation."""
        metrics = {
            "message_count": len(result.messages),
            "cost": float(result.cost.total_cost) if result.cost else 0.0,
        }

        # Add token count if result is string
        if isinstance(result.data, str):
            metrics["token_count"] = len(result.data.split())

        # Add tool usage metrics
        tool_calls = sum(
            1 for msg in result.messages if msg.role == "assistant" and msg.tool_calls
        )
        metrics["tool_calls"] = tool_calls

        return metrics
