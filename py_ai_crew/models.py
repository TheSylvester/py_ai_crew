from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent
from pydantic_ai.result import RunResult
from pydantic_ai.dependencies import RunContext


class TaskStatus(str, Enum):
    """Status of a task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskOutput(BaseModel):
    """Structured output from a task execution."""

    description: str = Field(
        description="Description of the task that generated this output"
    )
    raw: Any = Field(description="Raw output from the task execution")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """A task to be executed by an agent."""

    description: str = Field(description="Description of what needs to be done")
    expected_output: Optional[str] = Field(
        default=None, description="Expected format or structure of the output"
    )
    agent: Optional[Agent] = Field(
        default=None, description="Agent assigned to execute this task"
    )
    output_file: Optional[str] = Field(
        default=None, description="File to save task output to"
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING, description="Current status of the task"
    )
    context: Optional[List[Task]] = Field(
        default=None, description="Tasks whose outputs are required for this task"
    )
    async_execution: bool = Field(
        default=False, description="Whether to execute this task asynchronously"
    )
    human_input: bool = Field(
        default=False, description="Whether this task requires human input/validation"
    )
    output: Optional[TaskOutput] = Field(
        default=None, description="Output from the last execution of this task"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def execute(
        self,
        context: Optional[Dict[str, Any]] = None,
        run_context: Optional[RunContext] = None,
    ) -> RunResult:
        """Execute the task."""
        if not self.agent:
            raise ValueError("No agent assigned to execute this task")

        try:
            self.status = TaskStatus.RUNNING

            # Prepare execution context
            execution_context = {}

            # Add context from dependent tasks
            if self.context:
                for task in self.context:
                    if task.output:
                        execution_context[task.description] = task.output.raw

            # Add additional context
            if context:
                execution_context.update(context)

            # Execute with pydantic-ai's dependency injection
            result = await self.agent.execute(
                self.description,
                context=execution_context,
                dependencies=run_context,
            )

            # Store output
            self.output = TaskOutput(
                description=self.description,
                raw=result.data,
                metadata={
                    "expected_output": self.expected_output,
                    "status": self.status,
                },
            )

            self.status = TaskStatus.COMPLETED
            return result

        except Exception as e:
            self.status = TaskStatus.FAILED
            raise

    async def execute_async(
        self,
        context: Optional[Dict[str, Any]] = None,
        run_context: Optional[RunContext] = None,
    ) -> RunResult:
        """Execute the task asynchronously."""
        return await self.execute(context, run_context)

    def interpolate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Interpolate variables in description and expected_output."""
        if inputs:
            if self.description:
                self.description = self.description.format(**inputs)
            if self.expected_output:
                self.expected_output = self.expected_output.format(**inputs)
