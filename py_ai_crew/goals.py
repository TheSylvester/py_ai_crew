from typing import Any, Callable, Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai.result import RunResult


class GoalStatus(str, Enum):
    """Status of a goal."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    FAILED = "failed"


class Goal(BaseModel):
    """Base class for defining execution goals."""

    name: str = Field(description="Name of the goal")
    description: str = Field(
        description="Description of what this goal aims to achieve"
    )
    validation_fn: Callable[[RunResult, Any], bool] = Field(
        description="Function to validate if the goal is met"
    )
    status: GoalStatus = Field(default=GoalStatus.NOT_STARTED)
    max_attempts: int = Field(default=3, ge=1)
    current_attempts: int = Field(default=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def check(self, result: RunResult, context: Optional[Any] = None) -> bool:
        """Check if the goal is met based on the result."""
        self.current_attempts += 1
        is_achieved = self.validation_fn(result, context)

        if is_achieved:
            self.status = GoalStatus.ACHIEVED
        elif self.current_attempts >= self.max_attempts:
            self.status = GoalStatus.FAILED
        else:
            self.status = GoalStatus.IN_PROGRESS

        return is_achieved

    def reset(self) -> None:
        """Reset the goal status."""
        self.status = GoalStatus.NOT_STARTED
        self.current_attempts = 0


class GoalSet(BaseModel):
    """A collection of goals that need to be achieved."""

    goals: dict[str, Goal] = Field(default_factory=dict)
    require_all: bool = Field(
        default=True,
        description="If True, all goals must be met. If False, at least one goal must be met.",
    )

    def add_goal(self, goal_id: str, goal: Goal) -> None:
        """Add a goal to the set."""
        self.goals[goal_id] = goal

    def check_goals(self, result: RunResult, context: Optional[Any] = None) -> bool:
        """Check if goals are met based on the result."""
        if not self.goals:
            return True

        results = [goal.check(result, context) for goal in self.goals.values()]

        return all(results) if self.require_all else any(results)

    def get_failed_goals(self) -> list[str]:
        """Get list of failed goal IDs."""
        return [
            goal_id
            for goal_id, goal in self.goals.items()
            if goal.status == GoalStatus.FAILED
        ]

    def get_pending_goals(self) -> list[str]:
        """Get list of pending goal IDs."""
        return [
            goal_id
            for goal_id, goal in self.goals.items()
            if goal.status in (GoalStatus.NOT_STARTED, GoalStatus.IN_PROGRESS)
        ]

    def reset_all(self) -> None:
        """Reset all goals."""
        for goal in self.goals.values():
            goal.reset()
