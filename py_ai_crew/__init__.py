"""
py_ai_crew - A hierarchical task execution framework for AI agents with goal-based validation
"""

from .models import Task, TaskStatus
from .flow import Flow, FlowCondition, NextAction
from .executor import TaskExecutor
from .goals import Goal, GoalSet, GoalStatus

__version__ = "0.1.0"
__all__ = [
    "Task",
    "TaskStatus",
    "Flow",
    "FlowCondition",
    "NextAction",
    "TaskExecutor",
    "Goal",
    "GoalSet",
    "GoalStatus",
]
