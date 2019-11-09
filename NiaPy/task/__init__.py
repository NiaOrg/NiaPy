"""Module with implementations of tasks."""

from NiaPy.task.task import (
    Task,
    CountingTask,
    StoppingTask,
    ThrowingTask,
    OptimizationType
)
from NiaPy.task.utility import Utility

__all__ = [
    'Task',
    'CountingTask',
    'StoppingTask',
    'ThrowingTask',
    'OptimizationType',
    'Utility'
]
