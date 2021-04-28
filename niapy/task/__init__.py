"""Module with implementations of tasks."""

from niapy.task.task import (
    Task,
    CountingTask,
    StoppingTask,
    ThrowingTask,
    OptimizationType
)
from niapy.task.utility import Utility

__all__ = [
    'Task',
    'CountingTask',
    'StoppingTask',
    'ThrowingTask',
    'OptimizationType',
    'Utility'
]
