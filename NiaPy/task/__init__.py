"""Module with implementations of tasks."""

from NiaPy.task.task import Task
from NiaPy.task.countingtask import CountingTask
from NiaPy.task.stoppingtask import StoppingTask
from NiaPy.task.throwingtask import ThrowingTask
from NiaPy.task.optimizationtype import OptimizationType

__all__ = [
    'Task',
    'CountingTask',
    'StoppingTask',
    'ThrowingTask',
    'OptimizationType'
]
