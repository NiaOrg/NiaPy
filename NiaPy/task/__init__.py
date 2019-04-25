"""Module with implementations of tasks."""

from NiaPy.task.task import (
    Task,
    CountingTask,
    StoppingTask,
    MoveTask,
    TaskComposition,
    TaskConvPlot,
    TaskConvPrint,
    TaskConvSave,
    ThrowingTask,
    ScaledTask,
    OptimizationType
)

__all__ = [
    "Task",
    "CountingTask",
    "StoppingTask",
    "MoveTask",
    "TaskComposition",
    "TaskConvPlot",
    "TaskConvPrint",
    "TaskConvSave",
    "ThrowingTask",
    "ScaledTask",
    "OptimizationType"
]
