# encoding=utf8

"""The implementation of optimization type."""

from enum import Enum


class OptimizationType(Enum):
    r"""Enum representing type of optimization.

    Attributes:
        MINIMIZATION (int): Represents minimization problems and is default optimization type of all algorithms.
        MAXIMIZATION (int): Represents maximization problems.

    """

    MINIMIZATION = 1.0
    MAXIMIZATION = -1.0
