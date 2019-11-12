# encoding=utf8

"""Implementation of Ridge function."""

import math
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Ridge']


class Ridge(Benchmark):
    r"""Implementation of Ridge function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Ridge function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-64, 64]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2 $

        Equation:
                \begin{equation} f(\mathbf{x}) =
                \sum_{i=1}^D (\sum_{j=1}^i x_j)^2 \end{equation}

        Domain:
                $-64 \leq x_i \leq 64$

    Reference: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ridge.html
    """
    Name = ['Ridge']

    def __init__(self, Lower=-64.0, Upper=64.0):
        r"""Initialize of Ridge benchmark.

        Args:
            Lower (Optional[float]): Lower bound of problem.
            Upper (Optional[float]): Upper bound of problem.

        See Also:
            :func:`NiaPy.benchmarks.Benchmark.__init__`
        """
        Benchmark.__init__(self, Lower, Upper)

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code
        """
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2 $'''

    def function(self):
        r"""Return benchmark evaluation function.

        Returns:
            Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function
        """
        def evaluate(D, sol):
            r"""Fitness function.

            Args:
                D (int): Dimensionality of the problem
                sol (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

            Returns:
                float: Fitness value for the solution.
            """
            val = 0.0

            for i in range(D):
                val1 = 0.0
                for j in range(i + 1):
                    val1 += sol[j]
                val += math.pow(val1, 2)

            return val

        return evaluate
