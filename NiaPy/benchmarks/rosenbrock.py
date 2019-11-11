# encoding=utf8

"""Rosenbrock benchmark."""

import math
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Rosenbrock']


class Rosenbrock(Benchmark):
    r"""Implementation of Rosenbrock benchmark function.

    Date: 2018

    Authors: Iztok Fister Jr. and Lucija Brezočnik

    License: MIT

    Function: **Rosenbrock function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^{D-1} \left (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-30, 30]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (1,...,1)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^{D-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)$

        Equation:
                \begin{equation}
                f(\mathbf{x}) = \sum_{i=1}^{D-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
                \end{equation}

        Domain:
                $-30 \leq x_i \leq 30$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """
    Name = ['Rosenbrock']

    def __init__(self, Lower=-30.0, Upper=30.0):
        r"""Initialize of Rosenbrock benchmark.

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
        return r'''$f(\mathbf{x}) = \sum_{i=1}^{D-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)$'''

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

            for i in range(D - 1):
                val += 100.0 * math.pow(sol[i + 1] - math.pow((sol[i]), 2),
                                        2) + math.pow((sol[i] - 1), 2)

            return val

        return evaluate
