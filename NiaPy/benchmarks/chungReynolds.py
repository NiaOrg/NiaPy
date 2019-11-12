# encoding=utf8

"""Implementaion of Chung Reynolds function."""

import math
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['ChungReynolds']


class ChungReynolds(Benchmark):
    r"""Implementation of Chung Reynolds functions.

    Date: 2018

    Authors: Lucija Brezočnik

    License: MIT

    Function: **Chung Reynolds function**

        :math:`f(\mathbf{x}) = \left(\sum_{i=1}^D x_i^2\right)^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \left(\sum_{i=1}^D x_i^2\right)^2$

        Equation:
                \begin{equation} f(\mathbf{x}) = \left(\sum_{i=1}^D x_i^2\right)^2 \end{equation}

        Domain:
                $-100 \leq x_i \leq 100$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """
    Name = ['ChungReynolds']

    def __init__(self, Lower=-100.0, Upper=100.0):
        r"""Initialize of Chung Reynolds benchmark.

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
        return r'''$f(\mathbf{x}) = \left(\sum_{i=1}^D x_i^2\right)^2$'''

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
                val += math.pow(sol[i], 2)

            return math.pow(val, 2)

        return evaluate
