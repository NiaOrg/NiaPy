# encoding=utf8

"""Implementation of SchumerSteiglitz function."""

import math
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['SchumerSteiglitz']


class SchumerSteiglitz(Benchmark):
    r"""Implementation of Schumer Steiglitz function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Schumer Steiglitz function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D x_i^4`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D x_i^4$

        Equation:
                \begin{equation} f(\mathbf{x}) = \sum_{i=1}^D x_i^4 \end{equation}

        Domain:
                $-100 \leq x_i \leq 100$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """
    Name = ['SchumerSteiglitz']

    def __init__(self, Lower=-100.0, Upper=100.0):
        r"""Initialize of Schumer Steiglitz benchmark.

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
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D x_i^4$'''

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
                val += math.pow(sol[i], 4)

            return val

        return evaluate
