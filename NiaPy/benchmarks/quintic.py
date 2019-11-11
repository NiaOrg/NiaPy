# encoding=utf8

"""Implementaiton of Quintic funcion."""

import math
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Quintic']


class Quintic(Benchmark):
    r"""Implementation of Quintic function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Quintic function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \left| x_i^5 - 3x_i^4 +
        4x_i^3 + 2x_i^2 - 10x_i - 4\right|`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = f(-1\; \text{or}\; 2)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D \left| x_i^5 - 3x_i^4 +
                4x_i^3 + 2x_i^2 - 10x_i - 4\right|$

        Equation:
                \begin{equation} f(\mathbf{x}) =
                \sum_{i=1}^D \left| x_i^5 - 3x_i^4 + 4x_i^3 + 2x_i^2 -
                10x_i - 4\right| \end{equation}

        Domain:
                $-10 \leq x_i \leq 10$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """
    Name = ['Quintic']

    def __init__(self, Lower=-10.0, Upper=10.0):
        r"""Initialize of Quintic benchmark.

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
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D \left| x_i^5 - 3x_i^4 +
                4x_i^3 + 2x_i^2 - 10x_i - 4\right|$'''

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
                val += abs(math.pow(sol[i], 5) - 3.0 * math.pow(sol[i], 4) + 4.0 * math.pow(sol[i], 3) + 2.0 * math.pow(sol[i], 2) - 10.0 * sol[i] - 4)

            return val

        return evaluate
