# encoding=utf8

"""Implementation of Salomon function."""

import math
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Salomon']


class Salomon(Benchmark):
    r"""Implementation of Salomon function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Salomon function**

        :math:`f(\mathbf{x}) = 1 - \cos\left(2\pi\sqrt{\sum_{i=1}^D x_i^2}
        \right)+ 0.1 \sqrt{\sum_{i=1}^D x_i^2}`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = f(0, 0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = 1 - \cos\left(2\pi\sqrt{\sum_{i=1}^D x_i^2}
                \right)+ 0.1 \sqrt{\sum_{i=1}^D x_i^2}$

        Equation:
                \begin{equation} f(\mathbf{x}) =
                1 - \cos\left(2\pi\sqrt{\sum_{i=1}^D x_i^2}
                \right)+ 0.1 \sqrt{\sum_{i=1}^D x_i^2} \end{equation}

        Domain:
                $-100 \leq x_i \leq 100$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """
    Name = ['Salomon']

    def __init__(self, Lower=-100.0, Upper=100.0):
        r"""Initialize of Salomon benchmark.

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
        return r'''$f(\mathbf{x}) = 1 - \cos\left(2\pi\sqrt{\sum_{i=1}^D x_i^2}
                \right)+ 0.1 \sqrt{\sum_{i=1}^D x_i^2}$'''

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

            return 1.0 - math.cos(2.0 * math.pi * math.sqrt(val)) + 0.1 * val

        return evaluate
