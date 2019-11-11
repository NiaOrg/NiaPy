# encoding=utf8

"""Stybliski Tang benchmark."""

import math
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['StyblinskiTang']

class StyblinskiTang(Benchmark):
    r"""Implementation of Styblinski-Tang functions.

    Date: 2018

    Authors: Lucija Brezočnik

    License: MIT

    Function: **Styblinski-Tang function**

        :math:`f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^D \left(
        x_i^4 - 16x_i^2 + 5x_i \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-5, 5]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = -78.332`, at :math:`x^* = (-2.903534,...,-2.903534)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^D \left(
                x_i^4 - 16x_i^2 + 5x_i \right) $

        Equation:
                \begin{equation}f(\mathbf{x}) =
                \frac{1}{2} \sum_{i=1}^D \left( x_i^4 - 16x_i^2 + 5x_i \right) \end{equation}

        Domain:
                $-5 \leq x_i \leq 5$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """
    Name = ['StyblinskiTang']

    def __init__(self, Lower=-5.0, Upper=5.0):
        r"""Initialize of Styblinski Tang benchmark.

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
        return r'''$f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^D \left(
                x_i^4 - 16x_i^2 + 5x_i \right) $'''

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
                val += (math.pow(sol[i], 4) - 16.0 * math.pow(sol[i], 2) + 5.0 * sol[i])

            return 0.5 * val

        return evaluate
