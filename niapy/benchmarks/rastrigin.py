# encoding=utf8

"""Implementation of Rastrigin function."""

import math
from niapy.benchmarks.benchmark import Benchmark

__all__ = ['Rastrigin']


class Rastrigin(Benchmark):
    r"""Implementation of Rastrigin benchmark function.

    Date: 2018

    Authors: Lucija Brezočnik and Iztok Fister Jr.

    License: MIT

    Function: **Rastrigin function**

        :math:`f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-5.12, 5.12]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)$

        Equation:
            \begin{equation} f(\mathbf{x}) =
            10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)
            \end{equation}

        Domain:
            $-5.12 \leq x_i \leq 5.12$

    Reference:
        https://www.sfu.ca/~ssurjano/rastr.html

    """

    Name = ['Rastrigin']

    def __init__(self, lower=-5.12, upper=5.12):
        r"""Initialize of Rastrigin benchmark.

        Args:
            lower (Optional[float]): Lower bound of problem.
            upper (Optional[float]): Upper bound of problem.

        See Also:
            :func:`niapy.benchmarks.Benchmark.__init__`

        """
        super().__init__(lower, upper)

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)$'''

    def function(self):
        r"""Return benchmark evaluation function.

        Returns:
            Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function.

        """
        def evaluate(dimension, x):
            r"""Fitness function.

            Args:
                dimension (int): Dimensionality of the problem
                x (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

            Returns:
                float: Fitness value for the solution.

            """
            val = 0.0

            for i in range(dimension):
                val += math.pow(x[i], 2) - (10.0 * math.cos(2 * math.pi * x[i]))

            return 10 * dimension + val

        return evaluate
