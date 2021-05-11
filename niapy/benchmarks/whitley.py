# encoding=utf8

"""Whitley function."""

import math
from niapy.benchmarks.benchmark import Benchmark

__all__ = ['Whitley']


class Whitley(Benchmark):
    r"""Implementation of Whitley function.

    Date: 2018

    Authors: Grega Vrbančič and Lucija Brezočnik

    License: MIT

    Function: **Whitley function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \sum_{j=1}^D
        \left(\frac{(100(x_i^2-x_j)^2 + (1-x_j)^2)^2}{4000} -
        \cos(100(x_i^2-x_j)^2 + (1-x_j)^2)+1\right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-10.24, 10.24]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (1,...,1)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) =
            \sum_{i=1}^D \sum_{j=1}^D \left(\frac{(100(x_i^2-x_j)^2 +
            (1-x_j)^2)^2}{4000} - \cos(100(x_i^2-x_j)^2 + (1-x_j)^2)+1\right)$

        Equation:
            \begin{equation}f(\mathbf{x}) =
            \sum_{i=1}^D \sum_{j=1}^D \left(\frac{(100(x_i^2-x_j)^2 +
            (1-x_j)^2)^2}{4000} - \cos(100(x_i^2-x_j)^2 +
            (1-x_j)^2)+1\right) \end{equation}

        Domain:
            $-10.24 \leq x_i \leq 10.24$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    Name = ['Whitley']

    def __init__(self, lower=-10.24, upper=10.24):
        r"""Initialize of Whitley benchmark.

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
        return r'''$f(\mathbf{x}) =
                \sum_{i=1}^D \sum_{j=1}^D \left(\frac{(100(x_i^2-x_j)^2 +
                (1-x_j)^2)^2}{4000} - \cos(100(x_i^2-x_j)^2 + (1-x_j)^2)+1\right)$'''

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
                for j in range(dimension):
                    temp = 100 * math.pow((math.pow(x[i], 2) - x[j]), 2) + math.pow(1 - x[j], 2)
                    val += (float(math.pow(temp, 2)) / 4000.0) - math.cos(temp) + 1

            return val

        return evaluate
