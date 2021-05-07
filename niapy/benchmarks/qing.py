# encoding=utf8


"""Implementation of Qing funcion."""

import math
from niapy.benchmarks.benchmark import Benchmark

__all__ = ['Qing']


class Qing(Benchmark):
    r"""Implementation of Qing function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Qing function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \left(x_i^2 - i\right)^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-500, 500]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (\pm √i))`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = \sum_{i=1}^D \left (x_i^2 - i\right)^2$

        Equation:
            \begin{equation} f(\mathbf{x}) =
            \sum_{i=1}^D \left{(x_i^2 - i\right)}^2 \end{equation}

        Domain:
            $-500 \leq x_i \leq 500$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    Name = ['Qing']

    def __init__(self, lower=-500.0, upper=500.0):
        r"""Initialize of Qing benchmark.

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
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D \left (x_i^2 - i\right)^2$'''

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
                val += math.pow(math.pow(x[i], 2) - i, 2)

            return val

        return evaluate
