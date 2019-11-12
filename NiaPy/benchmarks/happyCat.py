# encoding=utf8

"""Impementation of Happy Cat funtion."""

import math
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['HappyCat']


class HappyCat(Benchmark):
    r"""Implementation of Happy cat function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Happy cat function**

        :math:`f(\mathbf{x}) = {\left |\sum_{i = 1}^D {x_i}^2 - D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 + \sum_{i = 1}^D x_i) / D + 0.5`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (-1,...,-1)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = {\left|\sum_{i = 1}^D {x_i}^2 -
                D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 +
                \sum_{i = 1}^D x_i) / D + 0.5$

        Equation:
                \begin{equation} f(\mathbf{x}) = {\left| \sum_{i = 1}^D {x_i}^2 -
                D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 +
                \sum_{i = 1}^D x_i) / D + 0.5 \end{equation}

        Domain:
                $-100 \leq x_i \leq 100$

    Reference: http://bee22.com/manual/tf_images/Liang%20CEC2014.pdf &
    Beyer, H. G., & Finck, S. (2012). HappyCat - A Simple Function Class Where Well-Known Direct Search Algorithms Do Fail.
    In International Conference on Parallel Problem Solving from Nature (pp. 367-376). Springer, Berlin, Heidelberg.
    """
    Name = ['HappyCat']

    def __init__(self, Lower=-100.0, Upper=100.0):
        r"""Initialize of Happy cat benchmark.

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
        return r'''$f(\mathbf{x}) = {\left|\sum_{i = 1}^D {x_i}^2 -
                D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 +
                \sum_{i = 1}^D x_i) / D + 0.5$'''

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
            val1 = 0.0
            val2 = 0.0
            alpha = 0.125

            for i in range(D):
                val1 += math.pow(abs(math.pow(sol[i], 2) - D), alpha)
                val2 += (0.5 * math.pow(sol[i], 2) + sol[i]) / D

            return val1 + val2 + 0.5

        return evaluate
