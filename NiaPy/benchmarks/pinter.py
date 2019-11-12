# encoding=utf8

"""Implementation of Pinter function."""

import math
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Pinter']


class Pinter(Benchmark):
    r"""Implementation of Pintér function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Pintér function**

        :math:`f(\mathbf{x}) =
        \sum_{i=1}^D ix_i^2 + \sum_{i=1}^D 20i \sin^2 A + \sum_{i=1}^D i
        \log_{10} (1 + iB^2);`
        :math:`A = (x_{i-1}\sin(x_i)+\sin(x_{i+1}))\quad \text{and} \quad`
        :math:`B = (x_{i-1}^2 - 2x_i + 3x_{i+1} - \cos(x_i) + 1)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) =
                \sum_{i=1}^D ix_i^2 + \sum_{i=1}^D 20i \sin^2 A + \sum_{i=1}^D i
                \log_{10} (1 + iB^2);
                A = (x_{i-1}\sin(x_i)+\sin(x_{i+1}))\quad \text{and} \quad
                B = (x_{i-1}^2 - 2x_i + 3x_{i+1} - \cos(x_i) + 1)$

        Equation:
                \begin{equation} f(\mathbf{x}) = \sum_{i=1}^D ix_i^2 +
                \sum_{i=1}^D 20i \sin^2 A + \sum_{i=1}^D i \log_{10} (1 + iB^2);
                A = (x_{i-1}\sin(x_i)+\sin(x_{i+1}))\quad \text{and} \quad
                B = (x_{i-1}^2 - 2x_i + 3x_{i+1} - \cos(x_i) + 1) \end{equation}

        Domain:
                $-10 \leq x_i \leq 10$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """
    Name = ['Pinter']

    def __init__(self, Lower=-10.0, Upper=10.0):
        r"""Initialize of Pinter benchmark.

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
        return r''' $f(\mathbf{x}) =
                \sum_{i=1}^D ix_i^2 + \sum_{i=1}^D 20i \sin^2 A + \sum_{i=1}^D i
                \log_{10} (1 + iB^2);
                A = (x_{i-1}\sin(x_i)+\sin(x_{i+1}))\quad \text{and} \quad
                B = (x_{i-1}^2 - 2x_i + 3x_{i+1} - \cos(x_i) + 1)$'''

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
            val3 = 0.0

            for i in range(D):

                if i == 0:
                    sub = sol[D - 1]
                    add = sol[i + 1]
                elif i == D - 1:
                    sub = sol[i - 1]
                    add = sol[0]
                else:
                    sub = sol[i - 1]
                    add = sol[i + 1]

                A = (sub * math.sin(sol[i]) + math.sin(add))
                B = (math.pow(sub, 2) - 2.0 * sol[i] + 3.0 * add - math.cos(sol[i]) + 1.0)

                val1 += (i + 1.0) * math.pow(sol[i], 2)
                val2 += 20.0 * (i + 1.0) * math.pow(math.sin(A), 2)
                val3 += (i + 1.0) * math.log10(1.0 + (i + 1.0) * math.pow(B, 2))

            return val1 + val2 + val3

        return evaluate
