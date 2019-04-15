# encoding=utf8

"""Implemenatation of Rastrigin benchmark."""

import math
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ["Rastrigin"]


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

    Reference: https://www.sfu.ca/~ssurjano/rastr.html

    """

    Name = ['Rastrigin']

    def __init__(self, Lower=-5.12, Upper=5.12):
        """Initialize Rastrigin benchmark.

        Args:
            Lower (Optional[float]): Lower bound of problem.
            Upper (Optional[float]): Upper bound of problem.

        See Also:
            :func:`NiaPy.benchmarks.Benchmark.__init__`

        """

        Benchmark.__init__(self, Lower, Upper)

    @staticmethod
    def latex_code():
        """Return the latex code of the problem.

        Returns:
            [str] -- latex code.

        """
        return r'''$f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)$'''

    @classmethod
    def function(cls):
        """Return benchmark evaluation function.

        Returns:
            [fun] -- Evaluation function.

        """

        def evaluate(D, sol):

            val = 0.0

            for i in range(D):
                val += math.pow(sol[i], 2) - (10.0 * math.cos(2 * math.pi * sol[i]))

            return 10 * D + val

        return evaluate
