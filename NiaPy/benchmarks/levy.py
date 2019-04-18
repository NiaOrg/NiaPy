# encoding=utf8

"""Implementation of Levy benchmark."""

from numpy import sin, pi
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ["Levy"]


class Levy(Benchmark):
    r"""Implementations of Levy functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function: **Levy Function**

        :math:`f(\textbf{x}) = \sin^2 (\pi w_1) + \sum_{i = 1}^{D - 1} (w_i - 1)^2 \left( 1 + 10 \sin^2 (\pi w_i + 1) \right) + (w_d - 1)^2 (1 + \sin^2 (2 \pi w_d)) \\ w_i = 1 + \frac{x_i - 1}{4}`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (1, \cdots, 1)`

    LaTeX formats:
        Inline:
                $f(\textbf{x}) = \sin^2 (\pi w_1) + \sum_{i = 1}^{D - 1} (w_i - 1)^2 \left( 1 + 10 \sin^2 (\pi w_i + 1) \right) + (w_d - 1)^2 (1 + \sin^2 (2 \pi w_d)) \\ w_i = 1 + \frac{x_i - 1}{4}$

        Equation:
                \begin{equation} f(\textbf{x}) = \sin^2 (\pi w_1) + \sum_{i = 1}^{D - 1} (w_i - 1)^2 \left( 1 + 10 \sin^2 (\pi w_i + 1) \right) + (w_d - 1)^2 (1 + \sin^2 (2 \pi w_d)) \\ w_i = 1 + \frac{x_i - 1}{4} \end{equation}

        Domain:
                $-10 \leq x_i \leq 10$

    Reference:
        https://www.sfu.ca/~ssurjano/levy.html

    """

    Name = ["Levy"]

    def __init__(self, Lower=0.0, Upper=pi):
        r"""Initialize Levy benchmark.

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

        return r"""$f(\textbf{x}) = \sin^2 (\pi w_1) + \sum_{i = 1}^{D - 1} (w_i - 1)^2 \left( 1 + 10 \sin^2 (\pi w_i + 1) \right) + (w_d - 1)^2 (1 + \sin^2 (2 \pi w_d)) \\ w_i = 1 + \frac{x_i - 1}{4}$"""

    @classmethod
    def function(cls):
        """Return benchmark evaluation function.

        Returns:
            [fun] -- Evaluation function.

        """

        def w(x):
            return 1 + (x - 1) / 4

        def evaluate(D, sol):

            v = 0.0

            for i in range(D - 1):
                v += (w(sol[i]) - 1) ** 2 * (1 + 10 * sin(pi * w(sol[i]) + 1) ** 2) + (w(sol[-1]) - 1) ** 2 * (1 + sin(2 * pi * w(sol[-1]) ** 2))

            return sin(pi * w(sol[0])) ** 2 + v

        return evaluate
