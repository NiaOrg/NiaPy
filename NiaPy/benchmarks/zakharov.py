# encoding=utf8

"""Implementations of Zakharov function."""

from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ["Zakharov"]


class Zakharov(Benchmark):
    r"""Implementations of Zakharov functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
    **Levy Function**

        :math:`f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-5, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (0, \cdots, 0)`

    LaTeX formats:
        Inline:
                $f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4$

        Equation:
                \begin{equation} f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4 \end{equation}

        Domain:
                $-5 \leq x_i \leq 10$

    Reference: https://www.sfu.ca/~ssurjano/levy.html

    """

    Name = ["Zakharov"]

    def __init__(self, Lower=-5.0, Upper=10.0):
        r"""Initialize Zakharov benchmark.

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

        return r"""$f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4$"""

    @classmethod
    def function(cls):
        """Return benchmark evaluation function.

        Returns:
            [fun] -- Evaluation function.

        """

        def evaluate(D, sol):
            v1, v2 = 0.0, 0.0

            for i in range(D):
                v1, v2 = v1 + sol[i] ** 2, v2 + 0.5 * (i + 1) * sol[i]

            return v1 + v2 ** 2 + v2 ** 4

        return evaluate
