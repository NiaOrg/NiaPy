# encoding=utf8

"""Implementations of Powell benchmark."""

from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ["Powell"]


class Powell(Benchmark):
    r"""Implementations of Powell functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function: **Levy Function**

        :math:`f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-4, 5]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (0, \cdots, 0)`

    LaTeX formats:
        Inline:
                $f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right)$

        Equation:
                \begin{equation} f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right) \end{equation}

        Domain:
                $-4 \leq x_i \leq 5$

    Reference:
        https://www.sfu.ca/~ssurjano/levy.html

    """

    Name = ["Powell"]

    def __init__(self, Lower=-4.0, Upper=5.0):
        r"""Initialize Powell benchmark.

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

        return r"""$f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right)$"""

    @classmethod
    def function(cls):
        """Return benchmark evaluation function.

        Returns:
            [fun] -- Evaluation function.

        """

        def evaluate(D, sol):
            v = 0.0

            for i in range(1, (D // 4) + 1):
                v += (sol[4 * i - 4] + 10 * sol[4 * i - 3]) ** 2 + 5 * (sol[4 * i - 2] - sol[4 * i - 1]) ** 2 + (sol[4 * i - 3] - 2 * sol[4 * i - 2]) ** 4 + 10 * (sol[4 * i - 4] - sol[4 * i - 1]) ** 4

            return v
        return evaluate
