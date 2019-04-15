# encoding=utf8

"""Implementation of Perm benchmark."""

from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ["Perm"]


class Perm(Benchmark):
    r"""Implementations of Perm functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Arguments:
    beta {real} -- value added to inner sum of funciton

    Function: **Perm Function**

        :math:`f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-D, D]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (1, \frac{1}{2}, \cdots , \frac{1}{i} , \cdots , \frac{1}{D})`

    LaTeX formats:
        Inline:
                $f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2$

        Equation:
                \begin{equation} f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2 \end{equation}

        Domain:
                $-D \leq x_i \leq D$

    Reference:
        https://www.sfu.ca/~ssurjano/perm0db.html

    """

    Name = ["Perm"]

    def __init__(self, D=10.0, beta=0.5):
        """Initialize Perm benchmark.

        Args:
            D [float] -- Dimension on problem. (default: {10.0})
            beta [float] -- beta parameter. (default: {0.5})

        See Also:
            :func:`NiaPy.benchmarks.Benchmark.__init__`

        """

        Benchmark.__init__(self, -D, D)
        Perm.beta = beta

    @staticmethod
    def latex_code():
        """Return the latex code of the problem.

        Returns:
            [str] -- latex code.

        """

        return r"""$f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2$"""

    @classmethod
    def function(cls):
        """Return benchmark evaluation function.

        Returns:
            [fun] -- Evaluation function.

        """

        def evaluate(D, sol):
            v = 0.0

            for i in range(1, D + 1):
                vv = .0

                for j in range(1, D + 1):
                    vv += (j + cls.beta) * (sol[j - 1] ** i - 1 / j ** i)

                v += vv ** 2

            return v

        return evaluate
