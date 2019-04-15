# encoding=utf8

"""Implementation of Stepint benchmark."""

import math
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ["Stepint"]


class Stepint(Benchmark):
    r"""Implementation of Stepint functions.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Stepint function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D x_i^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-5.12, 5.12]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (-5.12,...,-5.12)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D x_i^2$

        Equation:
                \begin{equation}f(\mathbf{x}) = \sum_{i=1}^D x_i^2 \end{equation}

        Domain:
                $0 \leq x_i \leq 10$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    Name = ["Stepint"]

    def __init__(self, Lower=-5.12, Upper=5.12):
        """Initialize Stepint benchmark.

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

        return r'''$f(\mathbf{x}) = \sum_{i=1}^D x_i^2$'''

    @classmethod
    def function(cls):
        """Return benchmark evaluation function.

        Returns:
            [fun] -- Evaluation function.

        """

        def evaluate(D, sol):

            val = 0.0

            for i in range(D):
                val += math.floor(sol[i])

            return 25.0 + val

        return evaluate
