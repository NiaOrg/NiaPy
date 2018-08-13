# encoding=utf8
# pylint: disable=anomalous-backslash-in-string, old-style-class
import math

__all__ = ['Rosenbrock']


class Rosenbrock:
    r"""Implementation of Rosenbrock benchmark function.

    Date: 2018

    Authors: Iztok Fister Jr. and Lucija Brezočnik

    License: MIT

    Function: **Rosenbrock function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^{D-1} \left (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-30, 30]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (1,...,1)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^{D-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)$

        Equation:
                \begin{equation}
                f(\mathbf{x}) = \sum_{i=1}^{D-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
                \end{equation}

        Domain:
                $-30 \leq x_i \leq 30$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """

    def __init__(self, Lower=-30.0, Upper=30.0):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D - 1):
                val += 100.0 * math.pow(sol[i + 1] - math.pow((sol[i]), 2),
                                        2) + math.pow((sol[i] - 1), 2)

            return val

        return evaluate
