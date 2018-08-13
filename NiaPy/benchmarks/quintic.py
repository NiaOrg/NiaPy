# encoding=utf8
# pylint: disable=anomalous-backslash-in-string, old-style-class
import math

__all__ = ['Quintic']


class Quintic:
    r"""Implementation of Quintic function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Quintic function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \left| x_i^5 - 3x_i^4 +
        4x_i^3 + 2x_i^2 - 10x_i - 4\right|`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = f(-1\; \text{or}\; 2)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D \left| x_i^5 - 3x_i^4 +
                4x_i^3 + 2x_i^2 - 10x_i - 4\right|$

        Equation:
                \begin{equation} f(\mathbf{x}) =
                \sum_{i=1}^D \left| x_i^5 - 3x_i^4 + 4x_i^3 + 2x_i^2 -
                10x_i - 4\right| \end{equation}

        Domain:
                $-10 \leq x_i \leq 10$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """

    def __init__(self, Lower=-10.0, Upper=10.0):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D):
                val += abs(math.pow(sol[i], 5) - 3.0 * math.pow(sol[i], 4) + 4.0 *
                           math.pow(sol[i], 3) + 2.0 * math.pow(sol[i], 2) - 10.0 * sol[i] - 4)

            return val

        return evaluate
