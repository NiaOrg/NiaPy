# encoding=utf8
# pylint: disable=anomalous-backslash-in-string, old-style-class
import math

__all__ = ['Qing']


class Qing:
    r"""Implementation of Qing function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Qing function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \left(x_i^2 - i\right)^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-500, 500]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (\pm √i))`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D \left (x_i^2 - i\right)^2$

        Equation:
                \begin{equation} f(\mathbf{x}) =
                \sum_{i=1}^D \left{(x_i^2 - i\right)}^2 \end{equation}

        Domain:
                $-500 \leq x_i \leq 500$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """

    def __init__(self, Lower=-500.0, Upper=500.0):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D):
                val += math.pow(math.pow(sol[i], 2) - i, 2)

            return val

        return evaluate
