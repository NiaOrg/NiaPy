# encoding=utf8
# pylint: disable=anomalous-backslash-in-string, old-style-class
import math

__all__ = ['SchumerSteiglitz']


class SchumerSteiglitz:
    r"""Implementation of Schumer Steiglitz function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Schumer Steiglitz function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D x_i^4`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D x_i^4$

        Equation:
                \begin{equation} f(\mathbf{x}) = \sum_{i=1}^D x_i^4 \end{equation}

        Domain:
                $-100 \leq x_i \leq 100$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """

    def __init__(self, Lower=-100.0, Upper=100.0):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D):
                val += math.pow(sol[i], 4)

            return val

        return evaluate
