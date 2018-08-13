# encoding=utf8
# pylint: disable=anomalous-backslash-in-string, old-style-class
import math

__all__ = ['Ridge']


class Ridge:
    r"""Implementation of Ridge function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Ridge function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-64, 64]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2 $

        Equation:
                \begin{equation} f(\mathbf{x}) =
                \sum_{i=1}^D (\sum_{j=1}^i x_j)^2 \end{equation}

        Domain:
                $-64 \leq x_i \leq 64$

    Reference: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ridge.html
    """

    def __init__(self, Lower=-64.0, Upper=64.0):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D):
                val1 = 0.0
                for j in range(i + 1):
                    val1 += sol[j]
                val += math.pow(val1, 2)

            return val

        return evaluate
