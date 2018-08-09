# encoding=utf8
# pylint: disable=anomalous-backslash-in-string, old-style-class
import math

__all__ = ['HappyCat']


class HappyCat:
    r"""Implementation of Happy cat function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Happy cat function**

        :math:`f(\mathbf{x}) = {\left |\sum_{i = 1}^D {x_i}^2 - D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 + \sum_{i = 1}^D x_i) / D + 0.5`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (-1,...,-1)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = {\left|\sum_{i = 1}^D {x_i}^2 -
                D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 +
                \sum_{i = 1}^D x_i) / D + 0.5$

        Equation:
                \begin{equation} f(\mathbf{x}) = {\left| \sum_{i = 1}^D {x_i}^2 -
                D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 +
                \sum_{i = 1}^D x_i) / D + 0.5 \end{equation}

        Domain:
                $-100 \leq x_i \leq 100$

    Reference: http://bee22.com/manual/tf_images/Liang%20CEC2014.pdf &
    Beyer, H. G., & Finck, S. (2012). HappyCat - A Simple Function Class Where Well-Known Direct Search Algorithms Do Fail.
    In International Conference on Parallel Problem Solving from Nature (pp. 367-376). Springer, Berlin, Heidelberg.
    """

    def __init__(self, Lower=-100.0, Upper=100.0):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val1 = 0.0
            val2 = 0.0
            alpha = 0.125

            for i in range(D):
                val1 += math.pow(abs(math.pow(sol[i], 2) - D), alpha)
                val2 += (0.5 * math.pow(sol[i], 2) + sol[i]) / D

            return val1 + val2 + 0.5

        return evaluate
