# encoding=utf8
# pylint: disable=anomalous-backslash-in-string, old-style-class
"""Implementations of Alpine functions."""

import math

__all__ = ['Alpine1', 'Alpine2']


class Alpine1:
    r"""Implementation of Alpine1 function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Alpine1 function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^{D} |x_i \sin(x_i)+0.1x_i|`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^{D} \left |x_i \sin(x_i)+0.1x_i \right|$

        Equation:
                \begin{equation} f(x) = \sum_{i=1}^{D} \left|x_i \sin(x_i) + 0.1x_i \right| \end{equation}

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
                val += abs(math.sin(sol[i]) + 0.1 * sol[i])

            return val

        return evaluate


class Alpine2:
    r"""Implementation of Alpine2 function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Alpine2 function**

        :math:`f(\mathbf{x}) = \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [0, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 2.808^D`, at :math:`x^* = (7.917,...,7.917)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i)$

        Equation:
                \begin{equation} f(\mathbf{x}) =
                \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i) \end{equation}

        Domain:
                $0 \leq x_i \leq 10$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.
    """

    def __init__(self, Lower=0.0, Upper=10.0):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 1.0

            for i in range(D):
                val *= math.sqrt(sol[i]) * math.sin(sol[i])

            return val

        return evaluate
