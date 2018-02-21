# encoding=utf8
# pylint: disable=anomalous-backslash-in-string
"""Implementations of Schwefels functions."""

import math

__all__ = ['Schwefel', 'Schwefel221', 'Schwefel222']


class Schwefel(object):
    r"""Implementation of Schewel function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Schwefel function**

        :math:`f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i \sin(\sqrt{|x_i|})`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-500, 500]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

    LaTeX formats:
        Inline:
                $f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i \sin(\sqrt{|x_i|})$

        Equation:
                \begin{equation} f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i
                \sin(\sqrt{|x_i|}) \end{equation}

        Domain:
                $-500 \leq x_i \leq 500$

    Reference: https://www.sfu.ca/~ssurjano/schwef.html
    """

    def __init__(self, Lower=-500.0, Upper=500.0):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D):
                val += (sol[i] * math.sin(math.sqrt(abs(sol[i]))))

            return 418.9829 * D - val

        return evaluate


class Schwefel221(object):
    r"""Schwefel 2.21 function implementation.

    Date: 2018

    Author: Grega Vrbančič

    Licence: MIT

    Function: **Schwefel 2.21 function**

        :math:`f(\mathbf{x})=\max_{i=1,...,D}|x_i|`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x})=\max_{i=1,...,D}|x_i|$

        Equation:
                \begin{equation}f(\mathbf{x}) = \max_{i=1,...,D}|x_i| \end{equation}

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
            maximum = 0.0

            for i in range(D):
                if abs(sol[i]) > maximum:
                    maximum = abs(sol[i])

            return maximum

        return evaluate


class Schwefel222(object):
    r"""Schwefel 2.22 function implementation.

    Date: 2018

    Author: Grega Vrbančič

    Licence: MIT

    Function: **Schwefel 2.22 function**

        :math:`f(\mathbf{x})=\sum_{i=1}^{D}|x_i|+\prod_{i=1}^{D}|x_i|`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x})=\sum_{i=1}^{D}|x_i|+\prod_{i=1}^{D}|x_i|$

        Equation:
                \begin{equation}f(\mathbf{x}) = \sum_{i=1}^{D}|x_i| +
                \prod_{i=1}^{D}|x_i| \end{equation}

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

            part1 = 0.0
            part2 = 1.0

            for i in range(D):
                part1 += abs(sol[i])
                part2 *= abs(sol[i])

            return part1 + part2

        return evaluate
