# encoding=utf8
# pylint: disable=anomalous-backslash-in-string
"""Implementations of Schwefels functions."""

import math

__all__ = ['Schwefel', 'Schwefel221', 'Schwefel222']


class Schwefel(object):
    """Implementation of Schewel function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: Schwefel function

    Input domain:
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube x_i ∈ [-500, 500], for all i = 1, 2,..., D.

    Global minimum:
        f(x*) = 0, at x* = (420.9687,...,420.9687)

    LaTeX formats:
        Inline: $f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i sin(\sqrt{|x_i|})$
        Equation: \begin{equation} f(\textbf{x}) =
                418.9829d - \sum_{i=1}^{D} x_i
                sin(\sqrt{|x_i|}) \end{equation}
        Domain: $-500 \leq x_i \leq 500$

    Reference: https://www.sfu.ca/~ssurjano/schwef.html
    """

    def __init__(self, Lower=-500, Upper=500):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0
            val1 = 0.0

            for i in range(D):
                val1 += (sol[i] * math.sin(math.sqrt(abs(sol[i]))))

            val = 418.9829 * D - val1

            return val

        return evaluate


class Schwefel221(object):
    """Schwefel 2.21 function implementation.

    Date: February 2018

    Author: Grega Vrbančič

    Licence: MIT

    Function: Schwefel 2.21 function

    Input domain:

    Global minimum:

    LaTeX formats:
        Inline:
        Equation:
        Domain:

    Implementation based on: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/schwefel221.html
    """

    def __init__(self, Lower=-100, Upper=100):
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
    """Schwefel 2.22 function implementation.

    Date: February 2018

    Author: Grega Vrbančič

    Licence: MIT

    Function: Schwefel 2.22 function

    Input domain:

    Global minimum:

    LaTeX formats:
        Inline:
        Equation:
        Domain:

    Implementation based on: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/schwefel222.html
    """

    def __init__(self, Lower=-10, Upper=10):
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
