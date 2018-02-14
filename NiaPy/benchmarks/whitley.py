# encoding=utf8
# pylint: disable=anomalous-backslash-in-string
"""Implementation of Whitley function.

Date: February 2018

Author: Grega Vrbančič

License: MIT

Function: Whitley function

Minimum:
    f(1,1,...,1) = 0

LaTeX formats:
    Inline:
    Equation:
    Domain:

Reference paper:
    Jamil, M., and Yang, X. S. (2013).
    A literature survey of benchmark functions for global optimisation problems.
    International Journal of Mathematical Modelling and Numerical Optimisation,
    4(2), 150-194.
"""

import math

__all__ = ['Whitley']


class Whitley(object):
    def __init__(self, Lower=-10.24, Upper=10.24):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                for j in range(D):
                    temp = 100 * math.pow((math.pow(sol[i], 2) - sol[j]), 2) + math.pow(1 - sol[j], 2)
                    val += (float(math.pow(temp, 2)) / 4000.0) - math.cos(temp) + 1
            return val

        return evaluate
