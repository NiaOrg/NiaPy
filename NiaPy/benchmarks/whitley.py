# encoding=utf8
# pylint: disable=anomalous-backslash-in-string
"""Implementation of Whitley function.

Date: Februar 2018

Author: Grega Vrbančič

License: MIT

Function: Whitley function

Minimum:
    f(1,1,...,1) = 0

LaTeX formats:
    Equation: \begin{equation}f(x_1 \cdots x_n) =
                \sum_{i=1}^n \sum_{j=1}^n (\frac{(100(x_i^2-x_j)^2 +
                (1-x_j)^2)^2}{4000} -cos(100(x_i^2-x_j)^2 + (1-x_j)^2)+1)
                \end{equation}
    Domain: $-10.24 \leq x_i \leq 10.24$

Reference paper: http://ieeexplore.ieee.org/document/4425088/

Implementation based on: http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/whitley.html
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
                    temp = 100 * (math.pow(sol[i], 2) - sol[j]) + math.pow(1 - sol[j], 2)
                    val += (float(math.pow(temp, 2)) / 4000.0) - math.cos(temp) + 1
            return val

        return evaluate
