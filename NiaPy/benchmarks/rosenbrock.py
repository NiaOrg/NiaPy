# encoding=utf8
# pylint: disable=anomalous-backslash-in-string
"""Implementation of Rosenbrock benchmark function.

Date: 2018

Authors: Iztok Fister Jr. and Lucija Brezočnik

License: MIT

Function: Rosenbrock function

Input domain:
    The function can be defined on any input domain but it is usually
    evaluated on the hypercube x_i ∈ [-30, 30], for all i = 1, 2,..., D.

Global minimum:
    f(x*) = 0, at x* = (1,...,1)

LaTeX formats:
    Inline: $f(x) = \sum_{i=1}^{D-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)$
    Equation: \begin{equation}
              f(x) = \sum_{i=1}^{D-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
              \end{equation}
    Domain: $-30 \leq x_i \leq 30$

Reference paper:
    Jamil, M., and Yang, X. S. (2013).
    A literature survey of benchmark functions for global optimisation problems.
    International Journal of Mathematical Modelling and Numerical Optimisation,
    4(2), 150-194.
"""

import math

__all__ = ['Rosenbrock']


class Rosenbrock(object):

    def __init__(self, Lower=-30, Upper=30):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0

            for i in range(D - 1):
                val += 100 * math.pow(sol[i + 1] - math.pow((sol[i]), 2), 2) + math.pow((sol[i] - 1), 2)

            return val

        return evaluate
