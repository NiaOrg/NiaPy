"""Implementation of Griewank function.

Date: February 2018

Authors: Iztok Fister Jr. and Lucija Brezočnik

License: MIT

Function: Griewank function

Input domain:
    The function can be defined on any input domain but it is usually
    evaluated on the hypercube x_i ∈ [-100, 100], for all i = 1, 2,..., D.

Global minimum:
    f(x*) = 0, at x* = (0,...,0)

LaTeX formats:
    Inline: $f(x) = \sum_{i=1}^D \frac{x_i^2}{4000} -
            \prod_{i=1}^D cos(\frac{x_i}{\sqrt{i}}) + 1$
    Equation: \begin{equation} f(x) = \sum_{i=1}^D \frac{x_i^2}{4000} -
              \prod_{i=1}^D cos(\frac{x_i}{\sqrt{i}}) + 1 \end{equation}
    Domain: $-100 \leq x_i \leq 100$

Reference paper:
    Jamil, M., and Yang, X. S. (2013).
    A literature survey of benchmark functions for global optimisation problems.
    International Journal of Mathematical Modelling and Numerical Optimisation,
    4(2), 150-194.
"""

import math

__all__ = ['Griewank']


class Griewank(object):

    def __init__(self, Lower=-100, Upper=100):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val1 = 0.0
            val2 = 1.0

            for i in range(D):
                val1 += (math.pow(sol[i], 2) / 4000)
                val2 *= (math.cos(sol[i] / math.sqrt(i + 1)))

            return val1 - val2 + 1

        return evaluate
