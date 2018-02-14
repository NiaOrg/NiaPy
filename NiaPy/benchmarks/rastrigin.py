"""Implementation of Rastrigin benchmark function.

Date: 2018

Authors: Lucija Brezočnik and Iztok Fister Jr.

License: MIT

Function: Rastrigin function

Input domain:
    The function can be defined on any input domain but it is usually
    evaluated on the hypercube x_i ∈ [-5.12, 5.12], for all i = 1, 2,..., D.

Global minimum:
    f(x*) = 0, at x* = (0,...,0)

LaTeX formats:
    Inline: $f(x) = 10D + \sum_{i=1}^D (x_i^2 -10cos(2\pi x_i))$
    Equation: \begin{equation}
              f(x) = 10D + \sum_{i=1}^D (x_i^2 -10cos(2\pi x_i))
              \end{equation}
    Domain: $-5.12 \leq x_i \leq 5.12$

Reference: https://www.sfu.ca/~ssurjano/rastr.html
"""

import math

__all__ = ['Rastrigin']


class Rastrigin(object):

    def __init__(self, Lower=-5.12, Upper=5.12):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            
            val = 0.0

            for i in range(D):
                val += math.pow(sol[i], 2) - (10 * math.cos(2 * math.pi * sol[i]))

            return 10 * D + val

        return evaluate
