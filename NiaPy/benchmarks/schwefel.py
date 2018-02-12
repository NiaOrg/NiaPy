# pylint: disable=anomalous-backslash-in-string
"""Implementation of Schwefel function.

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

import math

__all__ = ['Schwefel']


class Schwefel(object):

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
