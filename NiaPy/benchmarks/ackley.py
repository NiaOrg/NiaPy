# encoding=utf8
# pylint: disable=anomalous-backslash-in-string, old-style-class
import math

__all__ = ['Ackley']

class Ackley:
    r"""Implementation of Ackley function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Ackley function**

        :math:`f(\mathbf{x}) = -a\;\exp\left(-b \sqrt{\frac{1}{D}\sum_{i=1}^D x_i^2}\right)
        - \exp\left(\frac{1}{D}\sum_{i=1}^D \cos(c\;x_i)\right) + a + \exp(1)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-32.768, 32.768]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(\textbf{x}^*) = 0`, at  :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = -a\;\exp\left(-b \sqrt{\frac{1}{D}
                \sum_{i=1}^D x_i^2}\right) - \exp\left(\frac{1}{D}
                \sum_{i=1}^D cos(c\;x_i)\right) + a + \exp(1)$

        Equation:
                \begin{equation}f(\mathbf{x}) =
                -a\;\exp\left(-b \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2}\right) -
                \exp\left(\frac{1}{D} \sum_{i=1}^D \cos(c\;x_i)\right) +
                a + \exp(1) \end{equation}

        Domain:
                $-32.768 \leq x_i \leq 32.768$

    Reference: https://www.sfu.ca/~ssurjano/ackley.html
    """

    def __init__(self, Lower=-32.768, Upper=32.768):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        """Return benchmark evaluation function."""
        def evaluate(D, sol):

            a = 20  # Recommended variable value
            b = 0.2  # Recommended variable value
            c = 2 * math.pi  # Recommended variable value

            val = 0.0
            val1 = 0.0
            val2 = 0.0

            for i in range(D):
                val1 += math.pow(sol[i], 2)
                val2 += math.cos(c * sol[i])

            temp1 = -b * math.sqrt(val1 / D)
            temp2 = val2 / D

            val = -a * math.exp(temp1) - math.exp(temp2) + a + math.exp(1)

            return val

        return evaluate
