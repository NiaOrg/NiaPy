# encoding=utf8

"""Implementation of Ackley problem."""

import numpy as np

from niapy.problems.problem import Problem

__all__ = ['Ackley']


class Ackley(Problem):
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

    Reference:
        https://www.sfu.ca/~ssurjano/ackley.html

    """

    def __init__(self, dimension=4, lower=-32.768, upper=32.768, a=20.0, b=0.2, c=2 * np.pi, *args, **kwargs):
        r"""Initialize Ackley problem.

        Args:
            dimension (Optional[int]): Dimension of the problem.
            lower (Optional[Union[float, Iterable[float]]]): Lower bounds of the problem.
            upper (Optional[Union[float, Iterable[float]]]): Upper bounds of the problem.
            a (Optional[float]): a parameter.
            b (Optional[float]): b parameter.
            c (Optional[float]): c parameter.

        See Also:
            :func:`niapy.problems.Problem.__init__`

        """
        super().__init__(dimension, lower, upper, *args, **kwargs)
        self.a = a
        self.b = b
        self.c = c

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\mathbf{x}) = -a\;\exp\left(-b \sqrt{\frac{1}{D}
                \sum_{i=1}^D x_i^2}\right) - \exp\left(\frac{1}{D}
                \sum_{i=1}^D \cos(c\;x_i)\right) + a + \exp(1)$'''

    def _evaluate(self, x):
        val1 = np.sum(np.square(x))
        val2 = np.sum(np.cos(self.c * x))

        temp1 = -self.b * np.sqrt(val1 / self.dimension)
        temp2 = val2 / self.dimension

        return -self.a * np.exp(temp1) - np.exp(temp2) + self.a + np.exp(1)
