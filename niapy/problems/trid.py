# encoding=utf8

"""Implementations of Trid function."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Trid']


class Trid(Problem):
    r"""Implementations of Trid functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
    **Trid Function**

        :math:`f(\textbf{x}) = \sum_{i = 1}^D \left( x_i - 1 \right)^2 - \sum_{i = 2}^D x_i x_{i - 1}`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-D^2, D^2]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(\textbf{x}^*) = \frac{-D(D + 4)(D - 1)}{6}` at :math:`\textbf{x}^* = (1 (D + 1 - 1), \cdots , i (D + 1 - i) , \cdots , D (D + 1 - D))`

    LaTeX formats:
        Inline:
                $f(\textbf{x}) = \sum_{i = 1}^D \left( x_i - 1 \right)^2 - \sum_{i = 2}^D x_i x_{i - 1}$

        Equation:
                \begin{equation} f(\textbf{x}) = \sum_{i = 1}^D \left( x_i - 1 \right)^2 - \sum_{i = 2}^D x_i x_{i - 1} \end{equation}

        Domain:
                $-D^2 \leq x_i \leq D^2$

    Reference:
        https://www.sfu.ca/~ssurjano/trid.html

    """

    def __init__(self, dimension=4, *args, **kwargs):
        r"""Initialize Trid problem..

        Args:
            dimension (Optional[int]): Dimension of the problem.

        See Also:
            :func:`niapy.problems.Problem.__init__`

        """
        kwargs.pop('lower', None)
        kwargs.pop('upper', None)
        super().__init__(dimension, -(dimension ** 2), dimension ** 2, *args, **kwargs)

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\textbf{x}) = \sum_{i = 1}^D \left( x_i - 1 \right)^2 - \sum_{i = 2}^D x_i x_{i - 1}$'''

    def _evaluate(self, x):
        sum1 = np.sum((x - 1) ** 2)
        sum2 = np.sum(x[1:] * x[:-1])
        return sum1 - sum2

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
