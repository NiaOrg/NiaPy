# encoding=utf8
"""Implementations of Zakharov function."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Zakharov']


class Zakharov(Problem):
    r"""Implementations of Zakharov functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
    **Zakharov Function**

        :math:`f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-5, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (0, \cdots, 0)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4$

        Equation:
            \begin{equation} f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4 \end{equation}

        Domain:
            $-5 \leq x_i \leq 10$

    Reference:
        https://www.sfu.ca/~ssurjano/zakharov.html

    """

    def __init__(self, dimension=4, lower=-5.0, upper=10.0, *args, **kwargs):
        r"""Initialize Zakharov problem..

        Args:
            dimension (Optional[int]): Dimension of the problem.
            lower (Optional[Union[float, Iterable[float]]]): Lower bounds of the problem.
            upper (Optional[Union[float, Iterable[float]]]): Upper bounds of the problem.

        See Also:
            :func:`niapy.problems.Problem.__init__`

        """
        super().__init__(dimension, lower, upper, *args, **kwargs)

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\textbf{x}) = \sum_{i = 1}^D x_i^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^2 + \left( \sum_{i = 1}^D 0.5 i x_i \right)^4$'''

    def _evaluate(self, x):
        sum1 = np.sum(x * x)
        sum2 = np.sum(0.5 * np.arange(1, self.dimension + 1) * x)
        return sum1 + sum2 ** 2 + sum2 ** 4

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
