# encoding=utf8

"""Whitley function."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Whitley']


class Whitley(Problem):
    r"""Implementation of Whitley function.

    Date: 2018

    Authors: Grega Vrbančič and Lucija Brezočnik

    License: MIT

    Function: **Whitley function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \sum_{j=1}^D
        \left(\frac{(100(x_i^2-x_j)^2 + (1-x_j)^2)^2}{4000} -
        \cos(100(x_i^2-x_j)^2 + (1-x_j)^2)+1\right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-10.24, 10.24]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (1,...,1)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) =
            \sum_{i=1}^D \sum_{j=1}^D \left(\frac{(100(x_i^2-x_j)^2 +
            (1-x_j)^2)^2}{4000} - \cos(100(x_i^2-x_j)^2 + (1-x_j)^2)+1\right)$

        Equation:
            \begin{equation}f(\mathbf{x}) =
            \sum_{i=1}^D \sum_{j=1}^D \left(\frac{(100(x_i^2-x_j)^2 +
            (1-x_j)^2)^2}{4000} - \cos(100(x_i^2-x_j)^2 +
            (1-x_j)^2)+1\right) \end{equation}

        Domain:
            $-10.24 \leq x_i \leq 10.24$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-10.24, upper=10.24, *args, **kwargs):
        r"""Initialize Whitley problem..

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
        return r"""$f(\mathbf{x}) =
                \sum_{i=1}^D \sum_{j=1}^D \left(\frac{(100(x_i^2-x_j)^2 +
                (1-x_j)^2)^2}{4000} - \cos(100(x_i^2-x_j)^2 + (1-x_j)^2)+1\right)$"""

    def _evaluate(self, x):
        xi = np.tile(x, (self.dimension, 1)).T
        xj = np.tile(x, (self.dimension, 1))
        tmp = 100.0 * (xi ** 2 - xj) ** 2 + (1 - xj) ** 2
        return np.sum((tmp ** 2) / 4000.0 - np.cos(tmp) + 1.0)
