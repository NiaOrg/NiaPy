# encoding=utf8

"""Styblinski Tang problem."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['StyblinskiTang']


class StyblinskiTang(Problem):
    r"""Implementation of Styblinski-Tang functions.

    Date: 2018

    Authors: Lucija Brezočnik

    License: MIT

    Function: **Styblinski-Tang function**

        :math:`f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^D \left(
        x_i^4 - 16x_i^2 + 5x_i \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-5, 5]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = -78.332`, at :math:`x^* = (-2.903534,...,-2.903534)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^D \left(
            x_i^4 - 16x_i^2 + 5x_i \right) $

        Equation:
            \begin{equation}f(\mathbf{x}) =
            \frac{1}{2} \sum_{i=1}^D \left( x_i^4 - 16x_i^2 + 5x_i \right) \end{equation}

        Domain:
            $-5 \leq x_i \leq 5$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-5.0, upper=5.0, *args, **kwargs):
        r"""Initialize Styblinski Tang problem..

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
        return r'''$f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^D \left(
                x_i^4 - 16x_i^2 + 5x_i \right) $'''

    def _evaluate(self, x):
        return 0.5 * np.sum(x ** 4 - 16.0 * x ** 2 + 5.0 * x)
