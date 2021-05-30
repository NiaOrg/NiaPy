# encoding=utf8

"""Implementation of Quintic funcion."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Quintic']


class Quintic(Problem):
    r"""Implementation of Quintic function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Quintic function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \left| x_i^5 - 3x_i^4 +
        4x_i^3 + 2x_i^2 - 10x_i - 4\right|`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = f(-1\; \text{or}\; 2)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D \left| x_i^5 - 3x_i^4 +
                4x_i^3 + 2x_i^2 - 10x_i - 4\right|$

        Equation:
                \begin{equation} f(\mathbf{x}) =
                \sum_{i=1}^D \left| x_i^5 - 3x_i^4 + 4x_i^3 + 2x_i^2 -
                10x_i - 4\right| \end{equation}

        Domain:
                $-10 \leq x_i \leq 10$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-10.0, upper=10.0, *args, **kwargs):
        r"""Initialize Quintic problem..

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
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D \left| x_i^5 - 3x_i^4 +
                4x_i^3 + 2x_i^2 - 10x_i - 4\right|$'''

    def _evaluate(self, x):
        return np.sum(np.abs(x ** 5 - 3.0 * x ** 4 + 4.0 * x ** 3 + 2.0 * x ** 2 - 10.0 * x - 4.0))
