# encoding=utf8

"""Implementation of Ridge function."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Ridge']


class Ridge(Problem):
    r"""Implementation of Ridge function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Ridge function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-64, 64]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2 $

        Equation:
            \begin{equation} f(\mathbf{x}) =
            \sum_{i=1}^D (\sum_{j=1}^i x_j)^2 \end{equation}

        Domain:
            $-64 \leq x_i \leq 64$

    Reference:
        http://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/ridge.html

    """

    def __init__(self, dimension=4, lower=-64.0, upper=64.0, *args, **kwargs):
        r"""Initialize Ridge problem..

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
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D (\sum_{j=1}^i x_j)^2 $'''

    def _evaluate(self, x):
        inner = np.array([np.sum(x[:i]) for i in range(1, self.dimension + 1)])
        return np.sum(inner ** 2)
