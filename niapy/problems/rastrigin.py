# encoding=utf8

"""Implementation of Rastrigin function."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Rastrigin']


class Rastrigin(Problem):
    r"""Implementation of Rastrigin problem.

    Date: 2018

    Authors: Lucija Brezočnik and Iztok Fister Jr.

    License: MIT

    Function: **Rastrigin function**

        :math:`f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-5.12, 5.12]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)$

        Equation:
            \begin{equation} f(\mathbf{x}) =
            10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)
            \end{equation}

        Domain:
            $-5.12 \leq x_i \leq 5.12$

    Reference:
        https://www.sfu.ca/~ssurjano/rastr.html

    """

    def __init__(self, dimension=4, lower=-5.12, upper=5.12, *args, **kwargs):
        r"""Initialize Rastrigin problem..

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
        return r'''$f(\mathbf{x}) = 10D + \sum_{i=1}^D \left(x_i^2 -10\cos(2\pi x_i)\right)$'''

    def _evaluate(self, x):
        return 10.0 * self.dimension + np.sum(x * x - 10.0 * np.cos(2 * np.pi * x))
