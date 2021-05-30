# encoding=utf8


"""Implementation of Qing funcion."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Qing']


class Qing(Problem):
    r"""Implementation of Qing function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Qing function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \left(x_i^2 - i\right)^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-500, 500]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (\pm √i))`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = \sum_{i=1}^D \left (x_i^2 - i\right)^2$

        Equation:
            \begin{equation} f(\mathbf{x}) =
            \sum_{i=1}^D \left{(x_i^2 - i\right)}^2 \end{equation}

        Domain:
            $-500 \leq x_i \leq 500$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-500.0, upper=500.0, *args, **kwargs):
        r"""Initialize Qing problem..

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
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D \left (x_i^2 - i\right)^2$'''

    def _evaluate(self, x):
        return np.sum(np.power(x ** 2.0 - np.arange(1, self.dimension + 1), 2.0))
