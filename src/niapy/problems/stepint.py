# encoding=utf8

"""Step int problem."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Stepint']


class Stepint(Problem):
    r"""Implementation of Stepint functions.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Stepint function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D x_i^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-5.12, 5.12]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (-5.12,...,-5.12)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = \sum_{i=1}^D x_i^2$

        Equation:
            \begin{equation}f(\mathbf{x}) = \sum_{i=1}^D x_i^2 \end{equation}

        Domain:
            $0 \leq x_i \leq 10$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-5.12, upper=5.12, *args, **kwargs):
        r"""Initialize Stepint problem..

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
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D x_i^2$'''

    def _evaluate(self, x):
        return 25.0 + np.sum(np.floor(x))
