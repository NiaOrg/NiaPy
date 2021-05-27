# encoding=utf8

"""Implementation of Salomon function."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Salomon']


class Salomon(Problem):
    r"""Implementation of Salomon function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Salomon function**

        :math:`f(\mathbf{x}) = 1 - \cos\left(2\pi\sqrt{\sum_{i=1}^D x_i^2}
        \right)+ 0.1 \sqrt{\sum_{i=1}^D x_i^2}`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = f(0, 0)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = 1 - \cos\left(2\pi\sqrt{\sum_{i=1}^D x_i^2}
            \right)+ 0.1 \sqrt{\sum_{i=1}^D x_i^2}$

        Equation:
            \begin{equation} f(\mathbf{x}) =
            1 - \cos\left(2\pi\sqrt{\sum_{i=1}^D x_i^2}
            \right)+ 0.1 \sqrt{\sum_{i=1}^D x_i^2} \end{equation}

        Domain:
            $-100 \leq x_i \leq 100$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-100.0, upper=100.0, *args, **kwargs):
        r"""Initialize Salomon problem..

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
        return r'''$f(\mathbf{x}) = 1 - \cos\left(2\pi\sqrt{\sum_{i=1}^D x_i^2}
                \right)+ 0.1 \sqrt{\sum_{i=1}^D x_i^2}$'''

    def _evaluate(self, x):
        val = np.sum(x ** 2.0)
        return 1.0 - np.cos(2.0 * np.pi * np.sqrt(val)) + 0.1 * val
