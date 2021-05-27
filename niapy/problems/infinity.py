# encoding=utf8

"""Implementations of Infinity function."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Infinity']


class Infinity(Problem):
    r"""Implementations of Infinity function.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
    **Infinity Function**

        :math:`f(\textbf{x}) = \sum_{i = 1}^D x_i^6 \left( \sin \left( \frac{1}{x_i} \right) + 2 \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = \sum_{i = 1}^D x_i^6 \left( \sin \left( \frac{1}{x_i} \right) + 2 \right)$

        Equation:
            \begin{equation} f(\textbf{x}) = \sum_{i = 1}^D x_i^6 \left( \sin \left( \frac{1}{x_i} \right) + 2 \right) \end{equation}

        Domain:
            $-1 \leq x_i \leq 1$

    Reference:
        http://infinity77.net/global_optimization/test_functions_nd_I.html#go_benchmark.Infinity

    """

    def __init__(self, dimension=2, lower=-1.0, upper=1.0, *args, **kwargs):
        r"""Initialize Infinity problem..

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
        return r'''$f(\textbf{x}) = \sum_{i = 1}^D x_i^6 \left( \sin \left( \frac{1}{x_i} \right) + 2 \right)$'''

    def _evaluate(self, x):
        return np.sum(x ** 6.0 * (np.sin(1.0 / x) + 2.0))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
