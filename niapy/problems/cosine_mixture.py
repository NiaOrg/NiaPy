# encoding=utf8

"""Implementations of Cosine mixture functions."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['CosineMixture']


class CosineMixture(Problem):
    r"""Implementations of Cosine mixture function.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
    **Cosine Mixture Function**

        :math:`f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., D`.

        **Global maximum:**
        :math:`f(x^*) = -0.1 D`, at :math:`x^* = (0.0,...,0.0)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2$

        Equation:
            \begin{equation} f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2 \end{equation}

        Domain:
            $-1 \leq x_i \leq 1$

    Reference:
        http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.CosineMixture

    """

    def __init__(self, dimension=4, lower=-1.0, upper=1.0, *args, **kwargs):
        r"""Initialize Cosine mixture problem..

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
        return r'''$f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2$'''

    def _evaluate(self, x):
        return -0.1 * np.sum(np.cos(5 * np.pi * x)) - np.sum(x ** 2)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
