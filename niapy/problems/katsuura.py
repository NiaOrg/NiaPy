# encoding=utf8

"""Implementations of Katsuura functions."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Katsuura']


class Katsuura(Problem):
    r"""Implementations of Katsuura functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
        **Katsuura Function**

        :math:`f(\textbf{x}) = \prod_{i=1}^D \left( 1 + i \sum_{j=1}^{32} \frac{\lvert 2^j x_i - round\left(2^j x_i \right) \rvert}{2^j} \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 1`, at :math:`x_i^* = 0`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = \prod_{i=1}^D \left( 1 + i \sum_{j=1}^{32} \frac{\lvert 2^j x_i - round\left(2^j x_i \right) \rvert}{2^j} \right)$
        Equation:
            \begin{equation} f(\textbf{x}) = \prod_{i=1}^D \left( 1 + i \sum_{j=1}^{32} \frac{\lvert 2^j x_i - round\left(2^j x_i \right) \rvert}{2^j} \right)\end{equation}

        Domain:
            $-100 \leq x_i \leq 100$

    Reference:
        Adorio, E. P., & Diliman, U. P. MVF - Multivariate Test Functions Library in C for Unconstrained Global Optimization (2005). 

    """

    def __init__(self, dimension=5, lower=-100.0, upper=100.0, *args, **kwargs):
        r"""Initialize Katsuura problem..

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
        return r'''$f(\textbf{x}) = \prod_{i=1}^D \left( 1 + i \sum_{j=1}^{32} \frac{| 2^j x_i - round\left(2^j x_i \right) |}{2^j} \right)$'''

    def _evaluate(self, x):
        k = np.atleast_2d(np.arange(1, 33)).T
        i = np.arange(1, self.dimension + 1)
        inner = np.round(2 ** k * x) * (2.0 ** (-k))
        return np.prod(np.sum(inner, axis=0) * i + 1)
