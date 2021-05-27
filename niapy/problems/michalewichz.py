# encoding=utf8

"""Implementations of Michalewichz's function."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Michalewichz']


class Michalewichz(Problem):
    r"""Implementations of Michalewichz's functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
    **High Conditioned Elliptic Function**

        :math:`f(\textbf{x}) = \sum_{i=1}^D \left( 10^6 \right)^{ \frac{i - 1}{D - 1} } x_i^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [0, \pi]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        at :math:`d = 2` :math:`f(\textbf{x}^*) = -1.8013` at :math:`\textbf{x}^* = (2.20, 1.57)`
        at :math:`d = 5` :math:`f(\textbf{x}^*) = -4.687658`
        at :math:`d = 10` :math:`f(\textbf{x}^*) = -9.66015`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m}$

        Equation:
            \begin{equation} f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m} \end{equation}

        Domain:
            $0 \leq x_i \leq \pi$

    Reference URL:
        https://www.sfu.ca/~ssurjano/michal.html

    """

    def __init__(self, dimension=4, lower=0.0, upper=np.pi, m=10, *args, **kwargs):
        r"""Initialize Michalewichz problem..

        Args:
            dimension (Optional[int]): Dimension of the problem.
            lower (Optional[Union[float, Iterable[float]]]): Lower bounds of the problem.
            upper (Optional[Union[float, Iterable[float]]]): Upper bounds of the problem.
            m (float): Steepness of valleys and ridges. Recommended value is 10.

        See Also:
            :func:`niapy.problems.Problem.__init__`

        """
        super().__init__(dimension, lower, upper, *args, **kwargs)
        self.m = m

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m}$'''

    def _evaluate(self, x):
        return -np.sum(np.sin(x) * np.sin((np.arange(1, self.dimension + 1) * x ** 2.0) / np.pi) ** (2.0 * self.m))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
