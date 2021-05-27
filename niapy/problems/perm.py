# encoding=utf8

"""Implementations of Perm function."""
import numpy as np

from niapy.problems.problem import Problem

__all__ = ['Perm']


class Perm(Problem):
    r"""Implementations of Perm functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
    **Perm Function**

        :math:`f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-D, D]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (1, \frac{1}{2}, \cdots , \frac{1}{i} , \cdots , \frac{1}{D})`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2$

        Equation:
            \begin{equation} f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2 \end{equation}

        Domain:
            $-D \leq x_i \leq D$

    Reference:
        https://www.sfu.ca/~ssurjano/perm0db.html


    """

    def __init__(self, dimension=4, beta=0.5, *args, **kwargs):
        r"""Initialize Perm problem.

        Args:
            dimension (Optional[int]): Dimension of the problem.
            beta (Optional[float]): Beta parameter.

        See Also:
            :func:`niapy.problems.Problem.__init__`

        """
        super().__init__(dimension, -dimension, dimension, *args, **kwargs)
        self.beta = beta

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2$'''

    def _evaluate(self, x):
        ii = np.arange(1, self.dimension + 1)
        jj = np.tile(ii, (self.dimension, 1))
        x_matrix = np.tile(x, (self.dimension, 1))
        inner = np.sum((jj + self.beta) * (np.power(x_matrix, ii) - np.power(1.0 / jj, ii)), axis=0)
        return np.sum(inner ** 2)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
