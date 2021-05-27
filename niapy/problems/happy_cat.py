# encoding=utf8

"""Implementation of Happy Cat function."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['HappyCat']


class HappyCat(Problem):
    r"""Implementation of Happy cat function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Happy cat function**

        :math:`f(\mathbf{x}) = {\left |\sum_{i = 1}^D {x_i}^2 - D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 + \sum_{i = 1}^D x_i) / D + 0.5`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (-1,...,-1)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = {\left|\sum_{i = 1}^D {x_i}^2 -
            D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 +
            \sum_{i = 1}^D x_i) / D + 0.5$

        Equation:
            \begin{equation} f(\mathbf{x}) = {\left| \sum_{i = 1}^D {x_i}^2 -
            D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 +
            \sum_{i = 1}^D x_i) / D + 0.5 \end{equation}

        Domain:
            $-100 \leq x_i \leq 100$

    Reference: http://bee22.com/manual/tf_images/Liang%20CEC2014.pdf &
    Beyer, H. G., & Finck, S. (2012). HappyCat - A Simple Function Class Where Well-Known Direct Search Algorithms Do Fail.
    In International Conference on Parallel Problem Solving from Nature (pp. 367-376). Springer, Berlin, Heidelberg.

    """

    def __init__(self, dimension=4, lower=-100.0, upper=100.0, alpha=0.25, *args, **kwargs):
        r"""Initialize Happy cat problem..

        Args:
            dimension (Optional[int]): Dimension of the problem.
            lower (Optional[Union[float, Iterable[float]]]): Lower bounds of the problem.
            upper (Optional[Union[float, Iterable[float]]]): Upper bounds of the problem.

        See Also:
            :func:`niapy.problems.Problem.__init__`

        """
        super().__init__(dimension, lower, upper, *args, **kwargs)
        self.alpha = alpha

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\mathbf{x}) = {\left|\sum_{i = 1}^D {x_i}^2 -
                D \right|}^{1/4} + (0.5 \sum_{i = 1}^D {x_i}^2 +
                \sum_{i = 1}^D x_i) / D + 0.5$'''

    def _evaluate(self, x):
        val1 = np.sum(np.abs(x * x - self.dimension) ** self.alpha)
        val2 = np.sum((0.5 * x * x + x) / self.dimension)
        return val1 + val2 + 0.5
