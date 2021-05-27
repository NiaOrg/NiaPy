# encoding=utf8

"""Implementations of Powell function."""
import numpy as np

from niapy.problems.problem import Problem

__all__ = ['Powell']


class Powell(Problem):
    r"""Implementations of Powell functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
    **Powell Function**

        :math:`f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-4, 5]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (0, \cdots, 0)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right)$

        Equation:
            \begin{equation} f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right) \end{equation}

        Domain:
            $-4 \leq x_i \leq 5$

    Reference:
        https://www.sfu.ca/~ssurjano/powell.html

    """

    def __init__(self, dimension=4, lower=-4.0, upper=5.0, *args, **kwargs):
        r"""Initialize Powell problem..

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
        return r'''$f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right)$'''

    def _evaluate(self, x):
        x1 = x[range(1, self.dimension - 3, 4)]
        x2 = x[range(2, self.dimension - 2, 4)]
        x3 = x[range(3, self.dimension - 1, 4)]
        x4 = x[range(4, self.dimension, 4)]

        term1 = (x1 + 10 * x2) ** 2.0
        term2 = 5 * (x3 - x4) ** 2.0
        term3 = (x2 - 2 * x3) ** 4.0
        term4 = 10 * (x1 - x4) ** 4.0
        return np.sum(term1 + term2 + term3 + term4)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
