# encoding=utf8

"""Implementations of Dixon Price function."""
import numpy as np

from niapy.problems.problem import Problem

__all__ = ['DixonPrice']


class DixonPrice(Problem):
    r"""Implementations of Dixon Price function.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
    **Dixon Price Function**

        :math:`f(\textbf{x}) = (x_1 - 1)^2 + \sum_{i = 2}^D i (2x_i^2 - x_{i - 1})^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (2^{-\frac{2^1 - 2}{2^1}}, \cdots , 2^{-\frac{2^i - 2}{2^i}} , \cdots , 2^{-\frac{2^D - 2}{2^D}})`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = (x_1 - 1)^2 + \sum_{i = 2}^D i (2x_i^2 - x_{i - 1})^2$

        Equation:
            \begin{equation} f(\textbf{x}) = (x_1 - 1)^2 + \sum_{i = 2}^D i (2x_i^2 - x_{i - 1})^2 \end{equation}

        Domain:
            $-10 \leq x_i \leq 10$

    Reference:
        https://www.sfu.ca/~ssurjano/dixonpr.html

    """

    def __init__(self, dimension=4, lower=-10.0, upper=10.0, *args, **kwargs):
        r"""Initialize Dixon Price problem..

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
        return r'''$f(\textbf{x}) = (x_1 - 1)^2 + \sum_{i = 2}^D i (2x_i^2 - x_{i - 1})^2$'''

    def _evaluate(self, x):
        indices = np.arange(2, self.dimension)
        val = np.sum(indices * (2 * x[2:] ** 2 - x[1:self.dimension - 1]) ** 2)
        return (x[0] - 1) ** 2 + val

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
