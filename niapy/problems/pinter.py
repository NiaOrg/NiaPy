# encoding=utf8

"""Implementation of Pinter function."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Pinter']


class Pinter(Problem):
    r"""Implementation of Pintér function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Pintér function**

        :math:`f(\mathbf{x}) =
        \sum_{i=1}^D ix_i^2 + \sum_{i=1}^D 20i \sin^2 A + \sum_{i=1}^D i
        \log_{10} (1 + iB^2);`
        :math:`A = (x_{i-1}\sin(x_i)+\sin(x_{i+1}))\quad \text{and} \quad`
        :math:`B = (x_{i-1}^2 - 2x_i + 3x_{i+1} - \cos(x_i) + 1)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) =
            \sum_{i=1}^D ix_i^2 + \sum_{i=1}^D 20i \sin^2 A + \sum_{i=1}^D i
            \log_{10} (1 + iB^2);
            A = (x_{i-1}\sin(x_i)+\sin(x_{i+1}))\quad \text{and} \quad
            B = (x_{i-1}^2 - 2x_i + 3x_{i+1} - \cos(x_i) + 1)$

        Equation:
            \begin{equation} f(\mathbf{x}) = \sum_{i=1}^D ix_i^2 +
            \sum_{i=1}^D 20i \sin^2 A + \sum_{i=1}^D i \log_{10} (1 + iB^2);
            A = (x_{i-1}\sin(x_i)+\sin(x_{i+1}))\quad \text{and} \quad
            B = (x_{i-1}^2 - 2x_i + 3x_{i+1} - \cos(x_i) + 1) \end{equation}

        Domain:
            $-10 \leq x_i \leq 10$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-10.0, upper=10.0, *args, **kwargs):
        r"""Initialize Pinter problem..

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
        return r''' $f(\mathbf{x}) =
                \sum_{i=1}^D ix_i^2 + \sum_{i=1}^D 20i \sin^2 A + \sum_{i=1}^D i
                \log_{10} (1 + iB^2);
                A = (x_{i-1}\sin(x_i)+\sin(x_{i+1}))\quad \text{and} \quad
                B = (x_{i-1}^2 - 2x_i + 3x_{i+1} - \cos(x_i) + 1)$'''

    def _evaluate(self, x):
        sub = np.roll(x, 1)
        add = np.roll(x, - 1)
        indices = np.arange(1, self.dimension + 1)

        a = (sub * np.sin(x) + np.sin(add))
        b = ((sub * sub) - 2.0 * x + 3.0 * add - np.cos(x) + 1.0)

        val1 = np.sum(indices * x * x)
        val2 = np.sum(20.0 * indices * np.power(np.sin(a), 2.0))
        val3 = np.sum(indices * np.log10(1.0 + indices * np.power(b, 2.0)))

        return val1 + val2 + val3
