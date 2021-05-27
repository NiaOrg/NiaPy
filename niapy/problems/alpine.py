# encoding=utf8

"""Implementations of Alpine functions."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Alpine1', 'Alpine2']


class Alpine1(Problem):
    r"""Implementation of Alpine1 function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Alpine1 function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^{D} \lvert x_i \sin(x_i)+0.1x_i \rvert`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-10, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^{D} \lvert x_i \sin(x_i)+0.1x_i \rvert$

        Equation:
                \begin{equation} f(\mathbf{x}) = \sum_{i=1}^{D} \lvert x_i \sin(x_i)+0.1x_i \rvert \end{equation}

        Domain:
                $-10 \leq x_i \leq 10$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-10.0, upper=10.0, *args, **kwargs):
        r"""Initialize Alpine1 problem.

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
            str: Latex code

        """
        return r'''$f(\mathbf{x}) = \sum_{i=1}^{D} \lvert x_i \sin(x_i)+0.1x_i \rvert$'''

    def _evaluate(self, x):
        return np.sum(np.abs(np.sin(x) + 0.1 * x))


class Alpine2(Problem):
    r"""Implementation of Alpine2 function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Alpine2 function**

        :math:`f(\mathbf{x}) = \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [0, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 2.808^D`, at :math:`x^* = (7.917,...,7.917)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i)$

        Equation:
            \begin{equation} f(\mathbf{x}) =
            \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i) \end{equation}

        Domain:
            $0 \leq x_i \leq 10$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=0.0, upper=10.0, *args, **kwargs):
        r"""Initialize Alpine2 problem..

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
        return r'''$f(\mathbf{x}) = \prod_{i=1}^{D} \sqrt{x_i} \sin(x_i)$'''

    def _evaluate(self, x):
        return np.product(np.sqrt(x) * np.sin(x))
