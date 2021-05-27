# encoding=utf8

"""Implementations of Step functions."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Step', 'Step2', 'Step3']


class Step(Problem):
    r"""Implementation of Step function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Step function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor \left |
        x_i \right | \rfloor \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor \left |
            x_i \right | \rfloor \right)$

        Equation:
            \begin{equation} f(\mathbf{x}) = \sum_{i=1}^D \left(
            \lfloor \left | x_i \right | \rfloor \right) \end{equation}

        Domain:
            $-100 \leq x_i \leq 100$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-100.0, upper=100.0, *args, **kwargs):
        r"""Initialize Step problem..

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
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor \left | x_i \right | \rfloor \right)$'''

    def _evaluate(self, x):
        return np.sum(np.floor(np.abs(x)))


class Step2(Problem):
    r"""Step2 function implementation.

    Date: 2018

    Author: Lucija Brezočnik

    Licence: MIT

    Function: **Step2 function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i + 0.5 \rfloor \right)^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (-0.5,...,-0.5)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i + 0.5 \rfloor \right)^2$

        Equation:
            \begin{equation}f(\mathbf{x}) = \sum_{i=1}^D \left(
            \lfloor x_i + 0.5 \rfloor \right)^2 \end{equation}

        Domain:
            $-100 \leq x_i \leq 100$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-100.0, upper=100.0, *args, **kwargs):
        r"""Initialize Step2 problem..

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
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i + 0.5 \rfloor \right)^2$'''

    def _evaluate(self, x):
        return np.sum(np.floor(x + 0.5) ** 2)


class Step3(Problem):
    r"""Step3 function implementation.

    Date: 2018

    Author: Lucija Brezočnik

    Licence: MIT

    Function: **Step3 function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i^2 \rfloor \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i^2 \rfloor \right)$

        Equation:
            \begin{equation}f(\mathbf{x}) = \sum_{i=1}^D \left(
            \lfloor x_i^2 \rfloor \right)\end{equation}

        Domain:
            $-100 \leq x_i \leq 100$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-100.0, upper=100.0, *args, **kwargs):
        r"""Initialize Step3 problem..

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
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D \left( \lfloor x_i^2 \rfloor \right)$'''

    def _evaluate(self, x):
        return np.sum(np.floor(x ** 2))
