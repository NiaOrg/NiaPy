# encoding=utf8

"""Sphere problems."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Sphere', 'Sphere2', 'Sphere3']


class Sphere(Problem):
    r"""Implementation of Sphere functions.

    Date: 2018

    Authors: Iztok Fister Jr.

    License: MIT

    Function: **Sphere function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D x_i^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [0, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = \sum_{i=1}^D x_i^2$

        Equation:
            \begin{equation}f(\mathbf{x}) = \sum_{i=1}^D x_i^2 \end{equation}

        Domain:
            $0 \leq x_i \leq 10$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-5.12, upper=5.12, *args, **kwargs):
        r"""Initialize Sphere problem..

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
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D x_i^2$'''

    def _evaluate(self, x):
        return np.sum(x ** 2)


class Sphere2(Problem):
    r"""Implementation of Sphere with different powers function.

    Date: 2018

    Authors: Klemen Berkovič

    License: MIT

    Function: **Sun of different powers function**

        :math:`f(\textbf{x}) = \sum_{i = 1}^D \lvert x_i \rvert^{i + 1}`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = \sum_{i = 1}^D \lvert x_i \rvert^{i + 1}$

        Equation:
            \begin{equation} f(\textbf{x}) = \sum_{i = 1}^D \lvert x_i \rvert^{i + 1} \end{equation}

        Domain:
            $-1 \leq x_i \leq 1$

    Reference URL:
        https://www.sfu.ca/~ssurjano/sumpow.html

    """

    def __init__(self, dimension=4, lower=-1.0, upper=1.0, *args, **kwargs):
        r"""Initialize Sphere2 problem..

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
        return r'''$f(\textbf{x}) = \sum_{i = 1}^D \lvert x_i \rvert^{i + 1}$'''

    def _evaluate(self, x):
        indices = np.arange(2, self.dimension + 2)
        return np.sum(np.power(np.abs(x), indices))


class Sphere3(Problem):
    r"""Implementation of rotated hyper-ellipsoid function.

    Date: 2018

    Authors: Klemen Berkovič

    License: MIT

    Function: **Sun of rotated hyper-ellipsoid function**

        :math:`f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-65.536, 65.536]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2$

        Equation:
            \begin{equation} f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2 \end{equation}

        Domain:
            $-65.536 \leq x_i \leq 65.536$

    Reference URL:
        https://www.sfu.ca/~ssurjano/rothyp.html

    """

    def __init__(self, dimension=4, lower=-65.536, upper=65.536, *args, **kwargs):
        r"""Initialize Sphere3 problem..

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
        return r'''$f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2$'''

    def _evaluate(self, x):
        x_matrix = np.tile(x, (self.dimension, 1))
        val = np.sum(np.tril(x_matrix) ** 2.0, axis=0)
        return np.sum(val)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
