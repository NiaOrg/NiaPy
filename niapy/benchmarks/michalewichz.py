# encoding=utf8

"""Implementations of Michalewichz's function."""

import numpy as np
from niapy.benchmarks.benchmark import Benchmark

__all__ = ['Michalewichz']


class Michalewichz(Benchmark):
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

    Name = ['Michalewichz']

    def __init__(self, lower=0.0, upper=np.pi, m=10):
        r"""Initialize of Michalewichz benchmark.

        Args:
            lower (Optional[float]): Lower bound of problem.
            upper (Optional[float]): Upper bound of problem.
            m (float): Steepness of valleys and ridges. Recommended value is 10.

        See Also:
            :func:`niapy.benchmarks.Benchmark.__init__`

        """
        super().__init__(lower, upper)
        self.m = m

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m}$'''

    def function(self):
        r"""Return benchmark evaluation function.

        Returns:
            Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function.

        """
        def evaluate(dimension, x):
            r"""Fitness function.

            Args:
                dimension (int): Dimensionality of the problem
                x (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

            Returns:
                float: Fitness value for the solution.

            """
            v = 0.0
            for i in range(dimension):
                v += np.sin(x[i]) * np.sin(((i + 1) * x[i] ** 2) / np.pi) ** (2 * self.m)
            return -v

        return evaluate

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
