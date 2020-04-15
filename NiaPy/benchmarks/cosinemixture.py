# encoding=utf8

"""Implementations of Cosine mixture functions."""

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['CosineMixture']

class CosineMixture(Benchmark):
    r"""Implementations of Cosine mixture function.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
        **Cosine Mixture Function**

        :math:`f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., D`.

        **Global maximu:**
        :math:`f(x^*) = -0.1 D`, at :math:`x^* = (0.0,...,0.0)`

    LaTeX formats:
        Inline:
                $f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2$

        Equation:
                \begin{equation} f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2 \end{equation}

        Domain:
                $-1 \leq x_i \leq 1$

    Attributes:
        Name (List[str]): Names of the benchmark.

    See Also:
        * :class:`NiaPy.benchmark.Benchmark`

    Reference:
        http://infinity77.net/global_optimization/test_functions_nd_C.html#go_benchmark.CosineMixture
    """
    Name = ['CosineMixture', 'cosinemixture']

    def __init__(self, Lower=-1.0, Upper=1.0, **kwargs):
        r"""Initialize of Cosine mixture benchmark.

        Args:
            Lower (Optional[float]): Lower bound of problem.
            Upper (Optional[float]): Upper bound of problem.
            kwargs (Dict[str, Any]): Additional arguments.

        See Also:
            :func:`NiaPy.benchmarks.Benchmark.__init__`
        """
        Benchmark.__init__(self, Lower, Upper)

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code
        """
        return r'''$f(\textbf{x}) = - 0.1 \sum_{i = 1}^D \cos (5 \pi x_i) - \sum_{i = 1}^D x_i^2$'''

    def function(self):
        r"""Return benchmark evaluation function.

        Returns:
            Callable[[int, Union[int, float, list, numpy.ndarray], Dict[str, Any]], float]: Fitness function
        """
        def f(D, X, **kwargs):
            r"""Fitness function.

            Args:
                D (int): Dimensionality of the problem
                X (Union[int, float, list, numpy.ndarray]): Solution to check.
                kwargs (Dict[str, Any]): Additional arguments.

            Returns:
                float: Fitness value for the solution.
            """
            v1, v2 = 0.0, 0.0
            for i in range(D): v1, v2 = v1 + np.cos(5 * np.pi * X[i]), v2 + X[i] ** 2
            return -0.1 * v1 - v2
        return f
