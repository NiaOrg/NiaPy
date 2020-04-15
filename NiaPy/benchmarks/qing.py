# encoding=utf8


"""Implementaiton of Qing funcion."""

import math
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Qing']

class Qing(Benchmark):
    r"""Implementation of Qing function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Qing function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \left(x_i^2 - i\right)^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-500, 500]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (\pm √i))`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D \left (x_i^2 - i\right)^2$

        Equation:
                \begin{equation} f(\mathbf{x}) =
                \sum_{i=1}^D \left{(x_i^2 - i\right)}^2 \end{equation}

        Domain:
                $-500 \leq x_i \leq 500$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    Attributes:
        Name (List[str]): Names of the benchmark.

    See Also:
        * :class:`NiaPy.benchmarks.Benchmark`
    """
    Name = ['Qing', 'qing']

    def __init__(self, Lower=-500.0, Upper=500.0, **kwargs):
        r"""Initialize of Qing benchmark.

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
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D \left (x_i^2 - i\right)^2$'''

    def function(self):
        r"""Return benchmark evaluation function.

        Returns:
            Callable[[int, Union[int, float, list, numpy.ndarray], Dict[str, Any]], float]: Fitness function
        """
        def evaluate(D, sol, **kwargs):
            r"""Fitness function.

            Args:
                D (int): Dimensionality of the problem
                sol (Union[int, float, list, numpy.ndarray]): Solution to check.
                kwargs (Dict[str, Any]): Additional arguments.

            Returns:
                float: Fitness value for the solution.
            """
            val = 0.0

            for i in range(D):
                val += math.pow(math.pow(sol[i], 2) - i, 2)

            return val

        return evaluate
