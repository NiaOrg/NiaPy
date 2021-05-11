# encoding=utf8

"""Implementation of Ackley benchmark."""

import numpy as np

from niapy.benchmarks.benchmark import Benchmark

__all__ = ['Ackley']


class Ackley(Benchmark):
    r"""Implementation of Ackley function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Ackley function**

        :math:`f(\mathbf{x}) = -a\;\exp\left(-b \sqrt{\frac{1}{D}\sum_{i=1}^D x_i^2}\right)
        - \exp\left(\frac{1}{D}\sum_{i=1}^D \cos(c\;x_i)\right) + a + \exp(1)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-32.768, 32.768]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(\textbf{x}^*) = 0`, at  :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x}) = -a\;\exp\left(-b \sqrt{\frac{1}{D}
            \sum_{i=1}^D x_i^2}\right) - \exp\left(\frac{1}{D}
            \sum_{i=1}^D cos(c\;x_i)\right) + a + \exp(1)$

        Equation:
            \begin{equation}f(\mathbf{x}) =
            -a\;\exp\left(-b \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2}\right) -
            \exp\left(\frac{1}{D} \sum_{i=1}^D \cos(c\;x_i)\right) +
            a + \exp(1) \end{equation}

        Domain:
            $-32.768 \leq x_i \leq 32.768$

    Reference:
        https://www.sfu.ca/~ssurjano/ackley.html

    """

    Name = ['Ackley']

    def __init__(self, lower=-32.768, upper=32.768):
        r"""Initialize of Ackley benchmark.

        Args:
            lower (Optional[float]): Lower bound of problem.
            upper (Optional[float]): Upper bound of problem.

        See Also:
            :func:`niapy.benchmarks.Benchmark.__init__`

        """
        super().__init__(lower, upper)

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\mathbf{x}) = -a\;\exp\left(-b \sqrt{\frac{1}{D}
                \sum_{i=1}^D x_i^2}\right) - \exp\left(\frac{1}{D}
                \sum_{i=1}^D \cos(c\;x_i)\right) + a + \exp(1)$'''

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
            a = 20  # Recommended variable value
            b = 0.2  # Recommended variable value
            c = 2 * np.pi  # Recommended variable value

            val1 = 0.0
            val2 = 0.0

            for i in range(dimension):
                val1 += x[i] ** 2
                val2 += np.cos(c * x[i])

            temp1 = -b * np.sqrt(val1 / dimension)
            temp2 = val2 / dimension

            val = -a * np.exp(temp1) - np.exp(temp2) + a + np.exp(1)

            return val

        return evaluate
