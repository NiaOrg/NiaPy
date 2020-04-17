# encoding=utf8

"""Implementation of Ackley benchmark."""

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Ackley']

class Ackley(Benchmark):
    r"""Implementation of Ackley function.

    Date: 2018

    Author: Lucija Brezočnik and Klemen Berkovič

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

    Attributes:
        Name (List[str]): Names of the benchmark.
        a (float): Objective function argument.
        b (float): Objective function argument.
        c (float): Objective function argument.

    See Also:
        * :class:`NiaPy.benchmarks.Benchmark`
    """
    Name = ['Ackley', 'ackley']
    a = 20         # Recommended variable value
    b = 0.2        # Recommended variable value
    c = 2 * np.pi  # Recommended variable value

    def __init__(self, Lower=-32.768, Upper=32.768, a=20, b=0.2, c=2 * np.pi, **kwargs):
        r"""Initialize of Ackley benchmark.

        Args:
            Lower (Optional[float]): Lower bound of problem.
            Upper (Optional[float]): Upper bound of problem.
            a (Optional[float]): Objective function argument.
            b (Optional[float]): Objective function argument.
            c (Optional[float]): Objective function argument.
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
        return r'''$f(\mathbf{x}) = -a\;\exp\left(-b \sqrt{\frac{1}{D}
                \sum_{i=1}^D x_i^2}\right) - \exp\left(\frac{1}{D}
                \sum_{i=1}^D cos(c\;x_i)\right) + a + \exp(1)$'''

    def function(self):
        r"""Return benchmark evaluation function.

        Returns:
            Callable[[int, Union[int, float, list, numpy.ndarray], Optional[float], Optional[float], Optional[float], Dict[str, Any]], float]: Fitness function
        """
        self_a, self_b, self_c = self.a, self.b, self.c
        def evaluate(D, sol, a=None, b=None, c=None, **kwargs):
            r"""Fitness function.

            Args:
                D (int): Dimensionality of the problem
                sol (Union[int, float, list, numpy.ndarray]): Solution to check.
                a (Optional[float]): Function argument.
                b (Optional[float]): Function argument.
                c (Optional[float]): Function argument.
                kwargs (Dict[str, Any]): Additional arguments.

            Returns:
                float: Fitness value for the solution.
            """
            a = a if a is not None else self_a
            b = b if b is not None else self_b
            c = c if c is not None else self_c

            val = 0.0
            val1 = 0.0
            val2 = 0.0

            for i in range(D):
                val1 += sol[i] ** 2
                val2 += np.cos(c * sol[i])

            temp1 = -b * np.sqrt(val1 / D)
            temp2 = val2 / D

            val = -a * np.exp(temp1) - np.exp(temp2) + a + np.exp(1)

            return val

        return evaluate
