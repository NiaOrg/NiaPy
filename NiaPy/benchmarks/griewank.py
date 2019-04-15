# encoding=utf8

"""Implementations of Griewank benchmarks."""

from math import sqrt, cos
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ["Griewank", "ExpandedGriewankPlusRosenbrock"]


class Griewank(Benchmark):
    r"""Implementation of Griewank function.

    Date: 2018

    Authors: Iztok Fister Jr. and Lucija Brezočnik

    License: MIT

    Function: **Griewank function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D \frac{x_i^2}{4000} - \prod_{i=1}^D \cos(\frac{x_i}{\sqrt{i}}) + 1`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D \frac{x_i^2}{4000} -
                \prod_{i=1}^D \cos(\frac{x_i}{\sqrt{i}}) + 1$

        Equation:
                \begin{equation} f(\mathbf{x}) = \sum_{i=1}^D \frac{x_i^2}{4000} -
                \prod_{i=1}^D \cos(\frac{x_i}{\sqrt{i}}) + 1 \end{equation}

        Domain:
                $-100 \leq x_i \leq 100$

    Reference paper:
    Jamil, M., and Yang, X. S. (2013).
    A literature survey of benchmark functions for global optimisation problems.
    International Journal of Mathematical Modelling and Numerical Optimisation,
    4(2), 150-194.

    """

    Name = ["Griewank"]

    def __init__(self, Lower=-100.0, Upper=100.0):
        """Initialize Griewank benchmark.

        Args:
            Lower (Optional[float]): Lower bound of problem.
            Upper (Optional[float]): Upper bound of problem.

        See Also:
            :func:`NiaPy.benchmarks.Benchmark.__init__`

        """

        Benchmark.__init__(self, Lower, Upper)

    @staticmethod
    def latex_code():
        """Return the latex code of the problem.

        Returns:
            [str] -- latex code.

        """

        return r'''$f(\mathbf{x}) = \sum_{i=1}^D \frac{x_i^2}{4000} -
                \prod_{i=1}^D \cos(\frac{x_i}{\sqrt{i}}) + 1$'''

    @classmethod
    def function(cls):
        """Return benchmark evaluation function.

        Returns:
            [fun] -- Evaluation function.

        """

        def evaluate(D, sol):

            val1, val2 = 0.0, 1.0

            for i in range(D):
                val1 += sol[i] ** 2 / 4000.0
                val2 *= cos(sol[i] / sqrt(i + 1))

            return val1 - val2 + 1.0

        return evaluate


class ExpandedGriewankPlusRosenbrock(Benchmark):
    r"""Implementation of Expanded Griewank's plus Rosenbrock benchmark.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function: **Expanded Griewank's plus Rosenbrock function**

        :math:`f(\textbf{x}) = h(g(x_D, x_1)) + \sum_{i=2}^D h(g(x_{i - 1}, x_i)) \\ g(x, y) = 100 (x^2 - y)^2 + (x - 1)^2 \\ h(z) = \frac{z^2}{4000} - \cos \left( \frac{z}{\sqrt{1}} \right) + 1`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

    LaTeX formats:
        Inline:
                $f(\textbf{x}) = h(g(x_D, x_1)) + \sum_{i=2}^D h(g(x_{i - 1}, x_i)) \\ g(x, y) = 100 (x^2 - y)^2 + (x - 1)^2 \\ h(z) = \frac{z^2}{4000} - \cos \left( \frac{z}{\sqrt{1}} \right) + 1$

        Equation:
                \begin{equation} f(\textbf{x}) = h(g(x_D, x_1)) + \sum_{i=2}^D h(g(x_{i - 1}, x_i)) \\ g(x, y) = 100 (x^2 - y)^2 + (x - 1)^2 \\ h(z) = \frac{z^2}{4000} - \cos \left( \frac{z}{\sqrt{1}} \right) + 1 \end{equation}

        Domain:
                $-100 \leq x_i \leq 100$

    Reference: http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf

    """

    Name = ["ExpandedGriewankPlusRosenbrock"]

    def __init__(self, Lower=-100.0, Upper=100.0):
        """Initialize Expanded Griewank Plus Rosenbrock benchmark.

        Args:
            Lower (Optional[float]): Lower bound of problem.
            Upper (Optional[float]): Upper bound of problem.

        See Also:
            :func:`NiaPy.benchmarks.Benchmark.__init__`

        """

        Benchmark.__init__(self, Lower, Upper)

    @staticmethod
    def latex_code():
        """Return the latex code of the problem.

        Returns:
            [str] -- latex code.

        """

        return r'''$f(\textbf{x}) = h(g(x_D, x_1)) + \sum_{i=2}^D h(g(x_{i - 1}, x_i)) \\ g(x, y) = 100 (x^2 - y)^2 + (x - 1)^2 \\ h(z) = \frac{z^2}{4000} - \cos \left( \frac{z}{\sqrt{1}} \right) + 1$'''

    @classmethod
    def function(cls):
        """Return benchmark evaluation function.

        Returns:
            [fun] -- Evaluation function.

        """

        def h(z):
            return z ** 2 / 4000 - cos(z / sqrt(1)) + 1

        def g(x, y):
            return 100 * (x ** 2 - y ** 2) ** 2 + (x - 1) ** 2

        def evaluate(D, sol):
            val = 0.0

            for i in range(1, D):
                val += h(g(sol[i - 1], sol[i]))

            return h(g(sol[D - 1], sol[0])) + val

        return evaluate
