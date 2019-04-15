# encoding=utf8

"""Implementations of Weierstrass functions."""

from math import pi, cos
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ["Weierstrass"]


class Weierstrass(Benchmark):
    r"""Implementations of Weierstrass functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function: **Weierstass Function**

        :math:`f(\textbf{x}) = \sum_{i=1}^D \left( \sum_{k=0}^{k_{max}} a^k \cos\left( 2 \pi b^k ( x_i + 0.5) \right) \right) - D \sum_{k=0}^{k_{max}} a^k \cos \left( 2 \pi b^k \cdot 0.5 \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.
        Default value of a = 0.5, b = 3 and k_max = 20.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

    LaTeX formats:
        Inline:
                $$f(\textbf{x}) = \sum_{i=1}^D \left( \sum_{k=0}^{k_{max}} a^k \cos\left( 2 \pi b^k ( x_i + 0.5) \right) \right) - D \sum_{k=0}^{k_{max}} a^k \cos \left( 2 \pi b^k \cdot 0.5 \right)

        Equation:
                \begin{equation} f(\textbf{x}) = \sum_{i=1}^D \left( \sum_{k=0}^{k_{max}} a^k \cos\left( 2 \pi b^k ( x_i + 0.5) \right) \right) - D \sum_{k=0}^{k_{max}} a^k \cos \left( 2 \pi b^k \cdot 0.5 \right) \end{equation}

        Domain:
                $-100 \leq x_i \leq 100$

    Reference:
        http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf
    """

    Name = ["Weierstrass"]

    def __init__(self, Lower=-100.0, Upper=100.0, a=0.5, b=3, k_max=20):
        """Initialize Weierstrass benchmark.

        Args:
            Lower (Optional[float]): Lower bound of problem.
            Upper (Optional[float]): Upper bound of problem.
            a (Optional[float]) -- a value. (default: {0.5})
            b (Optional[int]) -- b value. (default: {3})
            k_max (Optional[int]) -- k_max value. (default: {20})
        """

        Benchmark.__init__(self, Lower, Upper)
        Weierstrass.a = a
        Weierstrass.b = b
        Weierstrass.k_max = k_max

    @staticmethod
    def latex_code():
        """Return the latex code of the problem.

        Returns:
            [str] -- latex code.

        """

        return r"""$$f(\textbf{x}) = \sum_{i=1}^D \left( \sum_{k=0}^{k_{max}} a^k \cos\left( 2 \pi b^k ( x_i + 0.5) \right) \right) - D \sum_{k=0}^{k_{max}} a^k \cos \left( 2 \pi b^k \cdot 0.5 \right)"""

    @classmethod
    def function(cls):
        """Return benchmark evaluation function.

        Returns:
            [fun] -- Evaluation function.

        """

        def evaluate(D, sol, a=cls.a, b=cls.b, k_max=cls.k_max):

            val1 = 0.0

            for i in range(D):
                val = 0.0

                for k in range(k_max):
                    val += a ** k * cos(2 * pi * b ** k * (sol[i] + 0.5))

                val1 += val
            val2 = 0.0

            for k in range(k_max):
                val2 += a ** k * cos(2 * pi * b ** k * 0.5)

            return val1 - D * val2

        return evaluate
