# encoding=utf8
"""Implementations of Weierstrass functions."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Weierstrass']


class Weierstrass(Problem):
    r"""Implementations of Weierstrass functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
    **Weierstrass Function**

        :math:`f(\textbf{x}) = \sum_{i=1}^D \left( \sum_{k=0}^{k_{max}} a^k \cos\left( 2 \pi b^k ( x_i + 0.5) \right) \right) - D \sum_{k=0}^{k_{max}} a^k \cos \left( 2 \pi b^k \cdot 0.5 \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.
        Default value of a = 0.5, b = 3 and k_max = 20.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x_i^* = 0`

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

    def __init__(self, dimension=4, lower=-100.0, upper=100.0, a=0.5, b=3, k_max=20, *args, **kwargs):
        r"""Initialize Weierstrass problem..

        Args:
            dimension (Optional[int]): Dimension of the problem.
            lower (Optional[Union[float, Iterable[float]]]): Lower bounds of the problem.
            upper (Optional[Union[float, Iterable[float]]]): Upper bounds of the problem.
            a (Optional[float]): The a parameter.
            b (Optional[float]): The b parameter.
            k_max (Optional[int]): Number of elements of the series to compute.

        See Also:
            :func:`niapy.problems.Problem.__init__`

        """
        super().__init__(dimension, lower, upper, *args, **kwargs)
        self.a = a
        self.b = b
        self.k_max = k_max

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\textbf{x}) = \sum_{i=1}^D \left( \sum_{k=0}^{k_{max}} a^k \cos\left( 2 \pi b^k ( x_i + 0.5) \right) \right) - D \sum_{k=0}^{k_{max}} a^k \cos \left( 2 \pi b^k \cdot 0.5 \right)$'''

    def _evaluate(self, x):
        k = np.atleast_2d(np.arange(self.k_max + 1)).T
        t1 = self.a ** k * np.cos(2 * np.pi * self.b ** k * (x + 0.5))
        t2 = self.dimension * np.sum(self.a ** k.T * np.cos(np.pi * self.b ** k.T))

        return np.sum(np.sum(t1, axis=0)) - t2
