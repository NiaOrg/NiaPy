# encoding=utf8

"""Implementations of Schaffer functions."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['SchafferN2', 'SchafferN4', 'ExpandedSchaffer']


class SchafferN2(Problem):
    r"""Implementations of Schaffer N. 2 functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
    **Schaffer N. 2 Function**
    :math:`f(\textbf{x}) = 0.5 + \frac{ \sin^2 \left( x_1^2 - x_2^2 \right) - 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 }`

    **Input domain:**
    The function can be defined on any input domain but it is usually
    evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

    **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = 0.5 + \frac{ \sin^2 \left( x_1^2 - x_2^2 \right) - 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 }$

        Equation:
            \begin{equation} f(\textbf{x}) = 0.5 + \frac{ \sin^2 \left( x_1^2 - x_2^2 \right) - 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 } \end{equation}

        Domain:
            $-100 \leq x_i \leq 100$

    Reference:
        http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf

    """

    def __init__(self, lower=-100.0, upper=100.0, *args, **kwargs):
        r"""Initialize SchafferN2 problem..

        Args:
            lower (Optional[Union[float, Iterable[float]]]): Lower bounds of the problem.
            upper (Optional[Union[float, Iterable[float]]]): Upper bounds of the problem.

        See Also:
            :func:`niapy.problems.Problem.__init__`

        """
        kwargs.pop('dimension', None)
        super().__init__(2, lower, upper, *args, **kwargs)

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\textbf{x}) = 0.5 + \frac{ \sin^2 \left( x_1^2 - x_2^2 \right) - 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 }$'''

    def _evaluate(self, x):
        return 0.5 + (np.sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2


class SchafferN4(Problem):
    r"""Implementations of Schaffer N. 2 functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
    **Schaffer N. 2 Function**
    :math:`f(\textbf{x}) = 0.5 + \frac{ \cos^2 \left( \sin \left( x_1^2 - x_2^2 \right) \right)- 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 }`

    **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

    **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = 0.5 + \frac{ \cos^2 \left( \sin \left( x_1^2 - x_2^2 \right) \right)- 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 }$

        Equation:
            \begin{equation} f(\textbf{x}) = 0.5 + \frac{ \cos^2 \left( \sin \left( x_1^2 - x_2^2 \right) \right)- 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 } \end{equation}

        Domain:
            $-100 \leq x_i \leq 100$

    Reference:
        http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf

    """

    def __init__(self, lower=-100.0, upper=100.0, *args, **kwargs):
        r"""Initialize SchafferN4 problem..

        Args:
            lower (Optional[Union[float, Iterable[float]]]): Lower bounds of the problem.
            upper (Optional[Union[float, Iterable[float]]]): Upper bounds of the problem.

        See Also:
            :func:`niapy.problems.Problem.__init__`

        """
        kwargs.pop('dimension', None)
        super().__init__(2, lower, upper, *args, **kwargs)

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code.

        """
        return r'''$f(\textbf{x}) = 0.5 + \frac{ \cos^2 \left( \sin \left( x_1^2 - x_2^2 \right) \right)- 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 }$'''

    def _evaluate(self, x):
        return 0.5 + (np.cos(np.sin(x[0] ** 2 - x[1] ** 2)) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2


class ExpandedSchaffer(Problem):
    r"""Implementations of Expanded Schaffer functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
        **Expanded Schaffer Function**
        :math:`f(\textbf{x}) = g(x_D, x_1) + \sum_{i=2}^D g(x_{i - 1}, x_i) \\ g(x, y) = 0.5 + \frac{\sin \left(\sqrt{x^2 + y^2} \right)^2 - 0.5}{\left( 1 + 0.001 (x^2 + y^2) \right)}^2`

    **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

    **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = g(x_D, x_1) + \sum_{i=2}^D g(x_{i - 1}, x_i) \\ g(x, y) = 0.5 + \frac{\sin \left(\sqrt{x^2 + y^2} \right)^2 - 0.5}{\left( 1 + 0.001 (x^2 + y^2) \right)}^2$

        Equation:
            \begin{equation} f(\textbf{x}) = g(x_D, x_1) + \sum_{i=2}^D g(x_{i - 1}, x_i) \\ g(x, y) = 0.5 + \frac{\sin \left(\sqrt{x^2 + y^2} \right)^2 - 0.5}{\left( 1 + 0.001 (x^2 + y^2) \right)}^2 \end{equation}

        Domain:
            $-100 \leq x_i \leq 100$

    Reference:
        http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf

    """

    def __init__(self, dimension=4, lower=-100.0, upper=100.0, *args, **kwargs):
        r"""Initialize Expanded Schaffer problem..

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
        return r'''$f(\textbf{x}) = g(x_D, x_1) + \sum_{i=2}^D g(x_{i - 1}, x_i) \\ g(x, y) = 0.5 + \frac{\sin \left(\sqrt{x^2 + y^2} \right)^2 - 0.5}{\left( 1 + 0.001 (x^2 + y^2) \right)}^2$'''

    def _evaluate(self, x):
        x_next = np.roll(x, -1)
        tmp = x ** 2 + x_next ** 2
        val = 0.5 + (np.sin(np.sqrt(tmp)) ** 2 - 0.5) / (1 + 0.001 * tmp) ** 2
        return np.sum(val)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
