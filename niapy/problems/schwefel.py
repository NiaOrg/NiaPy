# encoding=utf8

"""Implementations of Schwefel's functions."""

import numpy as np
from niapy.problems.problem import Problem

__all__ = ['Schwefel', 'Schwefel221', 'Schwefel222', 'ModifiedSchwefel']


class Schwefel(Problem):
    r"""Implementation of Schwefel function.

    Date: 2018

    Author: Lucija Brezočnik

    License: MIT

    Function: **Schwefel function**

        :math:`f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i \sin(\sqrt{\lvert x_i \rvert})`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-500, 500]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i \sin(\sqrt{\lvert x_i \rvert})$

        Equation:
            \begin{equation} f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i
            \sin(\sqrt{\lvert x_i \rvert}) \end{equation}

        Domain:
            $-500 \leq x_i \leq 500$

    Reference:
        https://www.sfu.ca/~ssurjano/schwef.html

    """

    def __init__(self, dimension=4, lower=-500.0, upper=500.0, *args, **kwargs):
        r"""Initialize Schwefel problem..

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
        return r'''$f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i \sin(\sqrt{\lvert x_i \rvert})$'''

    def _evaluate(self, x):
        return 418.9829 * self.dimension - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class Schwefel221(Problem):
    r"""Schwefel 2.21 function implementation.

    Date: 2018

    Author: Grega Vrbančič

    Licence: MIT

    Function: **Schwefel 2.21 function**

        :math:`f(\mathbf{x})=\max_{i=1,...,D}|x_i|`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x})=\max_{i=1,...,D} \lvert x_i \rvert$

        Equation:
            \begin{equation}f(\mathbf{x}) = \max_{i=1,...,D} \lvert x_i \rvert \end{equation}

        Domain:
            $-100 \leq x_i \leq 100$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-100.0, upper=100.0, *args, **kwargs):
        r"""Initialize Schwefel221 problem..

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
        return r'''$f(\mathbf{x})=\max_{i=1,...,D} \lvert x_i \rvert$'''

    def _evaluate(self, x):
        return np.amax(np.abs(x))


class Schwefel222(Problem):
    r"""Schwefel 2.22 function implementation.

    Date: 2018

    Author: Grega Vrbančič

    Licence: MIT

    Function: **Schwefel 2.22 function**

        :math:`f(\mathbf{x})=\sum_{i=1}^{D} \lvert x_i \rvert +\prod_{i=1}^{D} \lvert x_i \rvert`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\mathbf{x})=\sum_{i=1}^{D} \lvert x_i \rvert +\prod_{i=1}^{D} \lvert x_i \rvert$

        Equation:
            \begin{equation}f(\mathbf{x}) = \sum_{i=1}^{D} \lvert x_i \rvert + \prod_{i=1}^{D} \lvert x_i \rvert \end{equation}

        Domain:
            $-100 \leq x_i \leq 100$

    Reference paper:
        Jamil, M., and Yang, X. S. (2013).
        A literature survey of benchmark functions for global optimisation problems.
        International Journal of Mathematical Modelling and Numerical Optimisation,
        4(2), 150-194.

    """

    def __init__(self, dimension=4, lower=-100.0, upper=100.0, *args, **kwargs):
        r"""Initialize Schwefel222 problem..

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
        return r'''$f(\mathbf{x})=\sum_{i=1}^{D} \lvert x_i \rvert +\prod_{i=1}^{D} \lvert x_i \rvert$'''

    def _evaluate(self, x):
        return np.sum(np.abs(x)) + np.product(np.abs(x))


class ModifiedSchwefel(Problem):
    r"""Implementations of Modified Schwefel functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
    **Modified Schwefel Function**

        :math:`f(\textbf{x}) = 418.9829 \cdot D - \sum_{i=1}^D h(x_i) \\ h(x) = g(x + 420.9687462275036)  \\ g(z) = \begin{cases} z \sin \left( \lvert z \rvert^{\frac{1}{2}} \right) &\quad \lvert z \rvert \leq 500 \\ \left( 500 - \mod (z, 500) \right) \sin \left( \sqrt{\lvert 500 - \mod (z, 500) \rvert} \right) - \frac{ \left( z - 500 \right)^2 }{ 10000 D }  &\quad z > 500 \\ \left( \mod (\lvert z \rvert, 500) - 500 \right) \sin \left( \sqrt{\lvert \mod (\lvert z \rvert, 500) - 500 \rvert} \right) + \frac{ \left( z - 500 \right)^2 }{ 10000 D } &\quad z < -500\end{cases}`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = 418.9829 \cdot D - \sum_{i=1}^D h(x_i) \\ h(x) = g(x + 420.9687462275036)  \\ g(z) = \begin{cases} z \sin \left( \lvert z \rvert^{\frac{1}{2}} \right) &\quad \lvert z \rvert \leq 500 \\ \left( 500 - \mod (z, 500) \right) \sin \left( \sqrt{\lvert 500 - \mod (z, 500) \rvert} \right) - \frac{ \left( z - 500 \right)^2 }{ 10000 D }  &\quad z > 500 \\ \left( \mod (\lvert z \rvert, 500) - 500 \right) \sin \left( \sqrt{\lvert \mod (\lvert z \rvert, 500) - 500 \rvert} \right) + \frac{ \left( z - 500 \right)^2 }{ 10000 D } &\quad z < -500\end{cases}$

        Equation:
            \begin{equation} f(\textbf{x}) = 418.9829 \cdot D - \sum_{i=1}^D h(x_i) \\ h(x) = g(x + 420.9687462275036)  \\ g(z) = \begin{cases} z \sin \left( \lvert z \rvert^{\frac{1}{2}} \right) &\quad \lvert z \rvert \leq 500 \\ \left( 500 - \mod (z, 500) \right) \sin \left( \sqrt{\lvert 500 - \mod (z, 500) \rvert} \right) - \frac{ \left( z - 500 \right)^2 }{ 10000 D }  &\quad z > 500 \\ \left( \mod (\lvert z \rvert, 500) - 500 \right) \sin \left( \sqrt{\lvert \mod (\lvert z \rvert, 500) - 500 \rvert} \right) + \frac{ \left( z - 500 \right)^2 }{ 10000 D } &\quad z < -500\end{cases} \end{equation}

        Domain:
            $-100 \leq x_i \leq 100$

    Reference:
        http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf

    """

    def __init__(self, dimension=4, lower=-100.0, upper=100.0, *args, **kwargs):
        r"""Initialize Modified Schwefel problem..

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
        return r'''$f(\textbf{x}) = 418.9829 \cdot D - \sum_{i=1}^D h(x_i) \\ h(x) = g(x + 420.9687462275036)  \\ g(z) = \begin{cases} z \sin \left( \lvert z \rvert^{\frac{1}{2}} \right) &\quad \lvert z \rvert \leq 500 \\ \left( 500 - \mod (z, 500) \right) \sin \left( \sqrt{\lvert 500 - \mod (z, 500) \rvert} \right) - \frac{ \left( z - 500 \right)^2 }{ 10000 D }  &\quad z > 500 \\ \left( \mod (\lvert z \rvert, 500) - 500 \right) \sin \left( \sqrt{\lvert \mod (\lvert z \rvert, 500) - 500 \rvert} \right) + \frac{ \left( z - 500 \right)^2 }{ 10000 D } &\quad z < -500\end{cases}$'''

    def _evaluate(self, x):
        xx = x + 420.9687462275036
        conditions = [x > 500.0, x < -500.0]
        xx_mod = np.fmod(xx, 500.0)
        choices = [(500.0 - xx_mod) * np.sin(np.sqrt(np.abs(500.0 - xx_mod))) - (xx - 500.0) ** 2 / (10000 * self.dimension),
                   (xx_mod - 500.0) * np.sin(np.sqrt(np.abs(xx_mod - 500.0))) + (xx - 500.0) ** 2 / (10000 * self.dimension)]
        default = xx * np.sin(np.sqrt(np.abs(xx)))
        val = np.sum(np.select(conditions, choices, default=default))
        return 418.9829 * self.dimension - val

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
