# encoding=utf8

"""Implementations of Schwefel's functions."""

from math import sin, fmod, fabs, sqrt
from niapy.benchmarks.benchmark import Benchmark

__all__ = ['Schwefel', 'Schwefel221', 'Schwefel222', 'ModifiedSchwefel']


class Schwefel(Benchmark):
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

    Name = ['Schwefel']

    def __init__(self, lower=-500.0, upper=500.0):
        r"""Initialize of Schwefel benchmark.

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
        return r'''$f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i \sin(\sqrt{\lvert x_i \rvert})$'''

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
            val = 0.0
            for i in range(dimension):
                val += (x[i] * sin(sqrt(abs(x[i]))))
            return 418.9829 * dimension - val

        return evaluate


class Schwefel221(Benchmark):
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

    Name = ['Schwefel221']

    def __init__(self, lower=-100.0, upper=100.0):
        r"""Initialize of Schwefel221 benchmark.

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
        return r'''$f(\mathbf{x})=\max_{i=1,...,D} \lvert x_i \rvert$'''

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
                Callable[[int, numpy.ndarray, dict], float]: Fitness function.

            """
            maximum = 0.0
            for i in range(dimension):
                if abs(x[i]) > maximum:
                    maximum = abs(x[i])
            return maximum

        return evaluate


class Schwefel222(Benchmark):
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

    Name = ['Schwefel222']

    def __init__(self, lower=-100.0, upper=100.0):
        r"""Initialize of Schwefel222 benchmark.

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
        return r'''$f(\mathbf{x})=\sum_{i=1}^{D} \lvert x_i \rvert +\prod_{i=1}^{D} \lvert x_i \rvert$'''

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
            part1 = 0.0
            part2 = 1.0
            for i in range(dimension):
                part1 += abs(x[i])
                part2 *= abs(x[i])
            return part1 + part2

        return evaluate


class ModifiedSchwefel(Benchmark):
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

    Name = ['ModifiedSchwefel']

    def __init__(self, lower=-100.0, upper=100.0):
        r"""Initialize of Modified Schwefel benchmark.

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
        return r'''$f(\textbf{x}) = 418.9829 \cdot D - \sum_{i=1}^D h(x_i) \\ h(x) = g(x + 420.9687462275036)  \\ g(z) = \begin{cases} z \sin \left( \lvert z \rvert^{\frac{1}{2}} \right) &\quad \lvert z \rvert \leq 500 \\ \left( 500 - \mod (z, 500) \right) \sin \left( \sqrt{\lvert 500 - \mod (z, 500) \rvert} \right) - \frac{ \left( z - 500 \right)^2 }{ 10000 D }  &\quad z > 500 \\ \left( \mod (\lvert z \rvert, 500) - 500 \right) \sin \left( \sqrt{\lvert \mod (\lvert z \rvert, 500) - 500 \rvert} \right) + \frac{ \left( z - 500 \right)^2 }{ 10000 D } &\quad z < -500\end{cases}$'''

    def function(self):
        r"""Return benchmark evaluation function.

        Returns:
            Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function.

        """
        def g(z, dimension):
            if z > 500:
                return (500 - fmod(z, 500)) * sin(sqrt(fabs(500 - fmod(z, 500)))) - (z - 500) ** 2 / (10000 * dimension)
            elif z < -500:
                return (fmod(z, 500) - 500) * sin(sqrt(fabs(fmod(z, 500) - 500))) + (z - 500) ** 2 / (10000 * dimension)
            return z * sin(fabs(z) ** (1 / 2))

        def h(x, dimension):
            return g(x + 420.9687462275036, dimension)

        def f(dimension, x):
            r"""Fitness function.

            Args:
                dimension (int): Dimensionality of the problem.
                x (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

            Returns:
                float: Fitness value for the solution.

            """
            val = 0.0
            for i in range(dimension):
                val += h(x[i], dimension)
            return 418.9829 * dimension - val

        return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
