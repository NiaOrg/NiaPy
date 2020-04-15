# encoding=utf8

"""Implementations of Correlation function."""

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark

__all__ = [
    'AutoCorrelation',
    'AutoCorrelationEnergy'
]

class AutoCorrelation(Benchmark):
    r"""Implementations of AutoCorrelation functions.

    Date:
        2020

    Author:
        Klemen Berkovič

    License:
        MIT

    Function:
        **AutoCorelation Function**
        :math:`f(\textbf{x}) = \sum_{i = 1}^{D - k} x_i * x_{i + k}`

        **Input domain:**
        The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-\inf, \inf]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (1, \cdots, 1)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = \sum_{i = 1}^{D - k} x_i * x_{i + k}$

        Equation:
            \begin{equation} f(\textbf{x}) = \sum_{i = 1}^{D - k} x_i * x_{i + k} \end{equation}

        Domain:
            $-\inf \leq x_i \leq \inf$

    Reference:
        TODO

    Attributes:
        Name (List[str]): Names of benchmark.

    See Also:
        * :class:`NiaPy.benchmarks.Benchmark`
    """
    Name = ['AutoCorrelation', 'autocorrelation']

    def __init__(self, Lower=-np.inf, Upper=np.inf):
        r"""Initialize of Levy benchmark.

        Args:
            Lower (Optional[float]): Lower bound of problem.
            Upper (Optional[float]): Upper bound of problem.

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
        return r'''$f(\textbf{x}) = \sum_{i = 1}^{D - k} x_i * x_{i + k}$'''

    def function(self):
        r"""Return benchmark evaluation function.

        Returns:
            Callable[[int, Union[int, float, list, numpy.ndarray], dict], float]: Fitness function
        """
        def f(d, x, k=None, **kwargs):
            r"""Fitness function.

            Args:
                d (int): Dimensionality of the problem
                x (Union[int, float, list, numpy.ndarray]): Solution to check.
                k (int): Shift
                kwargs (dict): Additional arguments.

            Returns:
                float: Fitness value for the solution.
            """
            k = k if k is not None else len(x)
            return np.sum(x[:d - k] * x[k:d])
        return f

class AutoCorrelationEnergy(AutoCorrelation):
    r"""Implementations of AutoCorrelation Energy functions.

    Date:
        2020

    Author:
        Klemen Berkovič

    License:
        MIT

    Function:
        **AutoCorelation Energy Function**
        :math:`f(\textbf{x}) = \sum_{i = 1}^{D - k} x_i * x_{i + k}`

        **Input domain:**
        The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-\inf, \inf]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (1, \cdots, 1)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = \sum_{i = 1}^{D - k} x_i * x_{i + k}$

        Equation:
            \begin{equation} f(\textbf{x}) = \sum_{i = 1}^{D - k} x_i * x_{i + k} \end{equation}

        Domain:
            $-\inf \leq x_i \leq \inf$

    Reference:
        TODO

    Attributes:
        Name (List[str]): Names of benchmark.

    See Also:
        * :class:`NiaPy.benchmarks.AutoCorrelation`
    """
    Name = ['AutoCorelationEnergey', 'autocorrelationenergy']

    def __init__(self, Lower=-np.inf, Upper=np.inf):
        r"""Initialize of Levy benchmark.

        Args:
            Lower (Optional[float]): Lower bound of problem.
            Upper (Optional[float]): Upper bound of problem.

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
        return r'''$f(\textbf{x}) = \sum_{i = 1}^{D - k} x_i * x_{i + k}$'''

    def function(self):
        r"""Return benchmark evaluation function.

        Returns:
            Callable[[int, Union[int, float, list, numpy.ndarray], dict], float]: Fitness function
        """
        c = AutoCorrelation.function(self)
        def f(d, x, **kwargs):
            r"""Fitness function.

            Args:
                d (int): Dimensionality of the problem
                x (Union[int, float, list[int, float], numpy.ndarray]): Solution to check.
                kwargs (dict): Additional arguments.

            Returns:
                float: Fitness value for the solution.
            """
            return np.sum([c(d, x, k) ** 2 for k in range(1, d + 1)])
        return f
