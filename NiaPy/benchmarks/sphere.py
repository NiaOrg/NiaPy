# encoding=utf8

"""Sphere benchmarks."""

import numpy as np

from NiaPy.benchmarks.benchmark import Benchmark

__all__ = [
    'Sphere',
    'Sphere2',
    'Sphere3'
]

class Sphere(Benchmark):
    r"""Implementation of Sphere functions.

    Date: 2018

    Authors: Iztok Fister Jr.

    License: MIT

    Function:
        **Sphere function**

        :math:`f(\mathbf{x}) = \sum_{i=1}^D x_i^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [0, 10]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
                $f(\mathbf{x}) = \sum_{i=1}^D x_i^2$

        Equation:
                \begin{equation}f(\mathbf{x}) = \sum_{i=1}^D x_i^2 \end{equation}

        Domain:
                $0 \leq x_i \leq 10$

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
    Name = ['Sphere', 'sphere']

    def __init__(self, Lower=-5.12, Upper=5.12):
        r"""Initialize of Sphere benchmark.

        Args:
            Lower (Optional[float]): Lower bound of problem.
            Upper (Optional[float]): Upper bound of problem.

        See Also:
            * :func:`NiaPy.benchmarks.Benchmark.__init__`
        """
        Benchmark.__init__(self, Lower, Upper)

    @staticmethod
    def latex_code():
        r"""Return the latex code of the problem.

        Returns:
            str: Latex code
        """
        return r'''$f(\mathbf{x}) = \sum_{i=1}^D x_i^2$'''

    def function(self):
        r"""Return benchmark evaluation function.

        Returns:
            Callable[[int, numpy.ndarray, dict], float]: Fitness function
        """
        def evaluate(D, sol, **args):
            r"""Fitness function.

            Args:
                D (int): Dimensionality of the problem
                sol (numpy.ndarray): Solution to check.
                args (dict): Additional arguments.

            Returns:
                float: Fitness value for the solution.
            """
            val = 0.0
            for i in range(D): val += sol[i] ** 2
            return val
        return evaluate

class Sphere2(Benchmark):
    r"""Implementation of Sphere with different powers function.

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Function:
        **Sub of different powers function**
        :math:`f(\textbf{x}) = \sum_{i = 1}^D \lvert x_i \rvert^{i + 1}`

        **Input domain:**
        The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = \sum_{i = 1}^D \lvert x_i \rvert^{i + 1}$

        Equation:
            \begin{equation} f(\textbf{x}) = \sum_{i = 1}^D \lvert x_i \rvert^{i + 1} \end{equation}

        Domain:
            $-1 \leq x_i \leq 1$

    Reference URL:
        https://www.sfu.ca/~ssurjano/sumpow.html

    Attributes:
        Name (List[str]): Names of the benchmark.

    See Also:
        * :class:`NiaPy.benchmarks.Benchmark`
    """
    Name = ['Sphere2', 'sphere2']

    def __init__(self, Lower=-1., Upper=1., **kwargs):
        r"""Initialize of Sphere2 benchmark.

        Args:
            Lower (Optional[float]): Lower bound of problem.
            Upper (Optional[float]): Upper bound of problem.
            kwargs (dict): Additional arguments.

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
        return r'''$f(\textbf{x}) = \sum_{i = 1}^D \lvert x_i \rvert^{i + 1}$'''

    def function(self):
        r"""Return benchmark evaluation function.

        Returns:
            Callable[[int, Union[int, float, list, numpy.ndarray], dict], float]: Fitness function
        """
        def evaluate(D, sol, **kwargs):
            r"""Fitness function.

            Args:
                D (int): Dimensionality of the problem
                sol (Union[int, float, list, numpy.ndarray]): Solution to check.
                kwargs (dict): Additional arguments.

            Returns:
                float: Fitness value for the solution.
            """
            val = 0.0
            for i in range(D): val += np.abs(sol[i]) ** (i + 2)
            return val
        return evaluate

class Sphere3(Benchmark):
    r"""Implementation of rotated hyper-ellipsoid function.

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Function:
        **Sum of rotated hyper-elliposid function**
        :math:`f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2`

        **Input domain:**
        The function can be defined on any input domain but it is usually evaluated on the hypercube :math:`x_i ∈ [-65.536, 65.536]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2$

        Equation:
            \begin{equation} f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2 \end{equation}

        Domain:
            $-65.536 \leq x_i \leq 65.536$

    Reference URL:
        https://www.sfu.ca/~ssurjano/rothyp.html

    Attributes:
        Name (List[str]): Names of the benchmark.

    See Also:
        * :class:`NiaPy.benchmarks.Benchmark`
    """
    Name = ['Sphere3', 'sphere3']

    def __init__(self, Lower=-65.536, Upper=65.536, **kwargs):
        r"""Initialize of Sphere3 benchmark.

        Args:
            Lower (Optional[float]): Lower bound of problem.
            Upper (Optional[float]): Upper bound of problem.
            kwargs (dict): Additional arguments.

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
        return r'''$f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2$'''

    def function(self):
        r"""Return benchmark evaluation function.

        Returns:
            Callable[[int, Union[int, float, list, numpy.ndarray], dict], float]: Fitness function
        """
        def evaluate(D, sol, **kwargs):
            r"""Fitness function.

            Args:
                D (int): Dimensionality of the problem
                sol (Union[int, float, list, numpy.ndarray]): Solution to check.
                kwargs (dict): Additional arguments.

            Returns:
                float: Fitness value for the solution.
            """
            val = 0.0
            for i in range(D):
                v = .0
                for j in range(i + 1): val += np.abs(sol[j]) ** 2
                val += v
            return val
        return evaluate
