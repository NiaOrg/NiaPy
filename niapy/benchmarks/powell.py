# encoding=utf8

"""Implementations of Powell function."""

from niapy.benchmarks.benchmark import Benchmark

__all__ = ['Powell']


class Powell(Benchmark):
    r"""Implementations of Powell functions.

    Date: 2018

    Author: Klemen Berkovič

    License: MIT

    Function:
    **Powell Function**

        :math:`f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right)`

        **Input domain:**
        The function can be defined on any input domain but it is usually
        evaluated on the hypercube :math:`x_i ∈ [-4, 5]`, for all :math:`i = 1, 2,..., D`.

        **Global minimum:**
        :math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (0, \cdots, 0)`

    LaTeX formats:
        Inline:
            $f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right)$

        Equation:
            \begin{equation} f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right) \end{equation}

        Domain:
            $-4 \leq x_i \leq 5$

    Reference:
        https://www.sfu.ca/~ssurjano/powell.html

    """

    Name = ['Powell']

    def __init__(self, lower=-4.0, upper=5.0):
        r"""Initialize of Powell benchmark.

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
        return r'''$f(\textbf{x}) = \sum_{i = 1}^{D / 4} \left( (x_{4 i - 3} + 10 x_{4 i - 2})^2 + 5 (x_{4 i - 1} - x_{4 i})^2 + (x_{4 i - 2} - 2 x_{4 i - 1})^4 + 10 (x_{4 i - 3} - x_{4 i})^4 \right)$'''

    def function(self):
        r"""Return benchmark evaluation function.

        Returns:
            Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function.

        """
        def f(dimension, x):
            r"""Fitness function.

            Args:
                dimension (int): Dimensionality of the problem
                x (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

            Returns:
                float: Fitness value for the solution.

            """
            v = 0.0
            for i in range(1, (dimension // 4) + 1):
                v += (x[4 * i - 4] + 10 * x[4 * i - 3]) ** 2 + 5 * (x[4 * i - 2] - x[4 * i - 1]) ** 2 + (x[4 * i - 3] - 2 * x[4 * i - 2]) ** 4 + 10 * (x[4 * i - 4] - x[4 * i - 1]) ** 4
            return v

        return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
