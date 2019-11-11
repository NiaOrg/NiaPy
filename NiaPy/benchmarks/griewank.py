# encoding=utf8

"""Implementation of Griewank funcion."""

from math import sqrt, cos
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Griewank', 'ExpandedGriewankPlusRosenbrock']


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
	Name = ['Griewank']

	def __init__(self, Lower=-100.0, Upper=100.0):
		r"""Initialize of Griewank benchmark.

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
		return r'''$f(\mathbf{x}) = \sum_{i=1}^D \frac{x_i^2}{4000} -
				\prod_{i=1}^D \cos(\frac{x_i}{\sqrt{i}}) + 1$'''

	def function(self):
		r"""Return benchmark evaluation function.

		Returns:
			Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function
		"""
		def evaluate(D, sol):
			r"""Fitness function.

			Args:
				D (int): Dimensionality of the problem
				sol (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

			Returns:
				float: Fitness value for the solution.
			"""
			val1, val2 = 0.0, 1.0
			for i in range(D):
				val1 += sol[i] ** 2 / 4000.0
				val2 *= cos(sol[i] / sqrt(i + 1))
			return val1 - val2 + 1.0
		return evaluate

class ExpandedGriewankPlusRosenbrock(Benchmark):
	r"""Implementation of Expanded Griewank's plus Rosenbrock function.

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

	Reference:
		http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf
	"""
	Name = ['ExpandedGriewankPlusRosenbrock']

	def __init__(self, Lower=-100.0, Upper=100.0):
		r"""Initialize of Expanded Griewank's plus Rosenbrock benchmark.

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
		return r'''$f(\textbf{x}) = h(g(x_D, x_1)) + \sum_{i=2}^D h(g(x_{i - 1}, x_i)) \\ g(x, y) = 100 (x^2 - y)^2 + (x - 1)^2 \\ h(z) = \frac{z^2}{4000} - \cos \left( \frac{z}{\sqrt{1}} \right) + 1$'''

	def function(self):
		r"""Return benchmark evaluation function.

		Returns:
			Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function
		"""
		def h(z): return z ** 2 / 4000 - cos(z / sqrt(1)) + 1
		def g(x, y): return 100 * (x ** 2 - y ** 2) ** 2 + (x - 1) ** 2
		def f(D, x):
			r"""Fitness function.

			Args:
				D (int): Dimensionality of the problem
				sol (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

			Returns:
				float: Fitness value for the solution.
			"""
			val = 0.0
			for i in range(1, D): val += h(g(x[i - 1], x[i]))
			return h(g(x[D - 1], x[0])) + val
		return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
