# encoding=utf8

"""Implementations of Schwefels functions."""

from math import sin, cos, sqrt
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['SchafferN2', 'SchafferN4', 'ExpandedSchaffer']

class SchafferN2(Benchmark):
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
	Name = ['SchafferN2']

	def __init__(self, Lower=-100.0, Upper=100.0):
		r"""Initialize of SchafferN2 benchmark.

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
		return r'''$f(\textbf{x}) = 0.5 + \frac{ \sin^2 \left( x_1^2 - x_2^2 \right) - 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 }$'''

	def function(self):
		r"""Return benchmark evaluation function.

		Returns:
			Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function
		"""
		def f(D, x):
			r"""Fitness function.

			Args:
				D (int): Dimensionality of the problem
				sol (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

			Returns:
				float: Fitness value for the solution.
			"""
			return 0.5 + (sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
		return f

class SchafferN4(Benchmark):
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
	Name = ['SchafferN4']

	def __init__(self, Lower=-100.0, Upper=100.0):
		r"""Initialize of ScahfferN4 benchmark.

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
		return r'''$f(\textbf{x}) = 0.5 + \frac{ \cos^2 \left( \sin \left( x_1^2 - x_2^2 \right) \right)- 0.5 }{ \left( 1 + 0.001 \left(  x_1^2 + x_2^2 \right) \right)^2 }$'''

	def function(self):
		r"""Return benchmark evaluation function.

		Returns:
			Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function
		"""
		def f(D, x):
			r"""Fitness function.

			Args:
				D (int): Dimensionality of the problem
				sol (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

			Returns:
				float: Fitness value for the solution.
			"""
			return 0.5 + (cos(sin(x[0] ** 2 - x[1] ** 2)) ** 2 - 0.5) / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
		return f

class ExpandedSchaffer(Benchmark):
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
	Name = ['ExpandedSchaffer']

	def __init__(self, Lower=-100.0, Upper=100.0):
		r"""Initialize of Expanded Scaffer benchmark.

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
		return r'''$f(\textbf{x}) = g(x_D, x_1) + \sum_{i=2}^D g(x_{i - 1}, x_i) \\ g(x, y) = 0.5 + \frac{\sin \left(\sqrt{x^2 + y^2} \right)^2 - 0.5}{\left( 1 + 0.001 (x^2 + y^2) \right)}^2$'''

	def function(self):
		r"""Return benchmark evaluation function.

		Returns:
			Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function
		"""
		def g(x, y): return 0.5 + (sin(sqrt(x ** 2 + y ** 2)) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2
		def f(D, x):
			r"""Fitness function.

			Args:
				D (int): Dimensionality of the problem
				sol (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

			Returns:
				float: Fitness value for the solution.
			"""
			val = 0.0
			for i in range(1, D): val += g(x[i - 1], x[i])
			return g(x[D - 1], x[0]) + val
		return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
