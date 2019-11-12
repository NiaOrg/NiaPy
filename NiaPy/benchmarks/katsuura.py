# encoding=utf8

"""Implementations of Katsuura functions."""

from math import fabs
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Katsuura']

class Katsuura(Benchmark):
	r"""Implementations of Katsuura functions.

	Date: 2018

	Author: Klemen Berkovič

	License: MIT

	Function:
	**Katsuura Function**

		:math:`f(\textbf{x}) = \frac{10}{D^2} \prod_{i=1}^D \left( 1 + i \sum_{j=1}^{32} \frac{| 2^j x_i - round\left(2^j x_i \right) |}{2^j} \right)^\frac{10}{D^{1.2}} - \frac{10}{D^2}`

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		**Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

	LaTeX formats:
		Inline:
				$f(\textbf{x}) = \frac{10}{D^2} \prod_{i=1}^D \left( 1 + i \sum_{j=1}^{32} \frac{| 2^j x_i - round\left(2^j x_i \right) |}{2^j} \right)^\frac{10}{D^{1.2}} - \frac{10}{D^2}$

		Equation:
				\begin{equation} f(\textbf{x}) = \frac{10}{D^2} \prod_{i=1}^D \left( 1 + i \sum_{j=1}^{32} \frac{| 2^j x_i - round\left(2^j x_i \right) |}{2^j} \right)^\frac{10}{D^{1.2}} - \frac{10}{D^2} \end{equation}

		Domain:
				$-100 \leq x_i \leq 100$

	Reference:
		http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf
	"""
	Name = ['Katsuura']

	def __init__(self, Lower=-100.0, Upper=100.0, **kwargs):
		r"""Initialize of Katsuura benchmark.

		Args:
			Lower (Optional[float]): Lower bound of problem.
			Upper (Optional[float]): Upper bound of problem.

		See Also:
			:func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower, Upper, **kwargs)

	@staticmethod
	def latex_code():
		r"""Return the latex code of the problem.

		Returns:
			str: Latex code
		"""
		return r'''$f(\textbf{x}) = \frac{10}{D^2} \prod_{i=1}^D \left( 1 + i \sum_{j=1}^{32} \frac{| 2^j x_i - round\left(2^j x_i \right) |}{2^j} \right)^\frac{10}{D^{1.2}} - \frac{10}{D^2}$'''

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
			val = 1.0
			for i in range(D):
				valt = 1.0
				for j in range(1, 33): valt += fabs(2 ** j * x[i] - round(2 ** j * x[i])) / 2 ** j
				val *= (1 + (i + 1) * valt) ** (10 / D ** 1.2) - (10 / D ** 2)
			return 10 / D ** 2 * val
		return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
