# encoding=utf8

"""Implementations of Bent Cigar functions."""

from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['BentCigar']

class BentCigar(Benchmark):
	r"""Implementations of Bent Cigar functions.

	Date: 2018

	Author: Klemen Berkovič

	License: MIT

	Function:
	**Bent Cigar Function**

		:math:`f(\textbf{x}) = x_1^2 + 10^6 \sum_{i=2}^D x_i^2`

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		**Global minimum:**
		:math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

	LaTeX formats:
		Inline:
				$f(\textbf{x}) = x_1^2 + 10^6 \sum_{i=2}^D x_i^2$

		Equation:
				\begin{equation} f(\textbf{x}) = x_1^2 + 10^6 \sum_{i=2}^D x_i^2 \end{equation}

		Domain:
				$-100 \leq x_i \leq 100$

	Reference:
		http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf
	"""
	Name = ['BentCigar']

	def __init__(self, Lower=-100.0, Upper=100.0):
		r"""Initialize of Bent Cigar benchmark.

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
		return r'''$f(\textbf{x}) = x_1^2 + 10^6 \sum_{i=2}^D x_i^2$'''

	def function(slef):
		r"""Return benchmark evaluation function.

		Returns:
			Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function
		"""
		def f(D, sol):
			r"""Fitness function.

			Args:
				D (int): Dimensionality of the problem
				sol (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

			Returns:
				float: Fitness value for the solution.
			"""
			val = 0.0
			for i in range(1, D): val += sol[i] ** 2
			return sol[0] ** 2 + 10 ** 6 * val
		return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
