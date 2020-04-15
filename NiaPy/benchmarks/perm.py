# encoding=utf8

"""Implementations of Perm function."""

from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Perm']

class Perm(Benchmark):
	r"""Implementations of Perm functions.

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Function:
	**Perm Function**

		:math:`f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2`

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [-D, D]`, for all :math:`i = 1, 2,..., D`.

		**Global minimum:**
		:math:`f(\textbf{x}^*) = 0` at :math:`\textbf{x}^* = (1, \frac{1}{2}, \cdots , \frac{1}{i} , \cdots , \frac{1}{D})`

	LaTeX formats:
		Inline:
				$f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2$

		Equation:
				\begin{equation} f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2 \end{equation}

		Domain:
				$-D \leq x_i \leq D$

	Attributes:
		Name (List[str]): Names of the benchmark.
		D (float): Function argument.
		beta (float): Function argument.

	See Also:
		* :class:`NiaPy.benchmarks.Benchmark`

	Reference:
		https://www.sfu.ca/~ssurjano/perm0db.html
	"""
	Name = ['Perm', 'perm']
	D = 10.0
	beta = 0.5

	def __init__(self, D=10.0, beta=.5, **kwargs):
		r"""Initialize of Bent Cigar benchmark.

		Args:
			D (Optional[float]): Function argument.
			beta (Optional[float]): Function argument.
			kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			:func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, -D, D)
		self.beta = beta

	@staticmethod
	def latex_code():
		r"""Return the latex code of the problem.

		Returns:
			str: Latex code
		"""
		return r'''$f(\textbf{x}) = \sum_{i = 1}^D \left( \sum_{j = 1}^D (j - \beta) \left( x_j^i - \frac{1}{j^i} \right) \right)^2$'''

	def function(self):
		r"""Return benchmark evaluation function.

		Returns:
			Callable[[int, Union[int, float, list, numpy.ndarray], Dict[str, Any]], float]: Fitness function
		"""
		beta = self.beta
		def f(D, X, **kwargs):
			r"""Fitness function.

			Args:
				D (int): Dimensionality of the problem
				X (Union[int, float, list, numpy.ndarray]): Solution to check.
				kwargs (Dict[str, Any]): Additional arguments.

			Returns:
				float: Fitness value for the solution.
			"""
			v = .0
			for i in range(1, D + 1):
				vv = .0
				for j in range(1, D + 1): vv += (j + beta) * (X[j - 1] ** i - 1 / j ** i)
				v += vv ** 2
			return v
		return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
