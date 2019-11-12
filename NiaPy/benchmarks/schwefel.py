# encoding=utf8

"""Implementations of Schwefels functions."""

from math import sin, fmod, fabs, sqrt
from NiaPy.benchmarks.benchmark import Benchmark

__all__ = ['Schwefel', 'Schwefel221', 'Schwefel222', 'ModifiedSchwefel']

class Schwefel(Benchmark):
	r"""Implementation of Schewel function.

	Date: 2018

	Author: Lucija Brezočnik

	License: MIT

	Function: **Schwefel function**

		:math:`f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i \sin(\sqrt{|x_i|})`

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [-500, 500]`, for all :math:`i = 1, 2,..., D`.

		**Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

	LaTeX formats:
		Inline:
				$f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i \sin(\sqrt{|x_i|})$

		Equation:
				\begin{equation} f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i
				\sin(\sqrt{|x_i|}) \end{equation}

		Domain:
				$-500 \leq x_i \leq 500$

	Reference:
		https://www.sfu.ca/~ssurjano/schwef.html
	"""
	Name = ['Schwefel']

	def __init__(self, Lower=-500.0, Upper=500.0):
		r"""Initialize of Schwefel benchmark.

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
		return r'''$f(\textbf{x}) = 418.9829d - \sum_{i=1}^{D} x_i \sin(\sqrt{|x_i|})$'''

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
			val = 0.0
			for i in range(D):
				val += (sol[i] * sin(sqrt(abs(sol[i]))))
			return 418.9829 * D - val
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
				$f(\mathbf{x})=\max_{i=1,...,D}|x_i|$

		Equation:
				\begin{equation}f(\mathbf{x}) = \max_{i=1,...,D}|x_i| \end{equation}

		Domain:
				$-100 \leq x_i \leq 100$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013).
		A literature survey of benchmark functions for global optimisation problems.
		International Journal of Mathematical Modelling and Numerical Optimisation,
		4(2), 150-194.
	"""
	Name = ['Schwefel221']

	def __init__(self, Lower=-100.0, Upper=100.0):
		r"""Initialize of Schwefel221 benchmark.

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
		return r'''$f(\mathbf{x})=\max_{i=1,...,D}|x_i|$'''

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
			maximum = 0.0
			for i in range(D):
				if abs(sol[i]) > maximum:
					maximum = abs(sol[i])
			return maximum
		return evaluate

class Schwefel222(Benchmark):
	r"""Schwefel 2.22 function implementation.

	Date: 2018

	Author: Grega Vrbančič

	Licence: MIT

	Function: **Schwefel 2.22 function**

		:math:`f(\mathbf{x})=\sum_{i=1}^{D}|x_i|+\prod_{i=1}^{D}|x_i|`

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		**Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
				$f(\mathbf{x})=\sum_{i=1}^{D}|x_i|+\prod_{i=1}^{D}|x_i|$

		Equation:
				\begin{equation}f(\mathbf{x}) = \sum_{i=1}^{D}|x_i| +
				\prod_{i=1}^{D}|x_i| \end{equation}

		Domain:
				$-100 \leq x_i \leq 100$

	Reference paper:
		Jamil, M., and Yang, X. S. (2013).
		A literature survey of benchmark functions for global optimisation problems.
		International Journal of Mathematical Modelling and Numerical Optimisation,
		4(2), 150-194.
	"""
	Name = ['Schwefel222']

	def __init__(self, Lower=-100.0, Upper=100.0):
		r"""Initialize of Schwefel222 benchmark.

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
		return r'''$f(\mathbf{x})=\sum_{i=1}^{D}|x_i|+\prod_{i=1}^{D}|x_i|$'''

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
			part1 = 0.0
			part2 = 1.0
			for i in range(D):
				part1 += abs(sol[i])
				part2 *= abs(sol[i])
			return part1 + part2
		return evaluate

class ModifiedSchwefel(Benchmark):
	r"""Implementations of Modified Schwefel functions.

	Date: 2018

	Author: Klemen Berkovič

	License: MIT

	Function:
	**Modified Schwefel Function**

		:math:`f(\textbf{x}) = 418.9829 \cdot D - \sum_{i=1}^D h(x_i) \\ h(x) = g(x + 420.9687462275036)  \\ g(z) = \begin{cases} z \sin \left( | z |^{\frac{1}{2}} \right) &\quad | z | \leq 500 \\ \left( 500 - \mod (z, 500) \right) \sin \left( \sqrt{| 500 - \mod (z, 500) |} \right) - \frac{ \left( z - 500 \right)^2 }{ 10000 D }  &\quad z > 500 \\ \left( \mod (| z |, 500) - 500 \right) \sin \left( \sqrt{| \mod (|z|, 500) - 500 |} \right) + \frac{ \left( z - 500 \right)^2 }{ 10000 D } &\quad z < -500\end{cases}`

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [-100, 100]`, for all :math:`i = 1, 2,..., D`.

		**Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (420.968746,...,420.968746)`

	LaTeX formats:
		Inline:
				$f(\textbf{x}) = 418.9829 \cdot D - \sum_{i=1}^D h(x_i) \\ h(x) = g(x + 420.9687462275036)  \\ g(z) = \begin{cases} z \sin \left( | z |^{\frac{1}{2}} \right) &\quad | z | \leq 500 \\ \left( 500 - \mod (z, 500) \right) \sin \left( \sqrt{| 500 - \mod (z, 500) |} \right) - \frac{ \left( z - 500 \right)^2 }{ 10000 D }  &\quad z > 500 \\ \left( \mod (| z |, 500) - 500 \right) \sin \left( \sqrt{| \mod (|z|, 500) - 500 |} \right) + \frac{ \left( z - 500 \right)^2 }{ 10000 D } &\quad z < -500\end{cases}$

		Equation:
				\begin{equation} f(\textbf{x}) = 418.9829 \cdot D - \sum_{i=1}^D h(x_i) \\ h(x) = g(x + 420.9687462275036)  \\ g(z) = \begin{cases} z \sin \left( | z |^{\frac{1}{2}} \right) &\quad | z | \leq 500 \\ \left( 500 - \mod (z, 500) \right) \sin \left( \sqrt{| 500 - \mod (z, 500) |} \right) - \frac{ \left( z - 500 \right)^2 }{ 10000 D }  &\quad z > 500 \\ \left( \mod (| z |, 500) - 500 \right) \sin \left( \sqrt{| \mod (|z|, 500) - 500 |} \right) + \frac{ \left( z - 500 \right)^2 }{ 10000 D } &\quad z < -500\end{cases} \end{equation}

		Domain:
				$-100 \leq x_i \leq 100$

	Reference:
		http://www5.zzu.edu.cn/__local/A/69/BC/D3B5DFE94CD2574B38AD7CD1D12_C802DAFE_BC0C0.pdf
	"""
	Name = ['ModifiedSchwefel']

	def __init__(self, Lower=-100.0, Upper=100.0):
		r"""Initialize of Modified Schwefel benchmark.

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
		return r'''$f(\textbf{x}) = 418.9829 \cdot D - \sum_{i=1}^D h(x_i) \\ h(x) = g(x + 420.9687462275036)  \\ g(z) = \begin{cases} z \sin \left( | z |^{\frac{1}{2}} \right) &\quad | z | \leq 500 \\ \left( 500 - \mod (z, 500) \right) \sin \left( \sqrt{| 500 - \mod (z, 500) |} \right) - \frac{ \left( z - 500 \right)^2 }{ 10000 D }  &\quad z > 500 \\ \left( \mod (| z |, 500) - 500 \right) \sin \left( \sqrt{| \mod (|z|, 500) - 500 |} \right) + \frac{ \left( z - 500 \right)^2 }{ 10000 D } &\quad z < -500\end{cases}$'''

	def function(self):
		r"""Return benchmark evaluation function.

		Returns:
			Callable[[int, Union[int, float, List[int, float], numpy.ndarray]], float]: Fitness function
		"""
		def g(z, D):
			if z > 500: return (500 - fmod(z, 500)) * sin(sqrt(fabs(500 - fmod(z, 500)))) - (z - 500) ** 2 / (10000 * D)
			elif z < -500: return (fmod(z, 500) - 500) * sin(sqrt(fabs(fmod(z, 500) - 500))) + (z - 500) ** 2 / (10000 * D)
			return z * sin(fabs(z) ** (1 / 2))
		def h(x, D): return g(x + 420.9687462275036, D)
		def f(D, sol):
			r"""Fitness function.

			Args:
				D (int): Dimensionality of the problem
				sol (Union[int, float, List[int, float], numpy.ndarray]): Solution to check.

			Returns:
				float: Fitness value for the solution.
			"""
			val = 0.0
			for i in range(D): val += h(sol[i], D)
			return 418.9829 * D - val
		return f

class ExpandedScaffer(Benchmark):
	r"""Implementations of High Conditioned Elliptic functions.

	Date: 2018

	Author: Klemen Berkovič

	License: MIT

	Function:
	**High Conditioned Elliptic Function**

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
	Name = ['ExpandedScaffer']

	def __init__(self, Lower=-100.0, Upper=100.0):
		r"""Initialize of Expanded Scaffer benchmark.

		Args:
			Lower (Optional[float]): Lower bound of problem.
			Upper (Optional[float]): Upper bound of problem.

		See Also:
			:func:`NiaPy.benchmarks.Benchmark.__init__`
		"""
		Benchmark.__init__(self, Lower=Lower, Upper=Lower)

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
