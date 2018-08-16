# encoding=utf8
# pylint: disable=anomalous-backslash-in-string, mixed-indentation, redefined-builtin, multiple-statements, old-style-class

from numpy import abs

__all__ = ['Sphere', 'Sphere2', 'Sphere3']


class Sphere:
	r"""Implementation of Sphere functions.

	Date: 2018

	Authors: Iztok Fister Jr.

	License: MIT

	Function: **Sphere function**

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
	"""
	def __init__(self, Lower=-5.12, Upper=5.12):
		self.Lower = Lower
		self.Upper = Upper

	@classmethod
	def function(cls):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val += sol[i] ** 2
			return val
		return evaluate

class Sphere2:
	r"""Implementation of Sphere with different powers function.

	Date: 2018

	Authors: Klemen Berkovič

	License: MIT

	Function: **Sun of different powers function**

		:math:`f(\textbf{x}) = \sum_{i = 1}^D | x_i |^{i + 1}`

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [-1, 1]`, for all :math:`i = 1, 2,..., D`.

		**Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
				$f(\textbf{x}) = \sum_{i = 1}^D | x_i |^{i + 1}$

		Equation:
				\begin{equation} f(\textbf{x}) = \sum_{i = 1}^D | x_i |^{i + 1} \end{equation}

		Domain:
				$-1 \leq x_i \leq 1$

	Reference URL:
	https://www.sfu.ca/~ssurjano/sumpow.html
	"""
	def __init__(self, Lower=-1., Upper=1.):
		self.Lower = Lower
		self.Upper = Upper

	@classmethod
	def function(cls):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val += abs(sol[i]) ** (i + 2)
			return val
		return evaluate

class Sphere3:
	r"""Implementation of rotated hyper-ellipsoid function.

	Date: 2018

	Authors: Klemen Berkovič

	License: MIT

	Function: **Sun of rotated hyper-elliposid function**

		:math:`f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2`

		**Input domain:**
		The function can be defined on any input domain but it is usually
		evaluated on the hypercube :math:`x_i ∈ [-65.536, 65.536]`, for all :math:`i = 1, 2,..., D`.

		**Global minimum:** :math:`f(x^*) = 0`, at :math:`x^* = (0,...,0)`

	LaTeX formats:
		Inline:
				$f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2$

		Equation:
				\begin{equation} f(\textbf{x}) = \sum_{i = 1}^D \sum_{j = 1}^i x_j^2 \end{equation}

		Domain:
				$-65.536 \leq x_i \leq 65.536$

	Reference URL:
	https://www.sfu.ca/~ssurjano/rothyp.html
	"""
	def __init__(self, Lower=-65.536, Upper=65.536):
		self.Lower = Lower
		self.Upper = Upper

	@classmethod
	def function(cls):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D):
				v = .0
				for j in range(i + 1): val += abs(sol[j]) ** 2
				val += v
			return val
		return evaluate

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
