# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
"""Implementations of Perm function."""

__all__ = ['Perm']

class Perm:
	r"""Implementations of Perm functions.

	Date: 2018

	Author: Klemen Berkovič

	License: MIT

	Arguments:
	beta {real} -- value added to inner sum of funciton

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

	Reference:
	https://www.sfu.ca/~ssurjano/perm0db.html
	"""
	def __init__(self, D=10.0, beta=.5): self.Lower, self.Upper, Perm.beta = -D, D, beta

	@classmethod
	def function(cls):
		def f(D, X):
			v = .0
			for i in range(1, D + 1):
				vv = .0
				for j in range(1, D + 1): vv += (j + cls.beta) * (X[j - 1] ** i - 1 / j ** i)
				v += vv ** 2
			return v
		return f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
