# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
"""Implementations of Levy function."""

from numpy import sin, pi

__all__ = ['Levy']

class Levy:
	r"""Implementations of Levy functions.

	Date: 2018
	Author: Klemen Berkovič
	License: MIT

	Function:
	**High Conditioned Elliptic Function**
	:math:`f(\textbf{x}) = \sum_{i=1}^D \left( 10^6 \right)^{ \frac{i - 1}{D - 1} } x_i^2`

	**Input domain:**
	The function can be defined on any input domain but it is usually
	evaluated on the hypercube :math:`x_i ∈ [0, \pi]`, for all :math:`i = 1, 2,..., D`.

	**Global minimum:**
	:math:`f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m}`

	LaTeX formats:
	Inline:
	$f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m}$

	Equation:
	\begin{equation} f(\textbf{x}) = - \sum_{i = 1}^{D} \sin(x_i) \sin\left( \frac{ix_i^2}{\pi} \right)^{2m} \end{equation}

	Domain:
	$0 \leq x_i \leq \pi$

	Reference:
	https://www.sfu.ca/~ssurjano/michal.html
	"""
	def __init__(self, Lower=0.0, Upper=2 * pi, m=10): self.Lower, self.Upper, = Lower, Upper 

	@classmethod
	def function(cls):
		def evaluate(D, X):
			v = 0.0
			for i in range(D): v += sin(X[i]) * sin((i * X[i] ** 2) / pi) ** (2 * cls.m)
			return -v
		return evaluate

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
