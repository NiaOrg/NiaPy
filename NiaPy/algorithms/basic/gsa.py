# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, unused-argument, no-self-use, no-self-use, attribute-defined-outside-init, logging-not-lazy, len-as-condition, singleton-comparison, arguments-differ, redefined-builtin, bad-continuation
import logging
from numpy import apply_along_axis, asarray, inf, argmin, argmax, sum, full
from NiaPy.algorithms.algorithm import Algorithm

__all__ = ['GravitationalSearchAlgorithm']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

class GravitationalSearchAlgorithm(Algorithm):
	r"""Implementation of gravitational search algorithm.

	Algorithm:
		Gravitational Search Algorithm

	Date:
		2018

	Author:
		Klemen Berkoivč

	License:
		MIT

	Reference URL:
		https://doi.org/10.1016/j.ins.2009.03.004

	Reference paper:
		Esmat Rashedi, Hossein Nezamabadi-pour, Saeid Saryazdi, GSA: A Gravitational Search Algorithm, Information Sciences, Volume 179, Issue 13, 2009, Pages 2232-2248, ISSN 0020-0255

	Attributes:
		Name (list of str): TODO
	"""
	Name = ['GravitationalSearchAlgorithm', 'GSA']

	@staticmethod
	def typeParameters(): return {
			'NP': lambda x: isinstance(x, int) and x > 0,
			'G_0': lambda x: isinstance(x, (int, float)) and x >= 0,
			'epsilon': lambda x: isinstance(x, float) and 0 < x < 1
	}

	def setParameters(self, NP=40, G_0=2.467, epsilon=1e-17, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
			G_0 (float): Starting gravitational constant.

		See Also:
			:func:`Algorithm.setParameters`
		"""
		Algorithm.setParameters(NP=NP)
		self.G_0, self.epsilon = G_0, epsilon
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def G(self, t):
		r"""

		Args:
			t:

		Returns:

		"""
		return self.G_0 / t

	def d(self, x, y, ln=2):
		r"""

		Args:
			x:
			y:
			ln:

		Returns:

		"""
		return sum((x - y) ** ln) ** (1 / ln)

	def initPopulation(self, task):
		r"""

		Args:
			task:

		Returns:

		"""
		X, v = self.uniform(task.Lower, task.Upper, [self.NP, task.D]), full([self.NP, task.D], 0.0)
		X_f = apply_along_axis(task.eval, 1, X)
		return X, X_f, {'v':v}

	def runIteration(self, task, X, X_f, xb, fxb, v, **dparams):
		r"""

		Args:
			task:
			X:
			X_f:
			xb:
			fxb:
			v:
			**dparams:

		Returns:

		"""
		ib, iw = argmin(X_f), argmax(X_f)
		m = (X_f - X_f[iw]) / (X_f[ib] - X_f[iw])
		M = m / sum(m)
		Fi = asarray([[self.G(task.Iters) * ((M[i] * M[j]) / (self.d(X[i], X[j]) + self.epsilon)) * (X[j] - X[i]) for j in range(len(M))] for i in range(len(M))])
		F = sum(self.rand([self.NP, task.D]) * Fi, axis=1)
		a = F.T / (M + self.epsilon)
		v = self.rand([self.NP, task.D]) * v + a.T
		X = apply_along_axis(task.repair, 1, X + v, self.Rand)
		X_f = apply_along_axis(task.eval, 1, X)
		return X, X_f, {'v':v}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
