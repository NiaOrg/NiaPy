# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long, unused-argument, no-self-use, no-self-use, attribute-defined-outside-init, logging-not-lazy, len-as-condition, singleton-comparison, arguments-differ, redefined-builtin
import logging
from numpy import apply_along_axis, asarray, inf, argmin, argmax, sum, full
from NiaPy.algorithms.algorithm import Algorithm

__all__ = ['GravitationalSearchAlgorithm']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

class GravitationalSearchAlgorithm(Algorithm):
	r"""Implementation of gravitational search algorithm.

	**Algorithm:** Gravitational Search Algorithm

	**Date:** 2018

	**Author:** Klemen Berkoivč

	**License:** MIT

	**Reference URL:** https://doi.org/10.1016/j.ins.2009.03.004

	**Reference paper:** Esmat Rashedi, Hossein Nezamabadi-pour, Saeid Saryazdi, GSA: A Gravitational Search Algorithm, Information Sciences, Volume 179, Issue 13, 2009, Pages 2232-2248, ISSN 0020-0255
	"""
	def __init__(self, **kwargs):
		if kwargs.get('name', None) == None: Algorithm.__init__(self, name=kwargs.get('name', 'DifferentialEvolutionAlgorithm'), sName=kwargs.get('sName', 'DE'), **kwargs)
		else: Algorithm.__init__(self, **kwargs)

	def setParameters(self, NP=40, G_0=2.467, epsilon=1e-17, **ukwargs):
		r"""Set the algorithm parameters.

		**Arguments:**

		NP {integer} -- number of planets in population

		G_0 {real} -- starting gravitational constant
		"""
		self.NP, self.G_0, self.epsilon = NP, G_0, epsilon
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def G(self, t): return self.G_0 / t

	def d(self, x, y, ln=2): return sum((x - y) ** ln) ** (1 / ln)

	def runTask(self, task):
		X, v = self.uniform(task.Lower, task.Upper, [self.NP, task.D]), full([self.NP, task.D], 0.0)
		xb, xb_f = None, inf
		while not task.stopCondI():
			X_f = apply_along_axis(task.eval, 1, X)
			ib, iw = argmin(X_f), argmax(X_f)
			if xb_f > X_f[ib]: xb, xb_f = X[ib], X_f[ib]
			m = (X_f - X_f[iw]) / (X_f[ib] - X_f[iw])
			M = m / sum(m)
			Fi = asarray([[self.G(task.Iters) * ((M[i] * M[j]) / (self.d(X[i], X[j]) + self.epsilon)) * (X[j] - X[i]) for j in range(len(M))] for i in range(len(M))])
			F = sum(self.rand([self.NP, task.D]) * Fi, axis=1)
			a = F.T / (M + self.epsilon)
			v = self.rand([self.NP, task.D]) * v + a.T
			X = apply_along_axis(task.repair, 1, X + v)
		return xb, xb_f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
