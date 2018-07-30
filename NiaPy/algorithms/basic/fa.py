# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, redefined-builtin, line-too-long, no-self-use
import logging
from numpy import argsort, power as pow, sqrt, sum, exp, apply_along_axis, asarray
from NiaPy.algorithms.algorithm import Algorithm

__all__ = ['FireflyAlgorithm']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

class FireflyAlgorithm(Algorithm):
	r"""Implementation of Firefly algorithm.

	**Algorithm:** Firefly algorithm
	**Date:** 2016
	**Authors:** Iztok Fister Jr, Iztok Fister and Klemen Berkovič
	**License:** MIT
	**Reference paper:** Fister, I., Fister Jr, I., Yang, X. S., & Brest, J. (2013). A comprehensive review of firefly algorithms. Swarm and Evolutionary Computation, 13, 34-46.
	"""
	def __init__(self, **kwargs): super(FireflyAlgorithm, self).__init__(name='FireflyAlgorithm', sName='FA', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, NP=25, alpha=1, betamin=1, gamma=2, **ukwargs):
		r"""Set the parameters of the algorithm.

		**Arguments**:
		NP {integer} -- population size
		alpha {decimal} -- alpha parameter
		betamin {decimal} -- betamin parameter
		gamma {decimal} -- gamma parameter
		"""
		self.NP, self.alpha, self.betamin, self.gamma = NP, alpha, betamin, gamma
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def alpha_new(self, a, alpha):
		"""Optionally recalculate the new alpha value."""
		delta = 1.0 - pow(pow(10.0, -4.0) / 0.9, 1.0 / float(a))
		return (1 - delta) * alpha

	def move_ffa(self, i, Fireflies, Intensity, oFireflies, task):
		"""Move fireflies."""
		for j in range(self.NP):
			r = sqrt(sum((Fireflies[i] - Fireflies[j]) * (Fireflies[i] - Fireflies[j])))
			if Intensity[i] > Intensity[j]:
				beta = (1.0 - self.betamin) * exp(-self.gamma * pow(r, 2.0)) + self.betamin
				tmpf = self.alpha * (self.rand.uniform(0, 1, task.D) - 0.5) * task.bRange
				Fireflies[i] = Fireflies[i] * (1.0 - beta) + oFireflies[j] * beta + tmpf
		task.repair(Fireflies[i])
		return Fireflies[i]

	def runTask(self, task):
		"""Run."""
		Fireflies = self.rand.uniform(task.Lower, task.Upper, [self.NP, task.D])
		Intensity = apply_along_axis(task.eval, 1, Fireflies)
		alpha = self.alpha
		while not task.stopCond():
			alpha = self.alpha_new(task.nFES / self.NP, alpha)
			Index = argsort(Intensity)
			nFireflies, nIntensity = Fireflies[Index], Intensity[Index]
			Fireflies = asarray([self.move_ffa(i, nFireflies, nIntensity, Fireflies, task) for i in range(self.NP)])
			Intensity = apply_along_axis(task.eval, 1, Fireflies)
		return Fireflies[0], Intensity[0]

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
