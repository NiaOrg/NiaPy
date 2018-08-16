# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, redefined-builtin, line-too-long, no-self-use, arguments-differ, no-else-return
import logging
from numpy import argsort, argmin, sum, exp, apply_along_axis, asarray, where, inf
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
	def __init__(self, **kwargs): Algorithm.__init__(self, name='FireflyAlgorithm', sName='FA', **kwargs)

	def setParameters(self, NP=20, alpha=1, betamin=1, gamma=2, **ukwargs):
		r"""Set the parameters of the algorithm.

		**Arguments:**

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

	def move_ffa(self, i, Fireflies, Intensity, oFireflies, alpha, task):
		"""Move fireflies."""
		moved = False
		for j in range(self.NP):
			r = sum((Fireflies[i] - Fireflies[j]) ** 2) ** (1 / 2)
			if Intensity[i] <= Intensity[j]: continue
			beta = (1.0 - self.betamin) * exp(-self.gamma * r ** 2.0) + self.betamin
			tmpf = alpha * (self.uniform(0, 1, task.D) - 0.5) * task.bRange
			Fireflies[i] = task.repair(Fireflies[i] * (1.0 - beta) + oFireflies[j] * beta + tmpf)
			moved = True
		return Fireflies[i], moved

	def getBest(self, xb, xb_f, Fireflies, Intensity):
		ib = argmin(Intensity)
		if xb_f > Intensity[ib]: return Fireflies[ib], Intensity[ib]
		else: return xb, xb_f

	def runTask(self, task):
		"""Run."""
		Fireflies = self.uniform(task.Lower, task.Upper, [self.NP, task.D])
		Intensity = apply_along_axis(task.eval, 1, Fireflies)
		(xb, xb_f), alpha = self.getBest(None, inf, Fireflies, Intensity), self.alpha
		while not task.stopCondI():
			alpha = self.alpha_new(task.nFES / self.NP, alpha)
			Index = argsort(Intensity)
			tmp = [self.move_ffa(i, Fireflies[Index], Intensity[Index], Fireflies, alpha, task) for i in range(self.NP)]
			Fireflies, evalF = asarray([tmp[i][0] for i in range(len(tmp))]), asarray([tmp[i][1] for i in range(len(tmp))])
			Intensity[where(evalF)] = apply_along_axis(task.eval, 1, Fireflies[where(evalF)])
			xb, xb_f = self.getBest(xb, xb_f, Fireflies, Intensity)
		return xb, xb_f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
