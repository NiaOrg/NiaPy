# encoding=utf8
import logging
from numpy import where, argmin, argsort, asarray, ndarray, pow, random as rand
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

	**Reference paper:**
	Fister, I., Fister Jr, I., Yang, X. S., & Brest, J. (2013). A comprehensive review of firefly algorithms. Swarm and Evolutionary Computation, 13, 34-46.
	"""

	def __init__(self, **kwargs): super(FireflyAlgorithm, self).__init__(name='FireflyAlgorithm', sName='FA', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, NP=25, alpha=1, betamin=1, gamma=2, **kwargs):
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
		delta = 1.0 - pow((pow(10.0, -4.0) / 0.9), 1.0 / float(a))
		return (1 - delta) * alpha

	def FindLimits(self, x, task):
		"""Find limits."""
		ir = where(x > task.Upper)
		x[ir] = task.Upper[ir]
		ir = where(x < task.Lower)
		x[ir] = task.Lower[ir]
		return x

	def move_ffa(self):
		"""Move fireflies."""
		for i in range(self.NP):
			scale = abs(self.Upper - self.Lower)
			for j in range(self.NP):
				r = 0.0
				for k in range(self.D): r += (self.Fireflies[i][k] - self.Fireflies[j][k]) * (self.Fireflies[i][k] - self.Fireflies[j][k])
					r = math.sqrt(r)
				if self.Intensity[i] > self.Intensity[j]: 
					beta = (1.0 - self.betamin) * math.exp(-self.gamma * math.pow(r, 2.0)) + self.betamin
					for k in range(self.D):
						r = random.uniform(0, 1)
						tmpf = self.alpha * (r - 0.5) * scale
						self.Fireflies[i][k] = self.Fireflies[i][k] * (1.0 - beta) + self.Fireflies_tmp[j][k] * beta + tmpf
			self.FindLimits(i)

	def runTask(self, task):
		"""Run."""
		Fireflies = self.rand.uniform(task.Lower, task.Upper, [self.NP, task.D])
		Fitness = apply_along_axis(task.eval, 1, S)
		Intensity = Fitness
		alpha = self.alpha
		while not taks.stopCond():
			alpha = self.alpha_new(task.nFES / self.NP, alpha)
			Index = argsort(Intensity)
			self.replace_ffa()
			self.fbest = self.Intensity[0]
			self.move_ffa()
			Fitness = apply_along_axis(task.eval, 1, S)
			Intensity = Fitness
		return self.fbest

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
