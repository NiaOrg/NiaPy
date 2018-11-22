# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, unused-argument, arguments-differ, bad-continuation
import logging
from numpy import random as rand, exp
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['SimulatedAnnealing', 'SimulatedAnnealingF', 'coolDelta', 'coolLinear']

def coolDelta(currentT, T, deltaT, nFES):
	return currentT - deltaT

def coolLinear(currentT, T, deltaT, nFES):
	return currentT - T / nFES

def SimulatedAnnealingF(task, delta=1.5, delta_t=0.564, T=2000, cool=coolDelta, epsilon=1e-20, rnd=rand):
	x = task.Lower + task.bcRange() * rnd.rand(task.D)
	curT, xfit = T, task.eval(x)
	xb, xb_f = x, xfit
	while not task.stopCondI() and curT >= epsilon:
		c = task.repair(x - delta / 2 + rnd.rand(task.D) * delta, rnd=rnd)
		cfit = task.eval(c)
		deltaFit, r = cfit - xfit, rnd.rand()
		if deltaFit < 0 or r < exp(deltaFit / curT): x, xfit = c, cfit
		if xb_f > cfit: xb, xb_f = c, cfit
		curT = cool(curT, T, delta_t, nFES=task.nFES)
	return xb, xb_f

class SimulatedAnnealing(Algorithm):
	r"""Implementation of Simulated Annealing Algorithm.

	**Algorithm:** Simulated Annealing Algorithm

	**Date:** 2018

	**Authors:** Jan Popič

	**License:** MIT

	**Reference URL:**

	**Reference paper:**
	"""
	Name = ['SimulatedAnnealing', 'SA']

	@staticmethod
	def typeParameters(): return {
			'delta': lambda x: isinstance(x, (int, float)) and x > 0,
			'T': lambda x: isinstance(x, (int, float)) and x > 0,
			'deltaT': lambda x: isinstance(x, (int, float)) and x > 0,
			'epsilon': lambda x: isinstance(x, float) and 0 < x < 1
	}

	def setParameters(self, delta=0.5, T=2000, deltaT=0.8, coolingMethod=coolDelta, epsilon=1e-23, **ukwargs):
		r"""Set the algorithm parameters/arguments.

		Arguments:

		delta {real} -- Movemt for neighbour search

		T {real} -- Starting temperature

		deltaT {real} -- Change in temperature

		coolingMethod {function} -- Neigborhud function

		epsilon {real} -- Error value
		"""
		self.delta, self.T, self.deltaT, self.cool, self.epsilon = delta, T, deltaT, coolingMethod, epsilon
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task): return SimulatedAnnealingF(task, self.delta, self.deltaT, self.T, self.cool, self.epsilon, rnd=self.Rand)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
