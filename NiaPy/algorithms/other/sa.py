# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, unused-argument, arguments-differ
import logging
from numpy import exp
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['SimulatedAnnealing', 'coolDelta', 'coolLinear']

def coolDelta(currentT, T, deltaT, nFES):
	return currentT - deltaT

def coolLinear(currentT, T, deltaT, nFES):
	return currentT - T / nFES

class SimulatedAnnealing(Algorithm):
	r"""Implementation of Simulated Annealing Algorithm.

	**Algorithm:** Simulated Annealing Algorithm
	**Date:** 2018
	**Authors:** Jan Popič
	**License:** MIT
	**Reference URL:**
	**Reference paper:**
	"""
	def __init__(self, **kwargs): Algorithm.__init__(self, name='SimulatedAnnealing', sName='BBFA', **kwargs)

	def setParameters(self, delta=0.5, T=20, deltaT=0.8, coolingMethod=coolDelta, epsilon=1e-23, **ukwargs):
		r"""Set the algorithm parameters/arguments.

		Arguments:
		delta {real} --
		T {real} --
		deltaT {real} --
		coolingMethod {function} --
		epsilon {real}
		"""
		self.delta, self.T, self.deltaT, self.cool, self.epsilon = delta, T, deltaT, coolingMethod, epsilon
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		x = task.Lower + task.bRange * self.rand(task.D)  # Random solution
		curT, xfit = self.T, task.eval(x)
		while not task.stopCond():
			if (curT <= self.epsilon): curT = self.T
			c = task.repair(x - self.delta / 2 + self.rand(task.D) * self.delta)
			cfit = task.eval(c)
			deltaFit, rand = cfit - xfit, self.rand()
			if deltaFit < 0 or rand < exp(deltaFit / curT): x, xfit = c, cfit
			curT = self.cool(curT, self.T, self.deltaT, nFES=task.nFES)
		return x, xfit

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
