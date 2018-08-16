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
	def __init__(self, **kwargs):
		r"""Init Simulated Annealing Algorithm.

		**See**:
		Algorithm.__init__(self, **kwargs)
		"""
		Algorithm.__init__(self, name='SimulatedAnnealing', sName='BBFA', **kwargs)

	def setParameters(self, delta=0.5, T=20, deltaT=0.8, coolingMethod=coolDelta, **ukwargs):
		r"""Set the algorithm parameters/arguments.

		**See**:
		SimulatedAnnealing.__setparams(self, n=10, c_a=1.5, c_r=0.5, **ukwargs)
		"""
		self.delta, self.T, self.curT, self.deltaT, self.cool = delta, T, T, deltaT, coolingMethod
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		x = task.Lower + task.bRange * self.rand(task.D)  # Random solution
		curT = self.T
		xfit = task.eval(x)
		while not task.stopCond() or (curT < 0):
			c = x - self.delta / 2 + self.rand(task.D) * self.delta
			c = task.repair(c)
			cfit = task.eval(c)
			deltaFit = cfit - xfit
			rand = self.rand()
			if deltaFit < 0 or rand < exp(deltaFit / curT): x, xfit = c, cfit
			curT = self.cool(curT, self.T, self.deltaT, nFES=task.nFES)
		return x, xfit

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
