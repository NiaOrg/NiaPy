# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy
import logging
import random
from numpy import inf, copy, logical_or, exp, fabs, where
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['SimulatedAnnealing']

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
		r"""Simulated Annealing Algorithm.

		**See**:
		Algorithm.__init__(self, **kwargs)
		"""
		super(SimulatedAnnealing, self).__init__(name='SimulatedAnnealing', sName='BBFA', **kwargs)

	# Different cooling methods
	def coolDelta(self, nFES):
		return currentT - self.deltaT

	def coolLinear(self, currentT, nFES):
		deltaT = self.T / nFES
		return currentT - deltaT


	def setParameters(self, **kwargs):
		r"""Set the algorithm parameters/arguments.

		**See**:
		SimulatedAnnealing.__setparams(self, n=10, c_a=1.5, c_r=0.5, **ukwargs)
		"""
		self.__setParams(**kwargs)

	def __setParams(self, delta=0.5, T=20, deltaT=0.8, coolingMethod=coolDelta, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		delta {real} -- amount of change for new individual in the neighbourhood
		T {real} -- starting temperature
		deltaT {real} -- change of temperature
		"""
		self.delta, self.T, self.curT, self.deltaT = delta, T, T, deltaT
		self.cool = coolingMethod
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def repair(self, val, task):
		ir = where(val > task.Upper)
		val[ir] = task.Upper[ir]
		ir = where(val < task.Lower)
		val[ir] = task.Lower[ir]
		return val
	
	def runTask(self, task):
				
		x = task.Lower + task.bRange * self.rand.rand(task.D)  # Random solution
		# print ("START X: " , x)
		xfit = task.eval(x)

		while not task.stopCond() or (self.curT < 0):
					
			c = x - self.delta / 2 + self.rand.rand(task.D) * self.delta
			c = self.repair(c,task)
			cfit = task.eval(c)
			
			deltaFit = cfit - xfit
			rand = self.rand.random_sample()
			# print ("CFIT:" , cfit)
			# print("%f / %f = %f" % (deltaFit, self.curT, deltaFit / self.curT) )
			if deltaFit < 0:
				x = c
				xfit = cfit			
			elif rand < exp(deltaFit / self.curT):
				x = c
				xfit = cfit
			# print(xfit)
			
			currentT = self.cool(currentT, nFES=task.nFES)
		# print ("X %s\nXFIT: %f " %(x,xfit))
		return x, xfit

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
