# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy
import logging
from numpy import apply_along_axis, argmin, inf, copy
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['HillClimbAlgorithm']

class HillClimbAlgorithm(Algorithm):
	r"""Implementation of iterative hill climbing algorithm.

	**Algorithm:** Hill Climbing Algorithm

	**Date:** 2018

	**Authors:** Jan Popič

	**License:** MIT

	**Reference URL:** wg

	**Reference paper:** wg
	"""
	def __init__(self, **kwargs):
		r"""Initialize Bare Bones Fireworks algorithm class.

		**See**:
		Algorithm.__init__(self, **kwargs)
		"""
		super(HillClimbAlgorithm, self).__init__(name='HillClimbAlgorithm', sName='BBFA', **kwargs)

	def setParameters(self, **kwargs):
		r"""Set the algorithm parameters/arguments.

		**See**:
		HillClimbAlgorithm.__setparams(self, n=10, c_a=1.5, c_r=0.5, **ukwargs)
		"""
		self.__setParams(**kwargs)

	def __setParams(self, delta=0.5, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		delta {real} -- amount of change for new posible solution
		"""
		self.delta = delta
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		xb, xbfit = None, inf
		while not task.stopCond():
			lo, x = False, task.Lower + task.bRange * self.rand.rand(task.D)
			xfit = task.eval(x)
			while not lo:
				X = list()
				for i in range(task.D):
					y = copy(x)
					y[i] += self.delta
					X.append(y)
				for i in range(task.D):
					y = copy(x)
					y[i] -= self.delta
					X.append(y)
				Xfit = apply_along_axis(task.eval, 1, X)
				XminI = argmin(Xfit)
				if Xfit[XminI] < xfit:
					xfit, x = Xfit[XminI], X[XminI]
					if xfit < xbfit: xbfit, xb = xfit, x
				else: lo = True
		return xb, xbfit

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
