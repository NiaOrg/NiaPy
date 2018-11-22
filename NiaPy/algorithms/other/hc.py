# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, arguments-differ, redefined-outer-name, bad-continuation
import logging
from numpy import apply_along_axis, argmin, inf, copy
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['HillClimbAlgorithm']

def Neighborhood(x, delta, task):
	X = list()
	for i in range(task.D):
		y = copy(x)
		y[i] += delta
		X.append(y)
	for i in range(task.D):
		y = copy(x)
		y[i] -= delta
		X.append(y)
	Xfit = apply_along_axis(task.eval, 1, X)
	XminI = argmin(Xfit)
	return X[XminI], Xfit[XminI]

class HillClimbAlgorithm(Algorithm):
	r"""Implementation of iterative hill climbing algorithm.

	**Algorithm:** Hill Climbing Algorithm

	**Date:** 2018

	**Authors:** Jan Popič

	**License:** MIT

	**Reference URL:**

	**Reference paper:**
	"""
	Name = ['HillClimbAlgorithm', 'BBFA']

	@staticmethod
	def typeParameters(): return {
			'delta': lambda x: isinstance(x, (int, float)) and x > 0
	}

	def setParameters(self, delta=0.5, Neighborhood=Neighborhood, **ukwargs):
		r"""Set the algorithm parameters/arguments.

		**See**:
		HillClimbAlgorithm.__setparams(self, delta=0.5, Neighborhood=Neighborhood, **ukwargs)
		"""
		self.delta, self.Neighborhood = delta, Neighborhood
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		xb, xbfit = None, task.optType.value * inf
		while not task.stopCondI():
			lo, x = False, task.bcLower() + task.bcRange() * self.rand(task.D)
			xfit = task.eval(x)
			while not lo:
				Xn, XnFit = self.Neighborhood(x, self.delta, task)
				if XnFit < xfit:
					xfit, x = XnFit, Xn
					if xfit < xbfit: xbfit, xb = xfit, x
				else: lo = True
		return xb, xbfit

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
