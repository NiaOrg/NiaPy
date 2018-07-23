# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy
import logging
from numpy import apply_along_axis, argmin
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['BareBonesFireworksAlgorithm']

class BareBonesFireworksAlgorithm(Algorithm):
	r"""Implementation of bare bone fireworks algorithm.

	**Algorithm:** Bare Bones Fireworks Algorithm

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference URL:**
	https://www.sciencedirect.com/science/article/pii/S1568494617306609

	**Reference paper:**
	Junzhi Li, Ying Tan, The bare bones fireworks algorithm: A minimalist global optimizer, Applied Soft Computing, Volume 62, 2018, Pages 454-462, ISSN 1568-4946, https://doi.org/10.1016/j.asoc.2017.10.046.
	"""
	def __init__(self, **kwargs):
		r"""Initialize Bare Bones Fireworks algorithm class.

		**See**:
		Algorithm.__init__(self, **kwargs)
		"""
		super(BareBonesFireworksAlgorithm, self).__init__(name='BareBonesFireworksAlgorithm', sName='BBFA', **kwargs)

	def setParameters(self, **kwargs):
		r"""Set the algorithm parameters/arguments.

		**See**:
		BareBonesFireworksAlgorithm.__setparams(self, n=10, c_a=1.5, c_r=0.5, **ukwargs)
		"""
		self.__setParams(**kwargs)

	def __setParams(self, n=10, C_a=1.5, C_r=0.5, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		n {integer} -- number of sparks $\in [1, \infty)$
		C_a {real} -- amplification coefficient $\in [1, \infty)$
		C_r {real} -- reduction coefficient $\in (0, 1)$
		"""
		self.n, self.C_a, self.C_r = n, C_a, C_r
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		x, A = self.rand.uniform(task.Lower, task.Upper, task.D), task.bRange
		x_fit = task.eval(x)
		S = None
		while not task.stopCond():
			S = self.rand.uniform(x - A, x + A, [self.n, task.D])
			S_fit = apply_along_axis(task.eval, 1, S)
			iS = argmin(S_fit)
			if S_fit[iS] < x_fit: x, x_fit, A = S[iS], S_fit[iS], self.C_a * A
			else: A = self.C_r * A
		return x, x_fit, S

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
