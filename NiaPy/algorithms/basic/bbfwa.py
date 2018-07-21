# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy
import logging
import numpy as np
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.benchmarks.utility import Task

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
		r"""**Arguments**:
		NP {integer} -- population size
		D {integer} -- dimension of problem
		nGEN {integer} -- nuber of generation/iterations
		nFES {integer} -- number of function evaluations
		benchmark {object} -- benchmark implementation object
		task {Task} -- task to perform optimization on
		**See**: BareBonesFireworksAlgorithm.setParameters
		"""
		super().__init__(name='BareBonesFireworksAlgorithm', sName='BBFA')
		task = kwargs.get('task', None)
		self.task = task if task != None else Task(kwargs.get('D', 10), kwargs.get('nFES', 100000), None, kwargs.get('benchmark', 'ackley'))
		self.setParameters(**kwargs)
	
	def setParameters(self, **kwargs):
		r"""**See**: BareBonesFireworksAlgorithm.__setParams"""
		self.__setParams(**kwargs)

	def __setParams(self, n=10, C_a=1.5, C_r=0.5, **ukwargs):
		r"""Function that sets the argumets of an algorithm
		**Arguments**:
		n {integer} -- number of sparks
		C_a {real} -- amplification coefficient > 1
		C_r {real} -- reduction coefficient < 1
		"""
		self.n, self.C_a, self.C_r = n, C_a, C_r
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))
	
	def runTask(self, task):
		x, A = np.random.uniform(task.Lower, task.Upper, task.D), task.bRange
		x_fit = task.eval(x)
		while not task.stopCond():
			S = [np.random.uniform(x - A, x + A) for i in range(self.n)]
			S_fit = np.apply_along_axis(task.eval, 1, S)
			is_min = np.argmin(S_fit)
			if S_fit[is_min] < x_fit: x, x_fit, A = S[is_min], S_fit[is_min], self.C_a * A
			else: A = self.C_r * A
		return x, x_fit

	def run(self): return self.runTask(self.task)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
