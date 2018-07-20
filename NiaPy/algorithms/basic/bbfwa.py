# encoding=utf8
# pylint: disable=mixed-indentation
import numpy as np
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.benchmarks.utility import Task

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
	def __init__(self, D, nFES, n, C_a, C_r, benchmark):
		r"""**__init__(self, NP, D, nFES, T_min, T_max, omega, S_init, E_init, benchmark)**.
		Arguments:
		NP {integer} -- population size
		D {integer} -- dimension of problem
		nGEN {integer} -- nuber of generation/iterations
		nFES {integer} -- number of function evaluations
		n {integer} -- number of sparks
		C_a {real} -- amplification coefficient > 1
		C_r {real} -- reduction coefficient < 1
		benchmark {object} -- benchmark implementation object
		"""
		super().__init__('BareBonesFireworksAlgorithm', 'BBFA')
		self.n, self.C_a, self.C_r = n, C_a, C_r
		self.task = Task(D, nFES, None, benchmark)

	def __init__(self, **kwargs):
		# TODO
		pass

	def __init__(self, task, **kwargs):
		# TODO
		pass

	def setParameters(self, **kwargs):
		# TODO
		pass

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
