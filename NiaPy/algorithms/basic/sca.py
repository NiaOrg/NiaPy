# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, line-too-long, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, arguments-differ
import logging
from numpy import apply_along_axis, argmin, pi, inf, fabs, sin, cos
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['SineCosineAlgorithm']

class SineCosineAlgorithm(Algorithm):
	r"""Implementation of sine cosine algorithm.

	**Algorithm:** Sine Cosine Algorithm

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference URL:** https://www.sciencedirect.com/science/article/pii/S0950705115005043

	**Reference paper:** Seyedali Mirjalili, SCA: A Sine Cosine Algorithm for solving optimization problems, Knowledge-Based Systems, Volume 96, 2016, Pages 120-133, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2015.12.022.
	"""
	def __init__(self, **kwargs): Algorithm.__init__(self, name='SineCosineAlgorithm', sName='SCA', **kwargs)

	def setParameters(self, NP=25, a=3, Rmin=0, Rmax=2, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments:**

		NP {integer} -- number of individual in population

		a {real} -- parameter for controlon $r_1$ value

		Rmin {integer} -- minium value for $r_3$ value

		Rmax {integer} -- maximum value for $r_3$ value
		"""
		self.NP, self.a, self.Rmin, self.Rmax = NP, a, Rmin, Rmax
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def nextPos(self, x, x_b, r1, r2, r3, r4, task): return task.repair(x + r1 * (sin(r2) if r4 < 0.5 else cos(r2)) * fabs(r3 * x_b - x))

	def runTask(self, task):
		P, x, x_f = self.uniform(task.Lower, task.Upper, [self.NP, task.D]), None, inf
		while not task.stopCondI():
			P_f = apply_along_axis(task.eval, 1, P)
			ib = argmin(P_f)
			if P_f[ib] < x_f: x, x_f = P[ib], P_f[ib]
			r1, r2, r3, r4 = self.a - task.Iters * (self.a / task.nGEN), self.uniform(0, 2 * pi), self.uniform(self.Rmin, self.Rmax), self.rand()
			P = apply_along_axis(self.nextPos, 1, P, x, r1, r2, r3, r4, task)
		return x, x_f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
