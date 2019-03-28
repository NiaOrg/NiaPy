# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, line-too-long, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, arguments-differ, bad-continuation
import logging
from numpy import apply_along_axis, argmin, pi, inf, fabs, sin, cos
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic.SineCosineAlgorithm')
logger.setLevel('INFO')

__all__ = ['SineCosineAlgorithm']

# FIXME test if algorithm realy works OK

class SineCosineAlgorithm(Algorithm):
	r"""Implementation of sine cosine algorithm.

	Algorithm:
		Sine Cosine Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.sciencedirect.com/science/article/pii/S0950705115005043

	Reference paper:
		Seyedali Mirjalili, SCA: A Sine Cosine Algorithm for solving optimization problems, Knowledge-Based Systems, Volume 96, 2016, Pages 120-133, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2015.12.022.

	Attributes:
		Name (list of str): List of string representing algorithm names
	"""
	Name = ['SineCosineAlgorithm', 'SCA']

	@staticmethod
	def typeParameters():
		r"""TODO.

		Returns:
			dict:
				* NP (func): TODO
				* a (func): TODO
				* Rmin (func): TODO
				* Rmax (func): TODO
		"""
		return {
			'NP': lambda x: isinstance(x, int) and x > 0,
			'a': lambda x: isinstance(x, (float, int)) and x > 0,
			'Rmin': lambda x: isinstance(x, (float, int)),
			'Rmax': lambda x: isinstance(x, (float, int))
		}

	def setParameters(self, NP=25, a=3, Rmin=0, Rmax=2, **ukwargs):
		r"""Set the arguments of an algorithm.

		Args:
			NP (int): Number of individual in population
			a (float): Parameter for controlon $r_1$ value
			Rmin (float): Minium value for $r_3$ value
			Rmax (float): Maximum value for $r_3$ value
		"""
		self.NP, self.a, self.Rmin, self.Rmax = NP, a, Rmin, Rmax
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def nextPos(self, x, x_b, r1, r2, r3, r4, task):
		r"""Move individual to new position in search space.

		Args:
			x (array): Individual represented with components
			x_b (array): Best individual represented with components
			r1 (float): Number dependent on algorithm iteration/generations
			r2 (float): Random number in range of 0 and 2 * PI
			r3 (float): Random number in range [Rmin, Rmax]
			r4 (float): Random number in range [0, 1]
			task (Task): Optimization task

		Returns:
			array: New individual that is moved based on individual ``x``
		"""
		return task.repair(x + r1 * (sin(r2) if r4 < 0.5 else cos(r2)) * fabs(r3 * x_b - x), self.Rand)

	def initPopulation(self, task):
		r"""Initialization of individuals.

		Args:
			task (Task): Optimization task

		Returns:
			Tuple[array of array of (float or int), array of float, dict]:
				1. Initialized population of individuals
				2. Function/fitness values for individuals
				3. Additional arguments
		"""
		P = self.uniform(task.bcLower(), task.bcUpper(), [self.NP, task.D])
		P_f = apply_along_axis(task.eval, 1, P)
		return P, P_f, {}

	def runIteration(self, task, P, P_f, xb, fxb, **dparams):
		r"""Core function of Sine Cosine Algorithm.

		Args:
			task (:obj:util.utility.Task): Optimization task
			P (array of array): Current population individuals
			P_f (array of float): Current population individulas function/fitness values
			xb (array of float or int): Current best solution to optimization task
			fxb (float): Current best function/fitness value
			dparams (dict): Additional parameters

		Returns:
			Tuple[array of array of (float or int), array of float, dict]:
				1. New population
				2. New populations fitness/function values
				3. Additional arguments
		"""
		r1, r2, r3, r4 = self.a - task.Iters * (self.a / task.Iters), self.uniform(0, 2 * pi), self.uniform(self.Rmin, self.Rmax), self.rand()
		P = apply_along_axis(self.nextPos, 1, P, xb, r1, r2, r3, r4, task)
		P_f = apply_along_axis(task.eval, 1, P)
		return P, P_f, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
