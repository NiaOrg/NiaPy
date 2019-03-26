# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, line-too-long, arguments-differ, bad-continuation
import logging
from numpy import apply_along_axis, zeros, copy, argsort, concatenate, sort, array, exp, cos, pi
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['MothFlameOptimizer']

class MothFlameOptimizer(Algorithm):
	r"""MothFlameOptimizer of Moth flame optimizer.

	**Algorithm:** Moth flame optimizer

	**Date:** 2018

	**Author:** Kivanc Guckiran

	**License:** MIT

	**Reference paper:** Mirjalili, Seyedali. "Moth-flame optimization
	algorithm: A novel nature-inspired heuristic paradigm."
	Knowledge-Based Systems 89 (2015): 228-249.
	"""
	Name = ['MothFlameOptimizer', 'MFO']

	@staticmethod
	def typeParameters(): return {
			'NP': lambda x: isinstance(x, int) and x > 0
	}

	def setParameters(self, NP=25, **ukwargs):
		r"""Set the algorithm parameters.

		**Arguments:**

		NP {integer} -- Number of individuals in population
		"""
		self.NP = NP
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def initPopulation(self, task):
		# Init population and evaluate population
		moth_pos = task.Lower + task.bRange * self.rand([self.NP, task.D])
		moth_fitness = apply_along_axis(task.eval, 1, moth_pos)
		# Create best population
		indexes = argsort(moth_fitness)
		best_flames, best_flame_fitness = moth_pos[indexes], moth_fitness[indexes]
		# Init previous population
		previous_population, previous_fitness = zeros((self.NP, task.D)), zeros(self.NP)
		return moth_pos, moth_fitness, {'best_flames':best_flames, 'best_flame_fitness':best_flame_fitness, 'previous_population': previous_population, 'previous_fitness': previous_fitness}

	def runIteration(self, task, moth_pos, moth_fitness, xb, fxb, best_flames, best_flame_fitness, previous_population, previous_fitness, **dparams):
		# Previous positions
		previous_population, previous_fitness = moth_pos, moth_fitness
		# Create sorted population
		indexes = argsort(moth_fitness)
		sorted_fitness, sorted_population = moth_fitness[indexes], moth_pos[indexes]
		# Some parameters
		flame_no, a = round(self.NP - task.Iters * ((self.NP - 1) / task.nGEN)), -1 + task.Iters * ((-1) / task.nGEN)
		for i in range(self.NP):
			for j in range(task.D):
				distance_to_flame, b, t = abs(sorted_population[i, j] - moth_pos[i, j]), 1, (a - 1) * self.rand() + 1
				if i <= flame_no: moth_pos[i, j] = distance_to_flame * exp(b * t) * cos(2 * pi * t) + sorted_population[i, j]
				else: moth_pos[i, j] = distance_to_flame * exp(b * t) * cos(2 * pi * t) + sorted_population[flame_no, j]
		moth_pos = apply_along_axis(task.repair, 1, moth_pos, self.Rand)
		moth_fitness = apply_along_axis(task.eval, 1, moth_pos)
		double_population, double_fitness = concatenate((previous_population, best_flames), axis=0), concatenate((previous_fitness, best_flame_fitness), axis=0)
		indexes = argsort(double_fitness)
		double_sorted_fitness, double_sorted_population = double_fitness[indexes], double_population[indexes]
		for newIdx in range(2 * self.NP): double_sorted_population[newIdx] = array(double_population[indexes[newIdx], :])
		best_flame_fitness, best_flames = double_sorted_fitness[:self.NP], double_sorted_population[:self.NP]
		return moth_pos, moth_fitness, {'best_flames':best_flames, 'best_flame_fitness':best_flame_fitness, 'previous_population': previous_population, 'previous_fitness': previous_fitness}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
