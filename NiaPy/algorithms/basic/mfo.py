# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, line-too-long, arguments-differ, bad-continuation
import logging
import numpy as np
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

	def repair(self, x, task):
		"""Find limits."""
		ir = np.where(x > task.bcUpper())
		x[ir] = task.bcUpper()[ir]
		ir = np.where(x < task.bcLower())
		x[ir] = task.bcLower()[ir]
		return x

	def runTask(self, task):
		"""Run."""
		moth_pos = task.Lower + task.bRange * self.rand([self.NP, task.D])
		moth_fitness = np.zeros(self.NP)

		sorted_population = np.copy(moth_pos)
		sorted_fitness = np.zeros(self.NP)

		best_flames = np.copy(moth_pos)
		best_flame_fitness = np.zeros(self.NP)

		double_population = np.zeros((2 * self.NP, task.D))
		double_fitness = np.zeros(2 * self.NP)

		double_sorted_population = np.zeros((2 * self.NP, task.D))
		double_sorted_fitness = np.zeros(2 * self.NP)

		previous_population = np.zeros((self.NP, task.D))
		previous_fitness = np.zeros(self.NP)

		generation = 1

		while generation < task.nGEN:
			flame_no = round(self.NP - generation * ((self.NP - 1) / task.nGEN))

			for i in np.arange(0, self.NP):
				self.repair(moth_pos[i], task)
				moth_fitness[i] = task.eval(moth_pos[i])

			if generation == 1:
				sorted_fitness = np.sort(moth_fitness)
				indexes = np.argsort(moth_fitness)

				sorted_population = moth_pos[indexes, :]

				best_flames = sorted_population
				best_flame_fitness = sorted_fitness
			else:
				double_population = np.concatenate((previous_population, best_flames), axis=0)
				double_fitness = np.concatenate((previous_fitness, best_flame_fitness), axis=0)

				double_sorted_fitness = np.sort(double_fitness)
				indexes = np.argsort(double_fitness)

				for newIdx in np.arange(0, 2 * self.NP):
					double_sorted_population[newIdx, :] = np.array(double_population[indexes[newIdx], :])

				sorted_fitness = double_sorted_fitness[0:self.NP]
				sorted_population = double_sorted_population

				best_flames = sorted_population
				best_flame_fitness = sorted_fitness

			best_flame_score = sorted_fitness[0]
			best_flame_pos = sorted_population[0, :]

			previous_population = moth_pos
			previous_fitness = moth_fitness

			a = -1 + generation * ((-1) / task.nGEN)

			for i in np.arange(0, self.NP):
				for j in np.arange(0, task.D):
					distance_to_flame = abs(sorted_population[i, j] - moth_pos[i, j])
					b = 1
					t = (a - 1) * self.rand() + 1

					if i <= flame_no:
						moth_pos[i, j] = distance_to_flame * np.exp(b * t) * np.cos(2 * np.pi * t) + sorted_population[i, j]
					else:
						moth_pos[i, j] = distance_to_flame * np.exp(b * t) * np.cos(2 * np.pi * t) + sorted_population[flame_no, j]

			generation += 1

		return best_flame_pos, best_flame_score

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
