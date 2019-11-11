# encoding=utf8
import logging

from numpy import apply_along_axis, zeros, argsort, concatenate, array, exp, cos, pi

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['MothFlameOptimizer']

class MothFlameOptimizer(Algorithm):
	r"""MothFlameOptimizer of Moth flame optimizer.

	Algorithm:
		Moth flame optimizer

	Date:
		2018

	Author:
		Kivanc Guckiran and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Mirjalili, Seyedali. "Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm." Knowledge-Based Systems 89 (2015): 228-249.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.algorithm.Algorithm`
	"""
	Name = ['MothFlameOptimizer', 'MFO']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Mirjalili, Seyedali. "Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm." Knowledge-Based Systems 89 (2015): 228-249."""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]: TODO

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.typeParameters`
		"""
		return Algorithm.typeParameters()

	def setParameters(self, NP=25, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
			NP (int): Number of individuals in population

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)

	def initPopulation(self, task):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized population
				2. Initialized population function/fitness values
				3. Additional arguments:
					* best_flames (numpy.ndarray): Best individuals
					* best_flame_fitness (numpy.ndarray): Best individuals fitness/function values
					* previous_population (numpy.ndarray): Previous population
					* previous_fitness (numpy.ndarray[float]): Previous population fitness/function values

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.initPopulation`
		"""
		moth_pos, moth_fitness, d = Algorithm.initPopulation(self, task)
		# Create best population
		indexes = argsort(moth_fitness)
		best_flames, best_flame_fitness = moth_pos[indexes], moth_fitness[indexes]
		# Init previous population
		previous_population, previous_fitness = zeros((self.NP, task.D)), zeros(self.NP)
		d.update({'best_flames': best_flames, 'best_flame_fitness': best_flame_fitness, 'previous_population': previous_population, 'previous_fitness': previous_fitness})
		return moth_pos, moth_fitness, d

	def runIteration(self, task, moth_pos, moth_fitness, xb, fxb, best_flames, best_flame_fitness, previous_population, previous_fitness, **dparams):
		r"""Core function of MothFlameOptimizer algorithm.

		Args:
			task (Task): Optimization task.
			moth_pos (numpy.ndarray): Current population.
			moth_fitness (numpy.ndarray): Current population fitness/function values.
			xb (numpy.ndarray): Current population best individual.
			fxb (float): Current best individual
			best_flames (numpy.ndarray): Best found individuals
			best_flame_fitness (numpy.ndarray): Best found individuals fitness/function values
			previous_population (numpy.ndarray): Previous population
			previous_fitness (numpy.ndarray): Previous population fitness/function values
			**dparams (Dict[str, Any]): Additional parameters

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population.
				2. New population fitness/function values.
				3. New global best solution
				4. New global best fitness/objective value
				5. Additional arguments:
					* best_flames (numpy.ndarray): Best individuals.
					* best_flame_fitness (numpy.ndarray): Best individuals fitness/function values.
					* previous_population (numpy.ndarray): Previous population.
					* previous_fitness (numpy.ndarray): Previous population fitness/function values.
		"""
		# Previous positions
		previous_population, previous_fitness = moth_pos, moth_fitness
		# Create sorted population
		indexes = argsort(moth_fitness)
		sorted_population = moth_pos[indexes]
		# Some parameters
		flame_no, a = round(self.NP - task.Iters * ((self.NP - 1) / task.nGEN)), -1 + task.Iters * ((-1) / task.nGEN)
		for i in range(self.NP):
			for j in range(task.D):
				distance_to_flame, b, t = abs(sorted_population[i, j] - moth_pos[i, j]), 1, (a - 1) * self.rand() + 1
				if i <= flame_no: moth_pos[i, j] = distance_to_flame * exp(b * t) * cos(2 * pi * t) + sorted_population[i, j]
				else: moth_pos[i, j] = distance_to_flame * exp(b * t) * cos(2 * pi * t) + sorted_population[flame_no, j]
		moth_pos = apply_along_axis(task.repair, 1, moth_pos, self.Rand)
		moth_fitness = apply_along_axis(task.eval, 1, moth_pos)
		xb, fxb = self.getBest(moth_pos, moth_fitness, xb, fxb)
		double_population, double_fitness = concatenate((previous_population, best_flames), axis=0), concatenate((previous_fitness, best_flame_fitness), axis=0)
		indexes = argsort(double_fitness)
		double_sorted_fitness, double_sorted_population = double_fitness[indexes], double_population[indexes]
		for newIdx in range(2 * self.NP): double_sorted_population[newIdx] = array(double_population[indexes[newIdx], :])
		best_flame_fitness, best_flames = double_sorted_fitness[:self.NP], double_sorted_population[:self.NP]
		return moth_pos, moth_fitness, xb, fxb, {'best_flames': best_flames, 'best_flame_fitness': best_flame_fitness, 'previous_population': previous_population, 'previous_fitness': previous_fitness}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
