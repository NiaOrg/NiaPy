# encoding=utf8
import logging

from numpy import random as rand, argmin, abs, mean
from NiaPy.util import levy_flight

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['HarrisHawksOptimization']


class HarrisHawksOptimization(Algorithm):
	r"""Implementation of Harris Hawks Optimization algorithm.

	Algorithm:
				Harris Hawks Optimization

	Date:
				2020

	Authors:
				Francisco Jose Solis-Munoz

	License:
				MIT

	Reference paper:
				Heidari et al. "Harris hawks optimization: Algorithm and applications". Future Generation Computer Systems. 2019. Vol. 97. 849-872.

	Attributes:
				Name (List[str]): List of strings representing algorithm name.
				levy (float): Levy factor.

	See Also:
				* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['HarrisHawksOptimization', 'HHO']

	def __init__(self, **kwargs):
		super(HarrisHawksOptimization, self).__init__(**kwargs)

	@staticmethod
	def algorithmInfo():
		r"""Get algorithms information.

		Returns:
					str: Algorithm information.

		See Also:
					* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Heidari et al. "Harris hawks optimization: Algorithm and applications". Future Generation Computer Systems. 2019. Vol. 97. 849-872."""

	@staticmethod
	def typeParameters():
		r"""Return dict with where key of dict represents parameter name and values represent checking functions for selected parameter.

		Returns:
					Dict[str, Callable]:
								* levy (Callable[[Union[float, int]], bool]): Levy factor.

		See Also:
					* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
				'levy': lambda x: isinstance(x, (float, int)) and x > 0,
		})
		return d

	def setParameters(self, NP=40, levy=0.01, **ukwargs):
		r"""Set the parameters of the algorithm.

		Args:
					levy (Optional[float]): Levy factor.

		See Also:
					* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.levy = levy

	def getParameters(self):
		r"""Get parameters of the algorithm.

		Returns:
					Dict[str, Any]
		"""
		d = Algorithm.getParameters(self)
		d.update({
				'levy': self.levy
		})
		return d

	def initPopulation(self, task, rnd=rand):
		r"""Initialize the starting population.

		Parameters:
					task (Task): Optimization task

		Returns:
					Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
								1. New population.
								2. New population fitness/function values.

		See Also:
					* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		Sol, Fitness, d = Algorithm.initPopulation(self, task)
		return Sol, Fitness, d

	def runIteration(self, task, Sol, Fitness, xb, fxb, **dparams):
		r"""Core function of Harris Hawks Optimization.

		Parameters:
					task (Task): Optimization task.
					Sol (numpy.ndarray): Current population
					Fitness (numpy.ndarray[float]): Current population fitness/funciton values
					xb (numpy.ndarray): Current best individual
					fxb (float): Current best individual function/fitness value
					dparams (Dict[str, Any]): Additional algorithm arguments

		Returns:
					Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
								1. New population
								2. New population fitness/function vlues
								3. New global best solution
								4. New global best fitness/objective value
		"""
		# Decreasing energy factor
		decreasing_energy_factor = 2 * (1 - (task.Iters + 1) / task.nGEN)
		mean_sol = mean(Sol)
		# Update population
		for i in range(self.NP):
				jumping_energy = self.Rand.uniform(0, 2)
				decreasing_energy_random = self.Rand.uniform(-1, 1)
				escaping_energy = decreasing_energy_factor * decreasing_energy_random
				escaping_energy_abs = abs(escaping_energy)
				random_number = self.Rand.rand()
				if escaping_energy >= 1 and random_number >= 0.5:
					# 0. Exploration: Random tall tree
					rhi = self.Rand.randint(0, self.NP)
					random_agent = Sol[rhi]
					Sol[i] = random_agent - self.Rand.rand() * abs(random_agent - 2 * self.Rand.rand() * Sol[i])
				elif escaping_energy_abs >= 1 and random_number < 0.5:
					# 1. Exploration: Family members mean
					Sol[i] = (xb - mean_sol) - self.Rand.rand() * self.Rand.uniform(task.Lower, task.Upper)
				elif escaping_energy_abs >= 0.5 and random_number >= 0.5:
					# 2. Exploitation: Soft besiege
					Sol[i] = \
						(xb - Sol[i]) - \
						escaping_energy * \
						abs(jumping_energy * xb - Sol[i])
				elif escaping_energy_abs < 0.5 and random_number >= 0.5:
					# 3. Exploitation: Hard besiege
					Sol[i] = \
						xb - \
						escaping_energy * \
						abs(xb - Sol[i])
				elif escaping_energy_abs >= 0.5 and random_number < 0.5:
					# 4. Exploitation: Soft besiege with pprogressive rapid dives
					cand1 = task.repair(xb - escaping_energy * abs(jumping_energy * xb - Sol[i]), rnd=self.Rand)
					random_vector = self.Rand.rand(task.D)
					cand2 = task.repair(cand1 + random_vector * levy_flight(self.levy, size=task.D, rng=self.Rand), rnd=self.Rand)
					if task.eval(cand1) < Fitness[i]:
						Sol[i] = cand1
					elif task.eval(cand2) < Fitness[i]:
						Sol[i] = cand2
				elif escaping_energy_abs < 0.5 and random_number < 0.5:
					# 5. Exploitation: Hard besiege with progressive rapid dives
					cand1 = task.repair(xb - escaping_energy * abs(jumping_energy * xb - mean_sol), rnd=self.Rand)
					random_vector = self.Rand.rand(task.D)
					cand2 = task.repair(cand1 + random_vector * levy_flight(self.levy, size=task.D, rng=self.Rand), rnd=self.Rand)
					if task.eval(cand1) < Fitness[i]:
						Sol[i] = cand1
					elif task.eval(cand2) < Fitness[i]:
						Sol[i] = cand2
				# Repair agent (from population) values
				Sol[i] = task.repair(Sol[i], rnd=self.Rand)
				# Eval population
				Fitness[i] = task.eval(Sol[i])
		# Get best of population
		best_index = argmin(Fitness)
		xb_cand = Sol[best_index].copy()
		fxb_cand = Fitness[best_index].copy()
		if fxb_cand < fxb:
				fxb = fxb_cand
				xb = xb_cand.copy()
		return Sol, Fitness, xb, fxb, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
