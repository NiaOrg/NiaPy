# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, singleton-comparison, multiple-statements, attribute-defined-outside-init, no-self-use, logging-not-lazy, unused-variable, arguments-differ, bad-continuation
import logging
from numpy import apply_along_axis, argmin, full, inf, where
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.util import fullArray

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['ParticleSwarmAlgorithm']

class ParticleSwarmAlgorithm(Algorithm):
	r"""Implementation of Particle Swarm Optimization algorithm.

	Algorithm:
		Particle Swarm Optimization algorithm

	Date:
		2018

	Authors:
		Lucija Brezočnik, Grega Vrbančič, Iztok Fister Jr. and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Kennedy, J. and Eberhart, R. "Particle Swarm Optimization". Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942--1948, 1995.

	Attributes:
		Name (list of str): List of strings representing algorithm names
	"""
	Name = ['ParticleSwarmAlgorithm', 'PSO']

	@staticmethod
	def typeParameters():
		r"""TODO.

		Returns:
			dict:
				* NP (func): TODO
				* C1 (func): TODO
				* C2 (func): TODO
				* w (func): TODO
				* vMin (func): TODO
				* vMax (func): TODO
		"""
		return {
			'NP': lambda x: isinstance(x, int) and x > 0,
			'C1': lambda x: isinstance(x, (int, float)) and x >= 0,
			'C2': lambda x: isinstance(x, (int, float)) and x >= 0,
			'w': lambda x: isinstance(x, float) and x >= 0,
			'vMin': lambda x: isinstance(x, (int, float)),
			'vMax': lambda x: isinstance(x, (int, float))
		}

	def setParameters(self, NP=25, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, **ukwargs):
		r"""Set Particle Swarm Algorithm main parameters.

		Args:
			NP (Optional[int]): Population size
			C1 (Optional[float]): Cognitive component
			C2 (Optional[float]): Social component
			w (Optional[float]): Inertial weight
			vMin (Optional[float]): Mininal velocity
			vMax (Optional[float]): Maximal velocity
			**ukwargs: Additional arguments
		"""
		self.NP, self.C1, self.C2, self.w, self.vMin, self.vMax = NP, C1, C2, w, vMin, vMax
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def repair(self, x, l, u):
		r"""Repair array to range

		Args:
			x (array): array to repair
			l (array): lower limit of allowd range
			u (array): upper limit of allowd range

		Returns:
			array: Repaird array
		"""
		ir = where(x < l)
		x[ir] = l[ir]
		ir = where(x > u)
		x[ir] = u[ir]
		return x

	def init(self, task):
		r"""Initalize dynamic arguments of Particle Swarm Optimization algorithm.

		Args:
			task (Task): Optimization task

		Returns:
			dict:
				* w (array): Inertial weight
				* vMin (array): Mininal velocity
				* vMax (array): Maximal velocity
				* V (array): Initial velocity of particle
		"""
		return {'w':fullArray(self.w, task.D), 'vMin':fullArray(self.vMin, task.D), 'vMax':fullArray(self.vMax, task.D), 'V':full([self.NP, task.D], 0.0)}

	def initPopulation(self, task):
		r"""

		Args:
			task (Task): Optimization task

		Returns:
			Tuple[array of array of (float or int), array of float, dict]:
				1. Initial population
				2. Initial population fitness/function values
				3. dict:
					* popb (array of array of (float or int): paticles best population
					* fpopb (array of float): particles best positions function/fitness value
					* w (array): Inertial weight
					* vMin (array): Mininal velocity
					* vMax (array): Maximal velocity
					* V (array of array of (float or int)): Initial velocity of particle

		See Also:
			init()
		"""
		d = self.init(task)
		pop = task.Lower + task.bRange * self.rand([self.NP, task.D])
		fpop = apply_along_axis(task.eval, 1, pop)
		d.update({'popb':pop, 'fpopb':fpop})
		return pop, fpop, d

	def runIteration(self, task, pop, fpop, xb, fxb, popb, fpopb, w, vMin, vMax, V, **dparams):
		r"""Core function of Particle Swarm Optimization algorithm.

		Args:
			task (Task): Optimization task
			pop (array of array of (float or int): Current populations
			fpop (array of float): Current population fitness/function vlues
			xb (array of (float of int): Current best particle
			fxb (float): Current best poatricle fitness/function value
			popb (array of array of (float or int): Particles best position
			fpopb (array of float): Patricles best postion fitness/function values
			w (array of (float or int)): Inertial weights
			vMin (array of (float or int)): Mininal velocity
			vMax (array of (float or int)): Maximal velocity
			V (array of array of (float or int)): Velocity of paticles
			**dparams: Additional function arguments

		Returns:
			Tuple[array of array of (float or int), array of float, dict]:
				1. New population
				2. New population fitness/function values
				3. dict:
					* popb (array of array of (float or int): paticles best population
					* fpopb (array of float): particles best positions function/fitness value
					* w (array): Inertial weight
					* vMin (array): Mininal velocity
					* vMax (array): Maximal velocity
					* V (array): Initial velocity of particle
		"""
		V = w * V + self.C1 * self.rand([self.NP, task.D]) * (popb - pop) + self.C2 * self.rand([self.NP, task.D]) * (xb - pop)
		V = apply_along_axis(self.repair, 1, V, vMin, vMax)
		pop += V
		pop = apply_along_axis(task.repair, 1, pop, rnd=self.Rand)
		fpop = apply_along_axis(task.eval, 1, pop)
		ip_pb = where(fpop > fpopb)
		popb[ip_pb], fpopb[ip_pb] = pop[ip_pb], fpop[ip_pb]
		return pop, fpop, {'popb':popb, 'fpopb':fpopb, 'w':w, 'vMin':vMin, 'vMax':vMax, 'V':V}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
