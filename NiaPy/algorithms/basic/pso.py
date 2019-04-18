# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, singleton-comparison, multiple-statements, attribute-defined-outside-init, no-self-use, logging-not-lazy, unused-variable, arguments-differ, bad-continuation
import logging
from numpy import apply_along_axis, full, where
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
		Name (List[str]): List of strings representing algorithm names

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['ParticleSwarmAlgorithm', 'PSO']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Kennedy, J. and Eberhart, R. "Particle Swarm Optimization". Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942--1948, 1995."""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* NP (Callable[[int], bool])
				* C1 (Callable[[Union[int, float]], bool])
				* C2 (Callable[[Union[int, float]], bool])
				* w (Callable[[float], bool])
				* vMin (Callable[[Union[int, float]], bool])
				* vMax (Callable[[Union[int, float], bool])
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

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.C1, self.C2, self.w, self.vMin, self.vMax = C1, C2, w, vMin, vMax
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def repair(self, x, l, u):
		r"""Repair array to range.

		Args:
			x (numpy.ndarray): Array to repair.
			l (numpy.ndarray): Lower limit of allowed range.
			u (numpy.ndarray): Upper limit of allowed range.

		Returns:
			numpy.ndarray: Repaired array.
		"""
		ir = where(x < l)
		x[ir] = l[ir]
		ir = where(x > u)
		x[ir] = u[ir]
		return x

	def init(self, task):
		r"""Initialize dynamic arguments of Particle Swarm Optimization algorithm.

		Args:
			task (Task): Optimization task.

		Returns:
			Dict[str, Any]:
				* w (numpy.ndarray): Inertial weight.
				* vMin (numpy.ndarray): Mininal velocity.
				* vMax (numpy.ndarray): Maximal velocity.
				* V (numpy.ndarray): Initial velocity of particle.
		"""
		return {'w': fullArray(self.w, task.D), 'vMin': fullArray(self.vMin, task.D), 'vMax': fullArray(self.vMax, task.D), 'V': full([self.NP, task.D], 0.0)}

	def initPopulation(self, task):
		r"""Initialize population and dynamic arguments of the Particle Swarm Optimization algorithm.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. Initial population.
				2. Initial population fitness/function values.
				3. Additional arguments:
					* popb (numpy.ndarray): particles best population.
					* fpopb (numpy.ndarray[float]): particles best positions function/fitness value.
					* w (numpy.ndarray): Inertial weight.
					* vMin (numpy.ndarray): Minimal velocity.
					* vMax (numpy.ndarray): Maximal velocity.
					* V (numpy.ndarray): Initial velocity of particle.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		pop, fpop, d = Algorithm.initPopulation(self, task)
		d.update(self.init(task))
		d.update({'popb': pop, 'fpopb': fpop})
		return pop, fpop, d

	def runIteration(self, task, pop, fpop, xb, fxb, popb, fpopb, w, vMin, vMax, V, **dparams):
		r"""Core function of Particle Swarm Optimization algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current populations.
			fpop (numpy.ndarray[float]): Current population fitness/function values.
			xb (numpy.ndarray): Current best particle.
			fxb (float): Current best particle fitness/function value.
			popb (numpy.ndarray): Particles best position.
			fpopb (numpy.ndarray[float]): Particles best positions fitness/function values.
			w (numpy.ndarray): Inertial weights.
			vMin (numpy.ndarray): Minimal velocity.
			vMax (numpy.ndarray): Maximal velocity.
			V (numpy.ndarray): Velocity of particles.
			**dparams (Dict[str, Any]): Additional function arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population.
				2. New population fitness/function values.
				3. Additional arguments:
					* popb (numpy.ndarray): Particles best population.
					* fpopb (numpy.ndarray[float]): Particles best positions function/fitness value.
					* w (numpy.ndarray): Inertial weight.
					* vMin (numpy.ndarray): Minimal velocity.
					* vMax (numpy.ndarray): Maximal velocity.
					* V (numpy.ndarray): Initial velocity of particle.
		"""
		V = w * V + self.C1 * self.rand([self.NP, task.D]) * (popb - pop) + self.C2 * self.rand([self.NP, task.D]) * (xb - pop)
		V = apply_along_axis(self.repair, 1, V, vMin, vMax)
		pop = apply_along_axis(task.repair, 1, pop + V, rnd=self.Rand)
		fpop = apply_along_axis(task.eval, 1, pop)
		ip_pb = where(fpop < fpopb)
		popb[ip_pb], fpopb[ip_pb] = pop[ip_pb], fpop[ip_pb]
		return pop, fpop, {'popb': popb, 'fpopb': fpopb, 'w': w, 'vMin': vMin, 'vMax': vMax, 'V': V}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
