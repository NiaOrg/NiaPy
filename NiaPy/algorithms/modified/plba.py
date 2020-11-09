# encoding=utf8
import logging

from numpy import full

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['ParameterFreeBatAlgorithm']

class ParameterFreeBatAlgorithm(Algorithm):
	r"""Implementation of Parameter-free Bat algorithm.

	Algorithm:
		Parameter-free Bat algorithm

	Date:
		2020

	Authors:
		Iztok Fister Jr.
		This implementation is based on the implementation of basic BA from NiaPy

	License:
		MIT

	Reference paper:
		Iztok Fister Jr., Iztok Fister, Xin-She Yang. Towards the development of a parameter-free bat algorithm . In: FISTER Jr., Iztok (Ed.), BRODNIK, Andrej (Ed.). StuCoSReC : proceedings of the 2015 2nd Student Computer Science Research Conference. Koper: University of Primorska, 2015, pp. 31-34.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['ParameterFreeBatAlgorithm', 'PLBA']

	@staticmethod
	def algorithmInfo():
		r"""Get algorithms information.

		Returns:
			str: Algorithm information.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Iztok Fister Jr., Iztok Fister, Xin-She Yang. Towards the development of a parameter-free bat algorithm . In: FISTER, Iztok (Ed.), BRODNIK, Andrej (Ed.). StuCoSReC : proceedings of the 2015 2nd Student Computer Science Research Conference. Koper: University of Primorska, 2015, pp. 31-34."""

	def setParameters(self, **ukwargs):
		r"""Set the parameters of the algorithm.

		Args:
			A (Optional[float]): Loudness.
			r (Optional[float]): Pulse rate.
		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=80, **ukwargs)
		self.A, self.r = 0.9, 0.1

	def initPopulation(self, task):
		r"""Initialize the initial population.

		Parameters:
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population.
				2. New population fitness/function values.
				3. Additional arguments:
					* S (numpy.ndarray): Solutions
					* Q (numpy.ndarray[float]): Frequencies
					* v (numpy.ndarray[float]): Velocities

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		Sol, Fitness, d = Algorithm.initPopulation(self, task)
		S, Q, v = full([self.NP, task.D], 0.0), full(self.NP, 0.0), full([self.NP, task.D], 0.0)
		d.update({'S': S, 'Q': Q, 'v': v})
		return Sol, Fitness, d

	def localSearch(self, best, task, **kwargs):
		r"""Improve the best solution according to the Yang (2010).

		Args:
			best (numpy.ndarray): Global best individual.
			task (Task): Optimization task.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			numpy.ndarray: New solution based on global best individual.
		"""
		return task.repair(best + 0.001 * self.normal(0, 1, task.D))

	def runIteration(self, task, Sol, Fitness, xb, fxb, S, Q, v, **dparams):
		r"""Core function of Parameter-free Bat Algorithm.

		Parameters:
			task (Task): Optimization task.
			Sol (numpy.ndarray): Current population
			Fitness (numpy.ndarray[float]): Current population fitness/funciton values
			best (numpy.ndarray): Current best individual
			f_min (float): Current best individual function/fitness value
			S (numpy.ndarray): Solutions
			Q (numpy.ndarray): Frequencies
			v (numpy.ndarray): Velocities
			best (numpy.ndarray): Global best used by the algorithm
			f_min (float): Global best fitness value used by the algorithm
			dparams (Dict[str, Any]): Additional algorithm arguments

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population
				2. New population fitness/function vlues
				3. New global best solution
				4. New global best fitness/objective value
				5. Additional arguments:
					* S (numpy.ndarray): Solutions
					* Q (numpy.ndarray): Frequencies
					* v (numpy.ndarray): Velocities
					* best (numpy.ndarray): Global best
					* f_min (float): Global best fitness
		"""
		upper, lower = task.bcUpper(), task.bcLower()
		for i in range(self.NP):
			Q[i] = ((upper[0] - lower[0]) / float(self.NP)) * self.normal(0, 1)
			v[i] += (Sol[i] - xb) * Q[i]
			if self.rand() > self.r: S[i] = self.localSearch(best=xb, task=task, i=i, Sol=Sol)
			else: S[i] = task.repair(Sol[i] + v[i], rnd=self.Rand)
			Fnew = task.eval(S[i])
			if (Fnew <= Fitness[i]) and (self.rand() < self.A): Sol[i], Fitness[i] = S[i], Fnew
			if Fnew <= fxb: xb, fxb = S[i].copy(), Fnew
		return Sol, Fitness, xb, fxb, {'S': S, 'Q': Q, 'v': v}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
