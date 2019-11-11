# encoding=utf8
import logging

from numpy import apply_along_axis, pi, fabs, sin, cos

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic.SineCosineAlgorithm')
logger.setLevel('INFO')

__all__ = ['SineCosineAlgorithm']

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
		Name (List[str]): List of string representing algorithm names.
		a (float): Parameter for control in :math:`r_1` value
		Rmin (float): Minimu value for :math:`r_3` value
		Rmax (float): Maximum value for :math:`r_3` value

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['SineCosineAlgorithm', 'SCA']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Seyedali Mirjalili, SCA: A Sine Cosine Algorithm for solving optimization problems, Knowledge-Based Systems, Volume 96, 2016, Pages 120-133, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2015.12.022."""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* a (Callable[[Union[float, int]], bool]): TODO
				* Rmin (Callable[[Union[float, int]], bool]): TODO
				* Rmax (Callable[[Union[float, int]], bool]): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'a': lambda x: isinstance(x, (float, int)) and x > 0,
			'Rmin': lambda x: isinstance(x, (float, int)),
			'Rmax': lambda x: isinstance(x, (float, int))
		})
		return d

	def setParameters(self, NP=25, a=3, Rmin=0, Rmax=2, **ukwargs):
		r"""Set the arguments of an algorithm.

		Args:
			NP (Optional[int]): Number of individual in population
			a (Optional[float]): Parameter for control in :math:`r_1` value
			Rmin (Optional[float]): Minimu value for :math:`r_3` value
			Rmax (Optional[float]): Maximum value for :math:`r_3` value

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.a, self.Rmin, self.Rmax = a, Rmin, Rmax

	def getParameters(self):
		r"""Get algorithm parameters values.

		Returns:
			Dict[str, Any]:

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.getParameters`
		"""
		d = Algorithm.getParameters(self)
		d.update({
			'a': self.a,
			'Rmin': self.Rmin,
			'Rmax': self.Rmax
		})
		return d

	def nextPos(self, x, x_b, r1, r2, r3, r4, task):
		r"""Move individual to new position in search space.

		Args:
			x (numpy.ndarray): Individual represented with components.
			x_b (nmppy.ndarray): Best individual represented with components.
			r1 (float): Number dependent on algorithm iteration/generations.
			r2 (float): Random number in range of 0 and 2 * PI.
			r3 (float): Random number in range [Rmin, Rmax].
			r4 (float): Random number in range [0, 1].
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: New individual that is moved based on individual ``x``.
		"""
		return task.repair(x + r1 * (sin(r2) if r4 < 0.5 else cos(r2)) * fabs(r3 * x_b - x), self.Rand)

	def initPopulation(self, task):
		r"""Initialize the individuals.

		Args:
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
				1. Initialized population of individuals
				2. Function/fitness values for individuals
				3. Additional arguments
		"""
		return Algorithm.initPopulation(self, task)

	def runIteration(self, task, P, P_f, xb, fxb, **dparams):
		r"""Core function of Sine Cosine Algorithm.

		Args:
			task (Task): Optimization task.
			P (numpy.ndarray): Current population individuals.
			P_f (numpy.ndarray[float]): Current population individulas function/fitness values.
			xb (numpy.ndarray): Current best solution to optimization task.
			fxb (float): Current best function/fitness value.
			dparams (Dict[str, Any]): Additional parameters.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population.
				2. New populations fitness/function values.
				3. New global best solution
				4. New global best fitness/objective value
				5. Additional arguments.
		"""
		r1, r2, r3, r4 = self.a - task.Iters * (self.a / task.Iters), self.uniform(0, 2 * pi), self.uniform(self.Rmin, self.Rmax), self.rand()
		P = apply_along_axis(self.nextPos, 1, P, xb, r1, r2, r3, r4, task)
		P_f = apply_along_axis(task.eval, 1, P)
		xb, fxb = self.getBest(P, P_f, xb, fxb)
		return P, P_f, xb, fxb, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
