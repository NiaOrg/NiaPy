# encoding=utf8
import logging

from numpy import full

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['BatAlgorithm']

class BatAlgorithm(Algorithm):
	r"""Implementation of Bat algorithm.

	Algorithm:
		Bat algorithm

	Date:
		2015

	Authors:
		Iztok Fister Jr., Marko Burjek and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Yang, Xin-She. "A new metaheuristic bat-inspired algorithm." Nature inspired cooperative strategies for optimization (NICSO 2010). Springer, Berlin, Heidelberg, 2010. 65-74.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		A (float): Loudness.
		r (float): Pulse rate.
		Qmin (float): Minimum frequency.
		Qmax (float): Maximum frequency.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['BatAlgorithm', 'BA']

	@staticmethod
	def algorithmInfo():
		r"""Get algorithms information.

		Returns:
			str: Algorithm information.
		"""
		return r'''Yang, Xin-She. "A new metaheuristic bat-inspired algorithm." Nature inspired cooperative strategies for optimization (NICSO 2010). Springer, Berlin, Heidelberg, 2010. 65-74.'''

	@staticmethod
	def typeParameters():
		r"""Return dict with where key of dict represents parameter name and values represent checking functions for selected parameter.

		Returns:
			Dict[str, Callable]:
				* A (Callable[[Union[float, int]], bool]): Loudness.
				* r (Callable[[Union[float, int]], bool]): Pulse rate.
				* Qmin (Callable[[Union[float, int]], bool]): Minimum frequency.
				* Qmax (Callable[[Union[float, int]], bool]): Maximum frequency.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'A': lambda x: isinstance(x, (float, int)) and x > 0,
			'r': lambda x: isinstance(x, (float, int)) and x > 0,
			'Qmin': lambda x: isinstance(x, (float, int)),
			'Qmax': lambda x: isinstance(x, (float, int))
		})
		return d

	def setParameters(self, NP=40, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, **ukwargs):
		r"""Set the parameters of the algorithm.

		Args:
			A (Optional[float]): Loudness.
			r (Optional[float]): Pulse rate.
			Qmin (Optional[float]): Minimum frequency.
			Qmax (Optional[float]): Maximum frequency.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.A, self.r, self.Qmin, self.Qmax = A, r, Qmin, Qmax

	def getParameters(self):
		r"""Get parameters of the algorithm.

		Returns:
			Dict[str, Any]
		"""
		d = Algorithm.getParameters(self)
		d.update({
			'A': self.A,
			'r': self.r,
			'Qmin': self.Qmin,
			'Qmax': self.Qmax
		})
		return d

	def initPopulation(self, task):
		r"""Initialize the starting population.

		Parameters:
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population.
				2. New population fitness/function values.
				3. Additional arguments:
					* S (numpy.ndarray): TODO
					* Q (numpy.ndarray[float]): 	TODO
					* v (numpy.ndarray[float]): TODO

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
		r"""Core function of Bat Algorithm.

		Parameters:
			task (Task): Optimization task.
			Sol (numpy.ndarray): Current population
			Fitness (numpy.ndarray[float]): Current population fitness/funciton values
			best (numpy.ndarray): Current best individual
			f_min (float): Current best individual function/fitness value
			S (numpy.ndarray): TODO
			Q (numpy.ndarray): TODO
			v (numpy.ndarray): TODO
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
					* S (numpy.ndarray): TODO
					* Q (numpy.ndarray): TODO
					* v (numpy.ndarray): TODO
					* best (numpy.ndarray): TODO
					* f_min (float): TODO
		"""
		for i in range(self.NP):
			Q[i] = self.Qmin + (self.Qmax - self.Qmin) * self.uniform(0, 1)
			v[i] += (Sol[i] - xb) * Q[i]
			if self.rand() > self.r: S[i] = self.localSearch(best=xb, task=task, i=i, Sol=Sol)
			else: S[i] = task.repair(Sol[i] + v[i], rnd=self.Rand)
			Fnew = task.eval(S[i])
			if (Fnew <= Fitness[i]) and (self.rand() < self.A): Sol[i], Fitness[i] = S[i], Fnew
			if Fnew <= fxb: xb, fxb = S[i].copy(), Fnew
		return Sol, Fitness, xb, fxb, {'S': S, 'Q': Q, 'v': v}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
