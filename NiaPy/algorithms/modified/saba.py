# encoding=utf8
import logging

import numpy as np

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['AdaptiveBatAlgorithm', 'SelfAdaptiveBatAlgorithm']

class AdaptiveBatAlgorithm(Algorithm):
	r"""Implementation of Adaptive bat algorithm.

	Algorithm:
		Adaptive bat algorithm

	Date:
		April 2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		epsilon (float): Scaling factor.
		alpha (float): Constant for updating loudness.
		r (float): Pulse rate.
		Qmin (float): Minimum frequency.
		Qmax (float): Maximum frequency.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['AdaptiveBatAlgorithm', 'ABA']

	@staticmethod
	def typeParameters():
		r"""Return dict with where key of dict represents parameter name and values represent checking functions for selected parameter.

		Returns:
			Dict[str, Callable]:
				* epsilon (Callable[[Union[float, int]], bool]): Scale factor.
				* alpha (Callable[[Union[float, int]], bool]): Constant for updating loudness.
				* r (Callable[[Union[float, int]], bool]): Pulse rate.
				* Qmin (Callable[[Union[float, int]], bool]): Minimum frequency.
				* Qmax (Callable[[Union[float, int]], bool]): Maximum frequency.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'epsilon': lambda x: isinstance(x, (float, int)) and x > 0,
			'alpha': lambda x: isinstance(x, (float, int)) and x > 0,
			'r': lambda x: isinstance(x, (float, int)) and x > 0,
			'Qmin': lambda x: isinstance(x, (float, int)),
			'Qmax': lambda x: isinstance(x, (float, int))
		})
		return d

	def setParameters(self, NP=100, A=0.5, epsilon=0.001, alpha=1.0, r=0.5, Qmin=0.0, Qmax=2.0, **ukwargs):
		r"""Set the parameters of the algorithm.

		Args:
			A (Optional[float]): Starting loudness.
			epsilon (Optional[float]): Scaling factor.
			alpha (Optional[float]): Constant for updating loudness.
			r (Optional[float]): Pulse rate.
			Qmin (Optional[float]): Minimum frequency.
			Qmax (Optional[float]): Maximum frequency.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.A, self.epsilon, self.alpha, self.r, self.Qmin, self.Qmax = A, epsilon, alpha, r, Qmin, Qmax

	def getParameters(self):
		r"""Get algorithm parameters.

		Returns:
			Dict[str, Any]: Arguments values.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.getParameters`
		"""
		d = Algorithm.getParameters(self)
		d.update({
			'A': self.A,
			'epsilon': self.epsilon,
			'alpha': self.alpha,
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
					* A (float): Loudness.
					* S (numpy.ndarray): TODO
					* Q (numpy.ndarray[float]): 	TODO
					* v (numpy.ndarray[float]): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		Sol, Fitness, d = Algorithm.initPopulation(self, task)
		A, S, Q, v = np.full(self.NP, self.A), np.full([self.NP, task.D], 0.0), np.full(self.NP, 0.0), np.full([self.NP, task.D], 0.0)
		d.update({'A': A, 'S': S, 'Q': Q, 'v': v})
		return Sol, Fitness, d

	def localSearch(self, best, A, task, **kwargs):
		r"""Improve the best solution according to the Yang (2010).

		Args:
			best (numpy.ndarray): Global best individual.
			A (float): Loudness.
			task (Task): Optimization task.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			numpy.ndarray: New solution based on global best individual.
		"""
		return task.repair(best + self.epsilon * A * self.normal(0, 1, task.D), rnd=self.Rand)

	def updateLoudness(self, A):
		r"""Update loudness when the prey is found.

		Args:
			A (float): Loudness.

		Returns:
			float: New loudness.
		"""
		nA = A * self.alpha
		return nA if nA > 1e-13 else self.A

	def runIteration(self, task, Sol, Fitness, xb, fxb, A, S, Q, v, **dparams):
		r"""Core function of Bat Algorithm.

		Parameters:
			task (Task): Optimization task.
			Sol (numpy.ndarray): Current population
			Fitness (numpy.ndarray[float]): Current population fitness/funciton values
			best (numpy.ndarray): Current best individual
			f_min (float): Current best individual function/fitness value
			S (numpy.ndarray): TODO
			Q (numpy.ndarray[float]): TODO
			v (numpy.ndarray[float]): TODO
			dparams (Dict[str, Any]): Additional algorithm arguments

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population
				2. New population fitness/function vlues
				3. Additional arguments:
					* A (numpy.ndarray[float]): Loudness.
					* S (numpy.ndarray): TODO
					* Q (numpy.ndarray[float]): TODO
					* v (numpy.ndarray[float]): TODO
		"""
		for i in range(self.NP):
			Q[i] = self.Qmin + (self.Qmax - self.Qmin) * self.uniform(0, 1)
			v[i] += (Sol[i] - xb) * Q[i]
			if self.rand() > self.r: S[i] = self.localSearch(best=xb, A=A[i], task=task, i=i, Sol=Sol)
			else: S[i] = task.repair(Sol[i] + v[i], rnd=self.Rand)
			Fnew = task.eval(S[i])
			if (Fnew <= Fitness[i]) and (self.rand() < A[i]): Sol[i], Fitness[i] = S[i], Fnew
			if Fnew <= fxb: xb, fxb, A[i] = S[i].copy(), Fnew, self.updateLoudness(A[i])
		return Sol, Fitness, xb, fxb, {'A': A, 'S': S, 'Q': Q, 'v': v}

class SelfAdaptiveBatAlgorithm(AdaptiveBatAlgorithm):
	r"""Implementation of Hybrid bat algorithm.

	Algorithm:
		Hybrid bat algorithm

	Date:
		April 2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		Fister Jr., Iztok and Fister, Dusan and Yang, Xin-She. "A Hybrid Bat Algorithm". Elektrotehniski vestnik, 2013. 1-7.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		A_l (Optional[float]): Lower limit of loudness.
		A_u (Optional[float]): Upper limit of loudness.
		r_l (Optional[float]): Lower limit of pulse rate.
		r_u (Optional[float]): Upper limit of pulse rate.
		tao_1 (Optional[float]): Learning rate for loudness.
		tao_2 (Optional[float]): Learning rate for pulse rate.

	See Also:
		* :class:`NiaPy.algorithms.basic.BatAlgorithm`
	"""
	Name = ['SelfAdaptiveBatAlgorithm', 'SABA']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information about the algorithm.

		Returns:
			str: Basic information.
		"""
		return r"""Fister Jr., Iztok and Fister, Dusan and Yang, Xin-She. "A Hybrid Bat Algorithm". Elektrotehniski vestnik, 2013. 1-7."""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]: TODO

		See Also:
			* :func:`NiaPy.algorithms.basic.BatAlgorithm.typeParameters`
		"""
		d = AdaptiveBatAlgorithm.typeParameters()
		d.pop('A', None), d.pop('r', None)
		d.update({
			'A_l': lambda x: isinstance(x, (float, int)) and x >= 0,
			'A_u': lambda x: isinstance(x, (float, int)) and x >= 0,
			'r_l': lambda x: isinstance(x, (float, int)) and x >= 0,
			'r_u': lambda x: isinstance(x, (float, int)) and x >= 0,
			'tao_1': lambda x: isinstance(x, (float, int)) and 0 <= x <= 1,
			'tao_2': lambda x: isinstance(x, (float, int)) and 0 <= x <= 1
		})
		return d

	def setParameters(self, A_l=0.9, A_u=1.0, r_l=0.001, r_u=0.1, tao_1=0.1, tao_2=0.1, **ukwargs):
		r"""Set core parameters of HybridBatAlgorithm algorithm.

		Arguments:
			A_l (Optional[float]): Lower limit of loudness.
			A_u (Optional[float]): Upper limit of loudness.
			r_l (Optional[float]): Lower limit of pulse rate.
			r_u (Optional[float]): Upper limit of pulse rate.
			tao_1 (Optional[float]): Learning rate for loudness.
			tao_2 (Optional[float]): Learning rate for pulse rate.

		See Also:
			* :func:`NiaPy.algorithms.modified.AdaptiveBatAlgorithm.setParameters`
		"""
		AdaptiveBatAlgorithm.setParameters(self, **ukwargs)
		self.A_l, self.A_u, self.r_l, self.r_u, self.tao_1, self.tao_2 = A_l, A_u, r_l, r_u, tao_1, tao_2

	def getParameters(self):
		r"""Get parameters of the algorithm.

		Returns:
			Dict[str, Any]: Parameters of the algorithm.

		See Also:
			* :func:`NiaPy.algorithms.modified.AdaptiveBatAlgorithm.getParameters`
		"""
		d = AdaptiveBatAlgorithm.getParameters(self)
		d.update({
			'A_l': self.A_l,
			'A_u': self.A_u,
			'r_l': self.r_l,
			'r_u': self.r_u,
			'tao_1': self.tao_1,
			'tao_2': self.tao_2
		})
		return d

	def initPopulation(self, task):
		Sol, Fitness, d = AdaptiveBatAlgorithm.initPopulation(self, task)
		A, r = np.full(self.NP, self.A), np.full(self.NP, self.r)
		d.update({'A': A, 'r': r})
		return Sol, Fitness, d

	def selfAdaptation(self, A, r):
		r"""Adaptation step.

		Args:
			A (float): Current loudness.
			r (float): Current pulse rate.

		Returns:
			Tuple[float, float]:
				1. New loudness.
				2. Nwq pulse rate.
		"""
		return self.A_l + self.rand() * (self.A_u - self.A_l) if self.rand() < self.tao_1 else A, self.r_l + self.rand() * (self.r_u - self.r_l) if self.rand() < self.tao_2 else r

	def runIteration(self, task, Sol, Fitness, xb, fxb, A, r, S, Q, v, **dparams):
		r"""Core function of Bat Algorithm.

		Parameters:
			task (Task): Optimization task.
			Sol (numpy.ndarray): Current population
			Fitness (numpy.ndarray[float]): Current population fitness/funciton values
			xb (numpy.ndarray): Current best individual
			fxb (float): Current best individual function/fitness value
			A (numpy.ndarray[flaot]): Loudness of individuals.
			r (numpy.ndarray[float[): Pulse rate of individuals.
			S (numpy.ndarray): TODO
			Q (numpy.ndarray[float]): TODO
			v (numpy.ndarray[float]): TODO
			dparams (Dict[str, Any]): Additional algorithm arguments

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population
				2. New population fitness/function vlues
				3. Additional arguments:
					* A (numpy.ndarray[float]): Loudness.
					* r (numpy.ndarray[float]): Pulse rate.
					* S (numpy.ndarray): TODO
					* Q (numpy.ndarray[float]): TODO
					* v (numpy.ndarray[float]): TODO
		"""
		for i in range(self.NP):
			A[i], r[i] = self.selfAdaptation(A[i], r[i])
			Q[i] = self.Qmin + (self.Qmax - self.Qmin) * self.uniform(0, 1)
			v[i] += (Sol[i] - xb) * Q[i]
			if self.rand() > r[i]: S[i] = self.localSearch(best=xb, A=A[i], task=task, i=i, Sol=Sol)
			else: S[i] = task.repair(Sol[i] + v[i], rnd=self.Rand)
			Fnew = task.eval(S[i])
			if (Fnew <= Fitness[i]) and (self.rand() < (self.A_l - A[i]) / self.A): Sol[i], Fitness[i] = S[i], Fnew
			if Fnew <= fxb: xb, fxb = S[i].copy(), Fnew
		return Sol, Fitness, xb, fxb, {'A': A, 'r': r, 'S': S, 'Q': Q, 'v': v}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
