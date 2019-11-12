# encoding=utf8
import logging

from scipy.special import gamma as Gamma
from numpy import where, sin, fabs, pi, zeros

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['FlowerPollinationAlgorithm']

class FlowerPollinationAlgorithm(Algorithm):
	r"""Implementation of Flower Pollination algorithm.

	Algorithm:
		Flower Pollination algorithm

	Date:
		2018

	Authors:
		Dusan Fister, Iztok Fister Jr. and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Yang, Xin-She. "Flower pollination algorithm for global optimization. International conference on unconventional computing and natural computation. Springer, Berlin, Heidelberg, 2012.

	References URL:
		Implementation is based on the following MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/45112-flower-pollination-algorithm?requestedDomain=true

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		p (float): probability switch.
		beta (float): Shape of the gamma distribution (should be greater than zero).

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['FlowerPollinationAlgorithm', 'FPA']

	@staticmethod
	def typeParameters():
		r"""TODO.

		Returns:
			Dict[str, Callable]:
				* p (function): TODO
				* beta (function): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'p': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'beta': lambda x: isinstance(x, (float, int)) and x > 0,
		})
		return d

	def setParameters(self, NP=25, p=0.35, beta=1.5, **ukwargs):
		r"""Set core parameters of FlowerPollinationAlgorithm algorithm.

		Arguments:
			NP (int): Population size.
			p (float): Probability switch.
			beta (float): Shape of the gamma distribution (should be greater than zero).

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.p, self.beta = p, beta
		self.S = zeros((NP, 10))

	def repair(self, x, task):
		r"""Repair solution to search space.

		Args:
			x (numpy.ndarray): Solution to fix.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: fixed solution.
		"""
		ir = where(x > task.Upper)
		x[ir] = task.Lower[ir] + x[ir] % task.bRange[ir]
		ir = where(x < task.Lower)
		x[ir] = task.Lower[ir] + x[ir] % task.bRange[ir]
		return x

	def levy(self, D):
		r"""Levy function.

		Returns:
			float: Next Levy number.
		"""
		sigma = (Gamma(1 + self.beta) * sin(pi * self.beta / 2) / (Gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
		return 0.01 * (self.normal(0, 1, D) * sigma / fabs(self.normal(0, 1, D)) ** (1 / self.beta))

	def initPopulation(self, task):
		pop, fpop, d = Algorithm.initPopulation(self, task)
		d.update({'S': zeros((self.NP, task.D))})
		return pop, fpop, d

	def runIteration(self, task, Sol, Sol_f, xb, fxb, S, **dparams):
		r"""Core function of FlowerPollinationAlgorithm algorithm.

		Args:
			task (Task): Optimization task.
			Sol (numpy.ndarray): Current population.
			Sol_f (numpy.ndarray): Current population fitness/function values.
			xb (numpy.ndarray): Global best solution.
			fxb (float): Global best solution function/fitness value.
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population.
				2. New populations fitness/function values.
				3. New global best solution
				4. New global best solution fitness/objective value
				5. Additional arguments.
		"""
		for i in range(self.NP):
			if self.uniform(0, 1) > self.p: S[i] += self.levy(task.D) * (Sol[i] - xb)
			else:
				JK = self.Rand.permutation(self.NP)
				S[i] += self.uniform(0, 1) * (Sol[JK[0]] - Sol[JK[1]])
			S[i] = self.repair(S[i], task)
			f_i = task.eval(S[i])
			if f_i <= Sol_f[i]: Sol[i], Sol_f[i] = S[i], f_i
			if f_i <= fxb: xb, fxb = S[i].copy(), f_i
		return Sol, Sol_f, xb, fxb, {'S': S}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
