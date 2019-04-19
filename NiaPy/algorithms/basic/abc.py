# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, multiple-statements, attribute-defined-outside-init, logging-not-lazy, arguments-differ, bad-continuation
import copy
import logging

from numpy import asarray, full, argmax

from NiaPy.algorithms.algorithm import Algorithm, Individual, default_individual_init

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['ArtificialBeeColonyAlgorithm']

class SolutionABC(Individual):
	r"""Representation of solution for Artificial Bee Colony Algorithm.

	Date:
		2018

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.Individual`
	"""
	def __init__(self, **kargs):
		r"""Initialize individual.

		Args:
			kargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Individual.__init__`
		"""
		Individual.__init__(self, **kargs)

class ArtificialBeeColonyAlgorithm(Algorithm):
	r"""Implementation of Artificial Bee Colony algorithm.

	Algorithm:
		Artificial Bee Colony algorithm

	Date:
		2018

	Author:
		Uros Mlakar and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Karaboga, D., and Bahriye B. "A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm." Journal of global optimization 39.3 (2007): 459-471.

	Arguments
		name (List[str]): List containing strings that represent algorithm names
		Limit (Union[float, numpy.ndarray[float]]): Limt

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	name = ['ArtificialBeeColonyAlgorithm', 'ABC']

	@staticmethod
	def parameter_types():
		r"""Return functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* Limit (Callable[Union[float, numpy.ndarray[float]]]): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.parameter_types`
		"""
		d = Algorithm.parameter_types()
		d.update({'Limit': lambda x: isinstance(x, int) and x > 0})
		return d

	def set_parameters(self, NP=10, Limit=100, **ukwargs):
		r"""Set the parameters of Artificial Bee Colony Algorithm.

		Parameters:
			Limit (Optional[Union[float, numpy.ndarray[float]]]): Limt
			**ukwargs (Dict[str, Any]): Additional arguments

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.set_parameters`
		"""
		Algorithm.set_parameters(self, NP=NP, InitPopFunc=default_individual_init, itype=SolutionABC, **ukwargs)
		self.FoodNumber, self.Limit = int(self.NP / 2), Limit
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def CalculateProbs(self, Foods, Probs):
		r"""Calculate the probes.

		Parameters:
			Foods (numpy.ndarray): TODO
			Probs (numpy.ndarray): TODO

		Returns:
			numpy.ndarray: TODO
		"""
		Probs = [1.0 / (Foods[i].f + 0.01) for i in range(self.FoodNumber)]
		s = sum(Probs)
		Probs = [Probs[i] / s for i in range(self.FoodNumber)]
		return Probs

	def init_population(self, task):
		r"""Initialize the starting population.

		Parameters:
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population
				2. New population fitness/function values
				3. Additional arguments:
					* Probes (numpy.ndarray): TODO
					* Trial (numpy.ndarray): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.init_population`
		"""
		Foods, fpop, _ = Algorithm.init_population(self, task)
		Probs, Trial = full(self.FoodNumber, 0.0), full(self.FoodNumber, 0.0)
		return Foods, fpop, {'Probs': Probs, 'Trial': Trial}

	def run_iteration(self, task, Foods, fpop, xb, fxb, Probs, Trial, **dparams):
		r"""Core funciton of  the algorithm.

		Parameters:
			task (Task): Optimization task
			Foods (numpy.ndarray): Current population
			fpop (numpy.ndarray[float]): Function/fitness values of current population
			xb (numpy.ndarray): Current best individual
			fxb (float): Current best individual fitness/function value
			Probs (numpy.ndarray): TODO
			Trial (numpy.ndarray): TODO
			dparams (Dict[str, Any]): Additional parameters

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population
				2. New population fitness/function values
				3. Additional arguments:
					* Probes (numpy.ndarray): TODO
					* Trial (numpy.ndarray): TODO
		"""
		for i in range(self.FoodNumber):
			newSolution = copy.deepcopy(Foods[i])
			param2change = int(self.rand() * task.D)
			neighbor = int(self.FoodNumber * self.rand())
			newSolution.x[param2change] = Foods[i].x[param2change] + (-1 + 2 * self.rand()) * (Foods[i].x[param2change] - Foods[neighbor].x[param2change])
			newSolution.evaluate(task, rnd=self.Rand)
			if newSolution.f < Foods[i].f: Foods[i], Trial[i] = newSolution, 0
			else: Trial[i] += 1
		Probs, t, s = self.CalculateProbs(Foods, Probs), 0, 0
		while t < self.FoodNumber:
			if self.rand() < Probs[s]:
				t += 1
				Solution = copy.deepcopy(Foods[s])
				param2change = int(self.rand() * task.D)
				neighbor = int(self.FoodNumber * self.rand())
				while neighbor == s: neighbor = int(self.FoodNumber * self.rand())
				Solution.x[param2change] = Foods[s].x[param2change] + (-1 + 2 * self.rand()) * (Foods[s].x[param2change] - Foods[neighbor].x[param2change])
				Solution.evaluate(task, rnd=self.Rand)
				if Solution.f < Foods[s].f: Foods[s], Trial[s] = Solution, 0
				else: Trial[s] += 1
			s += 1
			if s == self.FoodNumber: s = 0
		mi = argmax(Trial)
		if Trial[mi] >= self.Limit: Foods[mi], Trial[mi] = SolutionABC(task=task, rnd=self.Rand), 0
		return Foods, asarray([f.f for f in Foods]), {'Probs': Probs, 'Trial': Trial}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
