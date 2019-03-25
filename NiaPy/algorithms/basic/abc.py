# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, multiple-statements, attribute-defined-outside-init, logging-not-lazy, arguments-differ, bad-continuation
import copy
import logging
from NiaPy.algorithms.algorithm import Algorithm, Individual

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['ArtificialBeeColonyAlgorithm']

class SolutionABC(Individual):
	def __init__(self, task, rand): Individual.__init__(self, task=task, rand=rand)

class ArtificialBeeColonyAlgorithm(Algorithm):
	r"""Implementation of Artificial Bee Colony algorithm.

	**Algorithm:** Artificial Bee Colony algorithm

	**Date:** 2018

	**Author:** Uros Mlakar and Klemen Berkovič

	**License:** MIT

	**Reference paper:**
	Karaboga, D., and Bahriye B. "A powerful and efficient algorithm for
	numerical function optimization: artificial bee colony (ABC) algorithm."
	Journal of global optimization 39.3 (2007): 459-471.
	"""
	Name = ['ArtificialBeeColonyAlgorithm', 'ABC']

	@staticmethod
	def typeParameters(): return {
			'NP': lambda x: isinstance(x, int) and x > 0,
			'Limit': lambda x: isinstance(x, int) and x > 0
	}

	def setParameters(self, NP=10, Limit=100, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments:**

		NP {integer} -- population size

		Limit {integer} -- Limit
		"""
		self.NP = NP  # population size; number of search agents
		self.FoodNumber = int(self.NP / 2)
		self.Limit = Limit

		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def CalculateProbs(self, Foods, Probs):
		"""Calculate probs."""
		Probs = [1.0 / (Foods[i].f + 0.01) for i in range(self.FoodNumber)]
		s = sum(Probs)
		Probs = [Probs[i] / s for i in range(self.FoodNumber)]
		return Probs

	def initPopulation(self, task):
		"""Initialize positions."""
		Foods, Probs, Trial = [], [0 for i in range(self.FoodNumber)], [0 for i in range(self.FoodNumber)]
		# self.Best = SolutionABC(task, self.Rand)
		for i in range(self.FoodNumber): Foods.append(SolutionABC(task, self.Rand))
		return Foods, [f.f for f in Foods], {'Probs':Probs, 'Trial':Trial}

	def runIteration(self, task, Foods, fpop, xb, fxb, Probs, Trial, **dparams):
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
		mi = Trial.index(max(Trial))
		if Trial[mi] >= self.Limit: Foods[mi], Trial[mi] = SolutionABC(task, self.Rand), 0
		return Foods, [f.f for f in Foods], {'Probs':Probs, 'Trial':Trial}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
