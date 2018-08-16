# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, multiple-statements, attribute-defined-outside-init, logging-not-lazy, arguments-differ
import copy
import logging
from NiaPy.algorithms.algorithm import Algorithm, Individual

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['ArtificialBeeColonyAlgorithm']

class SolutionABC(Individual):
	def __init__(self, task): Individual.__init__(self, task=task, e=False)

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
	def __init__(self, **kwargs):
		"""**__init__(self, D, NP, nFES, benchmark)**.

		**See**:
		Algorithm.__init__(self, **kwargs)
		"""
		Algorithm.__init__(self, name='ArtificialBeeColonyAlgorithm', sName='ABC', **kwargs)

	def setParameters(self, NP=10, Limit=100, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments:**

		NP {integer} -- population size

		Limit {integer} -- Limit
		"""
		self.NP = NP  # population size; number of search agents
		self.FoodNumber = int(self.NP / 2)
		self.Limit = Limit
		self.Trial = []  # trials
		self.Foods = []  # foods
		self.Probs = []  # probs
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def init(self, task):
		"""Initialize positions."""
		self.Probs = [0 for i in range(self.FoodNumber)]
		self.Trial = [0 for i in range(self.FoodNumber)]
		self.Best = SolutionABC(task)
		for i in range(self.FoodNumber):
			self.Foods.append(SolutionABC(task))
			self.Foods[i].evaluate(task)
			self.checkForBest(self.Foods[i])

	def CalculateProbs(self):
		"""Calculate probs."""
		self.Probs = [1.0 / (self.Foods[i].f + 0.01)	for i in range(self.FoodNumber)]
		s = sum(self.Probs)
		self.Probs = [self.Probs[i] / s for i in range(self.FoodNumber)]

	def checkForBest(self, Solution):
		"""Check best solution."""
		if Solution.f <= self.Best.f: self.Best = copy.deepcopy(Solution)

	def runTask(self, task):
		"""Run."""
		self.init(task)
		while not task.stopCond():
			for i in range(self.FoodNumber):
				newSolution = copy.deepcopy(self.Foods[i])
				param2change = int(self.rand() * task.D)
				neighbor = int(self.FoodNumber * self.rand())
				newSolution.x[param2change] = self.Foods[i].x[param2change] + (-1 + 2 * self.rand()) * (self.Foods[i].x[param2change] - self.Foods[neighbor].x[param2change])
				newSolution.evaluate(task)
				if newSolution.f < self.Foods[i].f:
					self.checkForBest(newSolution)
					self.Foods[i] = newSolution
					self.Trial[i] = 0
				else: self.Trial[i] += 1
			self.CalculateProbs()
			t, s = 0, 0
			while t < self.FoodNumber:
				if self.rand() < self.Probs[s]:
					t += 1
					Solution = copy.deepcopy(self.Foods[s])
					param2change = int(self.rand() * task.D)
					neighbor = int(self.FoodNumber * self.rand())
					while neighbor == s: neighbor = int(self.FoodNumber * self.rand())
					Solution.x[param2change] = self.Foods[s].x[param2change] + (-1 + 2 * self.rand()) * (self.Foods[s].x[param2change] - self.Foods[neighbor].x[param2change])
					Solution.evaluate(task)
					if Solution.f < self.Foods[s].f:
						self.checkForBest(newSolution)
						self.Foods[s] = Solution
						self.Trial[s] = 0
					else: self.Trial[s] += 1
				s += 1
				if s == self.FoodNumber: s = 0
			mi = self.Trial.index(max(self.Trial))
			if self.Trial[mi] >= self.Limit:
				self.Foods[mi] = SolutionABC(task)
				self.Foods[mi].evaluate(task)
				self.Trial[mi] = 0
		return self.Best.x, self.Best.f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
