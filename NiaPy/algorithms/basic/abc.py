# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy
import copy
import logging
from numpy import random as rnd, where
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.benchmarks.utility import Utility

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['ArtificialBeeColonyAlgorithm']

class SolutionABC(object):
	def __init__(self, task):
		self.Solution = None
		self.Fitness = float('inf')
		self.generateSolution(task)

	def generateSolution(self, task): self.Solution = task.Lower +  task.bRange * rnd.rand(task.D)

	def repair(self, task):
		ir = where(self.Solution > task.Upper)
		self.Solution[ir] = task.Upper[ir]
		ir = where(self.Solution < task.Lower)
		self.Solution[ir] = task.Lower[ir]

	def evaluate(self, task): self.Fitness = task.eval(self.Solution)

	def eval(self, task):
		self.repair(task)
		self.evaluate(task)

	def toString(self): pass

class ArtificialBeeColonyAlgorithm(Algorithm):
	r"""Implementation of Artificial Bee Colony algorithm.

	**Algorithm:** Artificial Bee Colony algorithm

	**Date:** 2018

	**Author:** Uros Mlakar

	**License:** MIT

	**Reference paper:**
	Karaboga, D., and Bahriye B. "A powerful and efficient algorithm for
	numerical function optimization: artificial bee colony (ABC) algorithm."
	Journal of global optimization 39.3 (2007): 459-471.
	"""
	def __init__(self, **kwargs):
		"""**__init__(self, D, NP, nFES, benchmark)**.

		Raises:
		TypeError -- Raised when given benchmark function which does not exists.
		"""
		super(ArtificialBeeColonyAlgorithm, self).__init__(name='ArtificialBeeColonyAlgorithm', sName='ABC', **kwargs)

	def setParameters(self, **kwargs):
		r"""Set the algorithm parameters/arguments.

		**See**:
		ArtificialBeeColonyAlgorithm.__setParams(self, NP, **kwargs)
		"""
		self.__setParams(**kwargs)

	def __setParams(self, NP=10, Limit=100, **kwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		NP {integer} -- population size
		Limit {integer} -- Limit
		"""
		self.NP = NP  # population size; number of search agents
		self.FoodNumber = int(self.NP / 2)
		self.Limit = Limit
		self.Trial = []  # trials
		self.Foods = []  # foods
		self.Probs = []  # probs

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
		self.Probs = [1.0 / (self.Foods[i].Fitness + 0.01)	for i in range(self.FoodNumber)]
		s = sum(self.Probs)
		self.Probs = [self.Probs[i] / s for i in range(self.FoodNumber)]

	def checkForBest(self, Solution):
		"""Check best solution."""
		if Solution.Fitness <= self.Best.Fitness: self.Best = copy.deepcopy(Solution)

	def runTask(self, task):
		"""Run."""
		self.init(task)
		while not task.stopCond():
			for i in range(self.FoodNumber):
				newSolution = copy.deepcopy(self.Foods[i])
				param2change = int(rnd.rand() * task.D)
				neighbor = int(self.FoodNumber * rnd.rand())
				newSolution.Solution[param2change] = self.Foods[i].Solution[param2change] + (-1 + 2 * rnd.rand()) * (self.Foods[i].Solution[param2change] - self.Foods[neighbor].Solution[param2change])
				newSolution.eval(task)
				if newSolution.Fitness < self.Foods[i].Fitness:
					self.checkForBest(newSolution)
					self.Foods[i] = newSolution
					self.Trial[i] = 0
				else: self.Trial[i] += 1
			self.CalculateProbs()
			t, s = 0, 0
			while t < self.FoodNumber:
				if rnd.rand() < self.Probs[s]:
					t += 1
					Solution = copy.deepcopy(self.Foods[s])
					param2change = int(rnd.rand() * task.D)
					neighbor = int(self.FoodNumber * rnd.rand())
					while neighbor == s: neighbor = int(self.FoodNumber * rnd.rand())
					Solution.Solution[param2change] = self.Foods[s].Solution[param2change] + (-1 + 2 * rnd.rand()) * (self.Foods[s].Solution[param2change] - self.Foods[neighbor].Solution[param2change])
					Solution.eval(task)
					if Solution.Fitness < self.Foods[s].Fitness:
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
		return self.Best.Solution, self.Best.Fitness

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
