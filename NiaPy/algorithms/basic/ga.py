# encoding=utf8
import random as rnd
import copy
from NiaPy.algorithms.algorithm import Algorithm

__all__ = ['GeneticAlgorithm']

class Chromosome(object):
	def __init__(self, D, LB, UB):
		self.D = D
		self.LB = LB
		self.UB = UB

		self.Solution = []
		self.Fitness = float('inf')
		self.generateSolution()

	def generateSolution(self): self.Solution = [self.LB + (self.UB - self.LB) * rnd.random() for _i in range(self.D)]

	def evaluate(self): self.Fitness = Chromosome.FuncEval(self.D, self.Solution)

	def repair(self):
		for i in range(self.D):
			if self.Solution[i] > self.UB: self.Solution[i] = self.UB
			if self.Solution[i] < self.LB: self.Solution[i] = self.LB

	def __eq__(self, other): return self.Solution == other.Solution and self.Fitness == other.Fitness

	def toString(self): print([i for i in self.Solution])

class GeneticAlgorithm(Algorithm):
	r"""Implementation of Genetic algorithm.

	**Algorithm:** Genetic algorithm

	**Date:** 2018

	**Author:** Uros Mlakar

	**License:** MIT
	"""
	def __init__(self, **kwargs): super(GeneticAlgorithm, self).__init__(name='GeneticAlgorithm', sName='GA', **kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, NP=25, Ts=5, Mr=0.25, gamma=0.2, **ukwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
		NP {integer} -- population size
		Ts {integer} -- tournament selection
		Mr {decimal} -- mutation rate
		gamma {decimal} -- minimum frequency
		"""
		self.NP, self.Ts, self.Mr, self.gamma = NP, Ts, Mr, gamma
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def TournamentSelection(self):
		"""Tournament selection."""
		indices = list(range(self.NP))
		rnd.shuffle(indices)
		tPop = []
		for i in range(self.Ts): tPop.append(self.Population[indices[i]])
		tPop.sort(key=lambda x: x.Fitness)
		self.Population.remove(tPop[0])
		self.Population.remove(tPop[1])
		return tPop[0], tPop[1]

	def CrossOver(self, parent1, parent2):
		"""Crossover."""
		alpha = [-self.gamma + (1 + 2 * self.gamma) * rnd.random() for i in range(self.D)]
		child1 = Chromosome(self.D, self.Lower, self.Upper)
		child2 = Chromosome(self.D, self.Lower, self.Upper)
		child1.Solution = [alpha[i] * parent1.Solution[i] + (1 - alpha[i]) * parent2.Solution[i] for i in range(self.D)]
		child2.Solution = [alpha[i] * parent2.Solution[i] + (1 - alpha[i]) * parent1.Solution[i] for i in range(self.D)]
		return child1, child2

	def Mutate(self, child):
		"""Mutation."""
		for i in range(self.D):
			if rnd.random() < self.Mr:
				sigma = 0.20 * float(child.UB - child.LB)
				child.Solution[i] = min(max(rnd.gauss(child.Solution[i], sigma), child.LB), child.UB)

	def runTask(self, task):
		"""Run."""
		self.init()
		while not task.stopCond():
			for _k in range(int(self.NP / 2)):
				parent1, parent2 = self.TournamentSelection()
				child1, child2 = self.CrossOver(parent1, parent2)
				self.Mutate(child1)
				self.Mutate(child2)
				child1.repair()
				child2.repair()
				self.tryEval(child1)
				self.tryEval(child2)
				tPop = [parent1, parent2, child1, child2]
				tPop.sort(key=lambda x: x.Fitness)
				self.Population.append(tPop[0])
				self.Population.append(tPop[1])
			for i in range(self.NP): self.checkForBest(self.Population[i])
		return self.Best.Fitness

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
