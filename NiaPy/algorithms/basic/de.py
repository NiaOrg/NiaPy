# encoding=utf8
import copy
import logging
from NiaPy.algorithms.algorithm import Algorithm

__all__ = ['DifferentialEvolutionAlgorithm']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

class SolutionDE(object):
	def __init__(self, task):
		self.Solution = []
		self.Fitness = float('inf')
		self.generateSolution(task)

	def generateSolution(self): self.Solution = [self.LB + (self.UB - self.LB) * rnd.random() for _i in range(self.D)]

	def evaluate(self, task): self.Fitness = task.eval(self.Solution)

	def repair(self, task):
		ir = where(self.Solution > task.Upper)
		self.Solution[ir] = task.Upper[ir]
		ir = where(self.Solution < task.Lower)
		self.Solution[ir] = task.Lower[ir]

	def __eq__(self, other): return self.Solution == other.Solution and self.Fitness == other.Fitness

class DifferentialEvolutionAlgorithm(Algorithm):
	r"""Implementation of Differential evolution algorithm.

	**Algorithm:** Differential evolution algorithm

	**Date:** 2018

	**Author:** Uros Mlakar

	**License:** MIT

	**Reference paper:**
	Storn, Rainer, and Kenneth Price. "Differential evolution - a simple and
	efficient heuristic for global optimization over continuous spaces."
	Journal of global optimization 11.4 (1997): 341-359.
	"""

	def __init__(self, D, NP, nFES, F, CR, benchmark):
		r"""**__init__(self, D, NP, nFES, F, CR, benchmark)**.

				Raises:
		TypeError -- Raised when given benchmark function which does not exists.
		"""
		super(DifferentialEvolutionAlgorithm, self).__init__(kwargs)

	def setParameters(self, **kwargs): self.__setParams(**kwargs)

	def __setParams(self, NP=25, F=2, CR=0.2, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
		NP {integer} -- population size
		F {decimal} -- scaling factor
		CR {decimal} -- crossover rate
		"""
		self.Np = NP  # population size
		self.F = F  # scaling factor
		self.CR = CR  # crossover rate
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def evalPopulation(self, x):
		"""Evaluate population."""
		for p in self.Population:
			p.evaluate()
			if p.Fitness < self.bestSolution.Fitness: self.bestSolution = copy.deepcopy(p)

	def generationStep(self, Population):
		"""Implement main generation step."""

        newPopulation = []
        for i in range(self.Np):
            newSolution = SolutionDE(self.D, self.Lower, self.Upper)

            r = rnd.sample(range(0, self.Np), 3)
            while i in r:
                r = rnd.sample(range(0, self.Np), 3)
            jrand = int(rnd.random() * self.Np)

            for j in range(self.D):
                if rnd.random() < self.CR or j == jrand:
                    newSolution.Solution[j] = Population[r[0]].Solution[j] + self.F * \
                        (Population[r[1]].Solution[j] - Population[r[2]].Solution[j])
                else:
                    newSolution.Solution[j] = Population[i].Solution[j]
            newSolution.repair()
            self.tryEval(newSolution)

            if newSolution.Fitness < self.bestSolution.Fitness:
                self.bestSolution = copy.deepcopy(newSolution)
            if newSolution.Fitness < self.Population[i].Fitness:
                newPopulation.append(newSolution)
            else:
                newPopulation.append(Population[i])
        return newPopulation

	def runTask(self, taks):
		"""Run."""
		pop = self.rand.uniform(task.Lower, task.Upper, [self.Np, task.D])
		self.evalPopulation(task)
		while not task.stopCond(): 
			self.Population = self.generationStep(self.Population)
		return self.bestSolution.Solution, self.bestSolution.Fitness

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
