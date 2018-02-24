import random as rnd
import copy
from NiaPy.benchmarks.utility import Utility

__all__ = ['DifferentialEvolutionAlgorithm']


class SolutionDE(object):

    def __init__(self, D, LB, UB):
        self.D = D
        self.LB = LB
        self.UB = UB

        self.Solution = []
        self.Fitness = float('inf')
        self.generateSolution()

    def generateSolution(self):
        self.Solution = [self.LB + (self.UB - self.LB) * rnd.random()
                         for _i in range(self.D)]

    def evaluate(self):
        self.Fitness = SolutionDE.FuncEval(self.D, self.Solution)

    def repair(self):
        for i in range(self.D):
            if self.Solution[i] > self.UB:
                self.Solution[i] = self.UB
            if self.Solution[i] < self.LB:
                self.Solution[i] = self.LB

    def __eq__(self, other):
        return self.Solution == other.Solution and self.Fitness == other.Fitness


class DifferentialEvolutionAlgorithm(object):
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

        Arguments:
            D {integer} -- dimension of problem

            NP {integer} -- population size

            nFES {integer} -- number of function evaluations

            F {decimal} -- scaling factor

            CR {decimal} -- crossover rate

            benchmark {object} -- benchmark implementation object

        Raises:
            TypeError -- Raised when given benchmark function which does not exists.

        """

        self.benchmark = Utility().get_benchmark(benchmark)
        self.D = D  # dimension of problem
        self.Np = NP  # population size
        self.nFES = nFES  # number of function evaluations
        self.F = F  # scaling factor
        self.CR = CR  # crossover rate
        self.Lower = self.benchmark.Lower  # lower bound
        self.Upper = self.benchmark.Upper  # upper bound

        SolutionDE.FuncEval = staticmethod(self.benchmark.function())
        self.Population = []
        self.bestSolution = SolutionDE(self.D, self.Lower, self.Upper)

    def evalPopulation(self):
        """Evaluate population."""

        for p in self.Population:
            p.evaluate()
            if p.Fitness < self.bestSolution.Fitness:
                self.bestSolution = copy.deepcopy(p)

    def initPopulation(self):
        """Initialize population."""

        for _i in range(self.Np):
            self.Population.append(SolutionDE(self.D, self.Lower, self.Upper))

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
                    newSolution.Solution[j] = Population[r[0]].Solution[j] + self.F * (Population[r[1]].Solution[j] - Population[r[2]].Solution[j])
                else:
                    newSolution.Solution[j] = Population[i].Solution[j]
            newSolution.repair()
            newSolution.evaluate()

            if newSolution.Fitness < self.bestSolution.Fitness:
                self.bestSolution = copy.deepcopy(newSolution)
            if newSolution.Fitness < self.Population[i].Fitness:
                newPopulation.append(newSolution)
            else:
                newPopulation.append(Population[i])
        return newPopulation

    def run(self):
        self.initPopulation()
        self.evalPopulation()
        FEs = self.Np
        while FEs <= self.nFES:
            self.Population = self.generationStep(self.Population)
            FEs += self.Np
        return self.bestSolution.Fitness
