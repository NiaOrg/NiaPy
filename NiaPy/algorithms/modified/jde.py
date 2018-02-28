import random as rnd
import copy
from NiaPy.benchmarks.utility import Utility

__all__ = ['SelfAdaptiveDifferentialEvolutionAlgorithm']


class SolutionjDE(object):

    def __init__(self, D, LB, UB, F, CR):
        self.D = D
        self.LB = LB
        self.UB = UB
        self.F = F
        self.CR = CR
        self.Solution = []
        self.Fitness = float('inf')
        self.generateSolution()

    def generateSolution(self):
        """Generate solution."""

        self.Solution = [self.LB + (self.UB - self.LB) * rnd.random()
                         for _i in range(self.D)]

    def evaluate(self):
        """Evaluate solution."""

        self.Fitness = SolutionjDE.FuncEval(self.D, self.Solution)

    def repair(self):
        for i in range(self.D):
            if self.Solution[i] > self.UB:
                self.Solution[i] = self.UB
            if self.Solution[i] < self.LB:
                self.Solution[i] = self.LB

    def __eq__(self, other):
        return self.Solution == other.Solution and self.Fitness == other.Fitness


class SelfAdaptiveDifferentialEvolutionAlgorithm(object):
    r"""Implementation of Self-adaptive differential evolution algorithm.

    **Algorithm:** Self-adaptive differential evolution algorithm

    **Date:** 2018

    **Author:** Uros Mlakar

    **License:** MIT

    **Reference paper:**
        Brest, J., Greiner, S., Boskovic, B., Mernik, M., Zumer, V.
        Self-adapting control parameters in differential evolution:
        A comparative study on numerical benchmark problems.
        IEEE transactions on evolutionary computation, 10(6), 646-657, 2006.
    """

    def __init__(self, D, NP, nFES, F, CR, Tao, benchmark):
        r"""**__init__(self, D, NP, nFES, F, CR, Tao, benchmark)**.

        Arguments:
            D {integer} -- dimension of problem

            NP {integer} -- population size

            nFES {integer} -- number of function evaluations

            F {decimal} -- scaling factor

            CR {decimal} -- crossover rate

            Tao {decimal}

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
        self.Tao = Tao
        self.Lower = self.benchmark.Lower  # lower bound
        self.Upper = self.benchmark.Upper  # upper bound

        SolutionjDE.FuncEval = staticmethod(self.benchmark.function())
        self.Population = []
        self.FEs = 0
        self.Done = False
        self.bestSolution = SolutionjDE(
            self.D,
            self.Lower,
            self.Upper,
            self.F,
            self.CR)

    def evalPopulation(self):
        """Evaluate population."""

        for p in self.Population:
            p.evaluate()
            if p.Fitness < self.bestSolution.Fitness:
                self.bestSolution = copy.deepcopy(p)

    def initPopulation(self):
        """Initialize population."""

        for _i in range(self.Np):
            self.Population.append(
                SolutionjDE(self.D,
                            self.Lower,
                            self.Upper,
                            self.F,
                            self.CR))

    def tryEval(self, v):
        if self.FEs <= self.nFES:
            v.evaluate()
            self.FEs += 1
        else:
            self.Done = True

    def generationStep(self, Population):
        """Implement main DE/jDE step."""

        newPopulation = []
        for i in range(self.Np):
            newSolution = SolutionjDE(
                self.D,
                self.Lower,
                self.Upper,
                self.F,
                self.CR)

            if rnd.random() < self.Tao:
                newSolution.F = rnd.random()
            else:
                newSolution.F = Population[i].F

            if rnd.random() < self.Tao:
                newSolution.CR = rnd.random()
            else:
                newSolution.CR = Population[i].CR

            r = rnd.sample(range(0, self.Np), 3)
            while i in r:
                r = rnd.sample(range(0, self.Np), 3)
            jrand = int(rnd.random() * self.Np)

            for j in range(self.D):
                if rnd.random() < newSolution.CR or j == jrand:
                    newSolution.Solution[j] = Population[r[0]].Solution[j] + newSolution.F * (
                        Population[r[1]].Solution[j] - Population[r[2]].Solution[j])
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

    def run(self):
        self.initPopulation()
        self.evalPopulation()
        self.FEs = self.Np
        while not self.Done:
            self.Population = self.generationStep(self.Population)
        return self.bestSolution.Fitness
