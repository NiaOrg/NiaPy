import random as rnd
import copy

__all__ = ['SelfAdaptiveDifferentialEvolutionAlgorithm']


class SolutionjDE(object):
    def __init__(self, D, LB, UB):
        self.D = D
        self.LB = LB
        self.UB = UB
        self.F = 0.5
        self.CR = 0.9
        self.Solution = []
        self.Fitness = float('inf')
        self.generateSolution()

    def generateSolution(self):
        self.Solution = [self.LB + (self.UB - self.LB) * rnd.random()
                         for _i in range(self.D)]

    def evaluate(self):
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
    """Self-adaptive differential evolution algorithm.

    Date: 7. 2. 2018

    Authors : Uros Mlakar

    License: MIT

    Reference paper: Brest, J., Greiner, S., Boskovic, B., Mernik, M., Zumer, V. Self-adapting control
    parameters in differential evolution: A comparative study on numerical benchmark problems.
    IEEE transactions on evolutionary computation, 10(6), 646-657, 2006.
    """

    def __init__(self, D, NP, nFES, Lower, Upper, function):
        # TODO: check for F and CR parameters!
        self.D = D  # dimension of problem
        self.Np = NP  # population size
        self.nFES = nFES  # number of function evaluations
        self.Lower = Lower  # lower bound
        self.Upper = Upper  # upper bound

        SolutionjDE.FuncEval = staticmethod(function)
        self.Population = []
        self.FEs = 0
        self.Done = False
        self.bestSolution = SolutionjDE(self.D, Lower, Upper)
        self.Tao = None  # EDITED: check please

    def evalPopulation(self):
        for p in self.Population:
            p.evaluate()
            if p.Fitness < self.bestSolution.Fitness:
                self.bestSolution = copy.deepcopy(p)

    def initPopulation(self):
        for _i in range(self.Np):
            self.Population.append(SolutionjDE(self.D, self.Lower, self.Upper))

    def tryEval(self, v):
        if self.FEs <= self.nFES:
            v.evaluate()
            self.FEs += 1
        else:
            self.Done = True

    def generationStep(self, Population):
        newPopulation = []
        for i in range(self.Np):
            newSolution = SolutionjDE(self.D, self.Lower, self.Upper)

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
