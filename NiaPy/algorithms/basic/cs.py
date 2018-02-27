# encoding=utf8
import random as rnd
import copy
import numpy as npx
from scipy.special import gamma as Gamma

from NiaPy.benchmarks.utility import Utility

__all__ = ['CuckooSearchAlgorithm']


class Cuckoo(object):
    """Defines cuckoo for population."""

    def __init__(self, D, LB, UB):
        self.D = D  # dimension of the problem
        self.LB = LB  # lower bound
        self.UB = UB  # upper bound
        self.Solution = []

        self.Fitness = float('inf')
        self.generateCuckoo()

    def generateCuckoo(self):
        self.Solution = [self.LB + (self.UB - self.LB) * rnd.random()
                         for _i in range(self.D)]

    def evaluate(self):
        self.Fitness = Cuckoo.FuncEval(self.D, self.Solution)

    def simpleBound(self):
        for i in range(self.D):
            if self.Solution[i] < self.LB:
                self.Solution[i] = self.LB
            if self.Solution[i] > self.UB:
                self.Solution[i] = self.UB

    def toString(self):
        return (self.Solution, self.Fitness)

    def __eq__(self, other):
        return self.Solution == other.Solution and self.Fitness == other.Fitness


class CuckooSearchAlgorithm(object):
    r"""Implementation of Cuckoo Search algorithm.

    **Algorithm:** Cuckoo Search algorithm

    **Date:** 2018

    **Author:** Uros Mlakar

    **License:** MIT

    **Reference paper:**
        Yang, Xin-She, and Suash Deb. "Cuckoo search via Lévy flights."
        Nature & Biologically Inspired Computing, 2009. NaBIC 2009.

    """

    def __init__(self, D, Np, nFES, Pa, Alpha, benchmark):
        r"""**__init__(self, D, NP, nFES, Pa, Alpha, benchmark)**.

        Arguments:
            D {integer} -- dimension of problem

            NP {integer} -- population size

            nFES {integer} -- number of function evaluations

            Pa {decimal} -- probability

            Alpha {decimal} -- alpha

            benchmark {object} -- benchmark implementation object

        """

        self.benchmark = Utility().get_benchmark(benchmark)
        self.Np = Np  # population size
        self.D = D  # dimension
        self.nFES = nFES  # number of function evaluations
        self.Pa = Pa
        self.Alpha = Alpha
        self.Lower = self.benchmark.Lower
        self.Upper = self.benchmark.Upper
        self.Nests = []
        self.FEs = 0
        self.Done = False
        self.Beta = 1.5
        Cuckoo.FuncEval = staticmethod(self.benchmark.function())

        self.gBest = Cuckoo(self.D, self.Lower, self.Upper)

    def evalNests(self):
        """Evaluate nests."""

        for c in self.Nests:
            self.tryEval(c)
            if c.Fitness <= self.gBest.Fitness:
                self.gBest = copy.deepcopy(c)

    def initNests(self):
        """Initialize nests."""

        for _i in range(self.Np):
            self.Nests.append(Cuckoo(self.D, self.Lower, self.Upper))

    def tryEval(self, c):
        """Check evaluations."""

        if self.FEs < self.nFES:
            c.evaluate()
            self.FEs += 1
        else:
            self.Done = True

    def findBest(self, NewNests):
        TempNests = copy.deepcopy(self.Nests)
        for i, n in enumerate(NewNests):
            self.tryEval(n)
            if n.Fitness <= self.Nests[i].Fitness:
                TempNests[i] = copy.deepcopy(n)
                if n.Fitness <= self.gBest.Fitness:
                    self.gBest = copy.deepcopy(n)
        return TempNests

    def performLevyFlights(self, Nests):
        """Move nests."""

        MovedNests = []
        sigma = (Gamma(1 + self.Beta) * npx.sin(npx.pi * self.Beta / 2) /
                 (Gamma((1 + self.Beta) / 2) * self.Beta * 2**((self.Beta - 1) / 2)))**(1 / self.Beta)
        
        for i in range(self.Np):
            c = Nests[i]
            u = npx.random.randn(self.D) * sigma
            v = npx.random.randn(self.D)
            step = u / (abs(v)**(1 / self.Beta))
            stepsize = self.Alpha * step * (npx.array(c.Solution) - npx.array(self.gBest.Solution)).flatten().tolist()
            c.Solution = (npx.array(c.Solution) + npx.array(stepsize) * npx.random.randn(self.D)).tolist()
            c.simpleBound()
            MovedNests.append(copy.deepcopy(c))
        return MovedNests

    def emptyNests(self, NewNests):


        tNest=npx.zeros((self.Np,self.D))
        Nest=npx.zeros((self.Np,self.D))
        K=npx.random.uniform(0,1,(self.Np,self.D))>self.Pa
        Nests = []

        for i, c in enumerate(NewNests):
            Nests.append(Cuckoo(self.D, self.Lower, self.Upper))
            for j in range(self.D):
                nest[i, j] = c.Solution[j]

        stepsize = rnd.random() * (nest[npx.random.permutation(self.Np), :] - nest[npx.random.permutation(self.Np), :])
        tempnest = nest + stepsize * K

        for i in range(self.Np):
            Nests[i].Solution = tNest[i].tolist()
            Nests[i].simpleBound()
        return Nests

    def run(self):
        self.initNests()
        self.evalNests()
        while not self.Done:
            MovedNests = self.performLevyFlights(self.Nests)
            self.Nests = self.findBest(MovedNests)
            ResetNests = self.emptyNests(MovedNests)
            self.Nests = self.findBest(ResetNests)

        return self.gBest.Fitness
