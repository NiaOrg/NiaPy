import random as rnd
import copy
from NiaPy.benchmarks.utility import Utility

__all__ = ['ParticleSwarmAlgorithm']


class Particle(object):
    """Defines particle for population."""

    def __init__(self, D, LB, UB, vMin, vMax):
        self.D = D  # dimension of the problem
        self.LB = LB  # lower bound
        self.UB = UB  # upper bound
        self.vMin = vMin  # minimal velocity
        self.vMax = vMax  # maximal velocity
        self.Solution = []
        self.Velocity = []

        self.pBestPosition = []
        self.pBestSolution = []
        self.bestFitness = float('inf')

        self.Fitness = float('inf')
        self.generateParticle()

    def generateParticle(self):
        self.Solution = [self.LB + (self.UB - self.LB) * rnd.random()
                         for _i in range(self.D)]
        self.Velocity = [0 for _i in range(self.D)]

        self.pBestSolution = [0 for _i in range(self.D)]
        self.bestFitness = float('inf')

    def evaluate(self):
        self.Fitness = Particle.FuncEval(self.D, self.Solution)
        self.checkPersonalBest()

    def checkPersonalBest(self):
        if self.Fitness < self.bestFitness:
            self.pBestSolution = self.Solution
            self.bestFitness = self.Fitness

    def simpleBound(self):
        for i in range(self.D):
            if self.Solution[i] < self.LB:
                self.Solution[i] = self.LB
            if self.Solution[i] > self.UB:
                self.Solution[i] = self.UB
            if self.Velocity[i] < self.vMin:
                self.Velocity[i] = self.vMin
            if self.Velocity[i] > self.vMax:
                self.Velocity[i] = self.vMax

    def toString(self):
        pass

    def __eq__(self, other):
        return self.Solution == other.Solution and self.Fitness == other.Fitness


class ParticleSwarmAlgorithm(object):
    r"""Implementation of Particle Swarm Optimization algorithm.

    **Algorithm:** Particle Swarm Optimization algorithm

    **Date:** 2018

    **Author:** Uros Mlakar

    **License:** MIT

    **Reference paper:**
        Kennedy, J. and Eberhart, R. "Particle Swarm Optimization".
        Proceedings of IEEE International Conference on Neural Networks.
        IV. pp. 1942--1948, 1995.

    EDITED: TODO: Tests and validation! Bug in code.
    """

    def __init__(self, Np, D, nFES, C1, C2, w, vMin, vMax, benchmark):
        r"""**__init__(self, Np, D, nFES, C1, C2, w, vMin, vMax, benchmark)**.

        Arguments:
            NP {integer} -- population size

            D {integer} -- dimension of problem

            nFES {integer} -- number of function evaluations

            C1 {decimal} -- cognitive component

            C2 {decimal} -- social component

            w {decimal} -- inertia weight

            vMin {decimal} -- minimal velocity

            vMax {decimal} -- maximal velocity

            benchmark {object} -- benchmark implementation object

        """

        self.benchmark = Utility().get_benchmark(benchmark)
        self.Np = Np  # population size; number of search agents
        self.D = D  # dimension of the problem
        self.C1 = C1  # cognitive component
        self.C2 = C2  # social component
        self.w = w  # inertia weight
        self.vMin = vMin  # minimal velocity
        self.vMax = vMax  # maximal velocity
        self.Lower = self.benchmark.Lower  # lower bound
        self.Upper = self.benchmark.Upper  # upper bound
        self.Swarm = []
        self.nFES = nFES  # number of function evaluations
        self.FEs = 0
        self.Done = False
        Particle.FuncEval = staticmethod(self.benchmark.function())

        self.gBest = Particle(
            self.D,
            self.Lower,
            self.Upper,
            self.vMin,
            self.vMax)

    def evalSwarm(self):
        for p in self.Swarm:
            p.evaluate()
            if p.Fitness < self.gBest.Fitness:
                self.gBest = copy.deepcopy(p)

    def initSwarm(self):
        for _i in range(self.Np):
            self.Swarm.append(
                Particle(self.D,
                         self.Lower,
                         self.Upper,
                         self.vMin,
                         self.vMax))

    def tryEval(self, p):
        if self.FEs <= self.nFES:
            p.evaluate()
            self.FEs += 1
        else:
            self.Done = True

    def moveSwarm(self, Swarm):
        MovedSwarm = []
        for p in Swarm:

            part1 = ([(a - b) * rnd.random() * self.C1 for a,
                      b in zip(p.pBestSolution, p.Solution)])
            part2 = ([(a - b) * rnd.random() * self.C2 for a,
                      b in zip(self.gBest.Solution, p.Solution)])

            p.Velocity = ([self.w * a + b + c for a, b,
                           c in zip(p.Velocity, part1, part2)])
            p.Solution = ([a + b for a, b in zip(p.Solution, p.Velocity)])

            p.simpleBound()
            self.tryEval(p)
            if p.Fitness < self.gBest.Fitness:
                self.gBest = copy.deepcopy(p)

            MovedSwarm.append(p)
        return MovedSwarm

    def run(self):
        self.initSwarm()
        self.evalSwarm()
        self.FEs += self.Np
        while not self.Done:
            MovedSwarm = self.moveSwarm(self.Swarm)
            self.Swarm = MovedSwarm

        return self.gBest.Fitness
