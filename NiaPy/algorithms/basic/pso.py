"""Particle Swarm Optimization algorithm.

Date: 12. 2. 2018

Authors : Uros Mlakar

License: MIT

Reference paper: Kennedy, J. and Eberhart, R. "Particle Swarm Optimization".
Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942--1948, 1995.

EDITED: TODO: Tests and validation! Bug in code.
"""

import random as rnd
import copy

__all__ = ['ParticleSwarmAlgorithm']


class Particle(object):
    '''Defines particle for population'''
    # pylint: disable=too-many-instance-attributes
    def __init__(self, D, LB, UB):
        self.D = D
        self.LB = LB
        self.UB = UB
        self.vmax = 6
        self.Solution = []
        self.Velocity = []

        self.pBestPosition = []
        self.bestFitness = float('inf')

        self.Fitness = float('inf')
        self.generateParticle()

    def generateParticle(self):
        self.Solution = [self.LB + (self.UB - self.LB)
                         * rnd.random() for _i in range(self.D)]
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
        for i in range(len(self.Solution)):
            if self.Solution[i] < self.LB:
                self.Solution[i] = self.LB
            if self.Solution[i] > self.UB:
                self.Solution[i] = self.UB
            if self.Velocity[i] > self.vmax:
                self.Velocity[i] = self.vmax
            if self.Velocity[i] < -self.vmax:
                self.Velocity[i] = -self.vmax

    def toString(self):
        pass

    def __eq__(self, other):
        return self.Solution == other.Solution and self.Fitness == other.Fitness


class ParticleSwarmAlgorithm(object):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, Np, D, nFES, C1, C2, w, Lower, Upper, function):
        '''Constructor'''
        self.Np = Np
        self.D = D
        self.nFES = nFES
        self.Swarm = []
        self.Lower = Lower
        self.Upper = Upper
        self.C1 = C1
        self.C2 = C2
        self.w = w
        self.vmax = 0.9
        self.vmin = 0.2
        Particle.FuncEval = staticmethod(function)

        self.gBest = Particle(D, Lower, Upper)

    def evalSwarm(self):
        for p in self.Swarm:
            p.evaluate()
            if p.Fitness < self.gBest.Fitness:
                self.gBest = copy.deepcopy(p)

    def initSwarm(self):
        for _i in range(self.Np):
            self.Swarm.append(Particle(self.D, self.Lower, self.Upper))

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

            p.evaluate()
            if p.Fitness < self.gBest.Fitness:
                self.gBest = copy.deepcopy(p)

            MovedSwarm.append(p)
        return MovedSwarm

    def run(self):
        self.initSwarm()
        self.evalSwarm()
        FEs = self.Np
        while FEs <= self.nFES:
            MovedSwarm = self.moveSwarm(self.Swarm)
            self.Swarm = MovedSwarm
            FEs += self.Np
        return self.gBest.Fitness
