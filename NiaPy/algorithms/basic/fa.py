"""Firefly algorithm.

Date: 2016

Authors : Iztok Fister Jr. and Iztok Fister
License: MIT

Reference paper: Fister, I., Fister Jr, I., Yang, X. S., & Brest, J. (2013).
A comprehensive review of firefly algorithms. Swarm and Evolutionary Computation, 13, 34-46.
"""

import random
import math
from NiaPy.benchmarks.utility import Utility

__all__ = ['FireflyAlgorithm']


class FireflyAlgorithm(object):
    """Firefly Algorithm implementation."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, D, NP, nFES, alpha, betamin, gamma, benchmark):
        self.benchmark = Utility.get_benchmark(benchmark)
        self.D = D  # dimension of the problem
        self.NP = NP  # population size
        self.nFES = nFES  # number of function evaluations
        self.alpha = alpha  # alpha parameter
        self.betamin = betamin  # beta parameter
        self.gamma = gamma  # gamma parameter
        # sort of fireflies according to fitness value
        self.Index = [0] * self.NP
        self.Fireflies = [[0 for _i in range(self.D)]
                          for _j in range(self.NP)]  # firefly agents
        self.Fireflies_tmp = [[0 for _i in range(self.D)] for _j in range(
            self.NP)]  # intermediate pop
        self.Fitness = [0.0] * self.NP  # fitness values
        self.Intensity = [0.0] * self.NP  # light intensity
        self.nbest = [0.0] * self.NP  # the best solution found so far
        self.Lower = self.benchmark.Lower  # lower bound
        self.Upper = self.benchmark.Upper  # upper bound
        self.fbest = None  # the best
        self.evaluations = 0
        self.Fun = self.benchmark.function()

    def init_ffa(self):
        for i in range(self.NP):
            for j in range(self.D):
                self.Fireflies[i][j] = random.uniform(
                    0, 1) * (self.Upper - self.Lower) + self.Lower
            self.Fitness[i] = 1.0  # initialize attractiveness
            self.Intensity[i] = self.Fitness[i]

    def alpha_new(self, a):
        delta = 1.0 - math.pow((math.pow(10.0, -4.0) / 0.9), 1.0 / float(a))
        return (1 - delta) * self.alpha

    def sort_ffa(self):  # implementation of bubble sort
        for i in range(self.NP):
            self.Index[i] = i

        for i in range(0, (self.NP - 1)):
            j = i + 1
            for j in range(j, self.NP):
                if self.Intensity[i] > self.Intensity[j]:
                    z = self.Intensity[i]  # exchange attractiveness
                    self.Intensity[i] = self.Intensity[j]
                    self.Intensity[j] = z
                    z = self.Fitness[i]  # exchange fitness
                    self.Fitness[i] = self.Fitness[j]
                    self.Fitness[j] = z
                    z = self.Index[i]  # exchange indexes
                    self.Index[i] = self.Index[j]
                    self.Index[j] = z

    def replace_ffa(self):  # replace the old population according to the new Index values
        # copy original population to a temporary area
        for i in range(self.NP):
            for j in range(self.D):
                self.Fireflies_tmp[i][j] = self.Fireflies[i][j]

        # generational selection in the sense of an EA
        for i in range(self.NP):
            for j in range(self.D):
                self.Fireflies[i][j] = self.Fireflies_tmp[self.Index[i]][j]

    def FindLimits(self, k):
        for i in range(self.D):
            if self.Fireflies[k][i] < self.Lower:
                self.Fireflies[k][i] = self.Lower
            if self.Fireflies[k][i] > self.Upper:
                self.Fireflies[k][i] = self.Upper

    def move_ffa(self):
        for i in range(self.NP):
            scale = abs(self.Upper - self.Lower)
            for j in range(self.NP):
                r = 0.0
                for k in range(self.D):
                    r += (self.Fireflies[i][k] - self.Fireflies[j][k]) * \
                        (self.Fireflies[i][k] - self.Fireflies[j][k])
                r = math.sqrt(r)
                if self.Intensity[i] > self.Intensity[j]:  # brighter and more attractive
                    beta0 = 1.0
                    beta = (beta0 - self.betamin) * \
                        math.exp(-self.gamma * math.pow(r, 2.0)) + self.betamin
                    for k in range(self.D):
                        r = random.uniform(0, 1)
                        tmpf = self.alpha * (r - 0.5) * scale
                        self.Fireflies[i][k] = self.Fireflies[i][
                            k] * (1.0 - beta) + self.Fireflies_tmp[j][k] * beta + tmpf
            self.FindLimits(i)

    def run(self):
        self.init_ffa()

        while True:
            if self.evaluations == self.nFES:
                break

            # optional reducing of alpha
            self.alpha = self.alpha_new(self.nFES / self.NP)

            # evaluate new solutions
            for i in range(self.NP):
                self.Fitness[i] = self.Fun(self.D, self.Fireflies[i])
                self.evaluations = self.evaluations + 1
                self.Intensity[i] = self.Fitness[i]

            # ranking fireflies by their light intensity
            self.sort_ffa()
            # replace old population
            self.replace_ffa()
            # find the current best
            self.fbest = self.Intensity[0]
            # move all fireflies to the better locations
            self.move_ffa()

        return self.fbest
