import random
import numpy as np
from scipy.special import gamma as Gamma
from NiaPy.benchmarks.utility import Utility

__all__ = ['FlowerPollinationAlgorithm']


class FlowerPollinationAlgorithm(object):
    r"""Implementation of Flower Pollination algorithm.

    **Algorithm:** Flower Pollination algorithm

    **Date:** 2018

    **Authors:** Dusan Fister & Iztok Fister Jr.

    **License:** MIT

    **Reference paper:**
        Yang, Xin-She. "Flower pollination algorithm for global optimization."
        International conference on unconventional computing and natural computation.
        Springer, Berlin, Heidelberg, 2012.

    Implementation is based on the following MATLAB code:
    https://www.mathworks.com/matlabcentral/fileexchange/45112-flower-pollination-algorithm?requestedDomain=true
    """

    def __init__(self, D, NP, nFES, p, benchmark):
        r"""**__init__(self, D, NP, nFES, p, benchmark)**.

        Arguments:
            D {integer} -- dimension of problem

            NP {integer} -- population size

            nFES {integer} -- number of function evaluations

            p {decimal} -- probability switch

            benchmark {object} -- benchmark implementation object

        Raises:
            TypeError -- Raised when given benchmark function which does not exists.

        """

        self.benchmark = Utility().get_benchmark(benchmark)
        self.D = D  # dimension
        self.NP = NP  # population size
        self.nFES = nFES  # number of function evaluations
        self.p = p  # probability switch
        self.Lower = self.benchmark.Lower  # lower bound
        self.Upper = self.benchmark.Upper  # upper bound
        self.Fun = self.benchmark.function()  # function

        self.f_min = 0.0  # minimum fitness

        self.Lb = [0] * self.D  # lower bound
        self.Ub = [0] * self.D  # upper bound

        self.dS = [[0 for _i in range(self.D)]
                   for _j in range(self.NP)]  # differential
        self.Sol = [[0 for _i in range(self.D)]
                    for _j in range(self.NP)]  # population of solutions
        self.Fitness = [0] * self.NP  # fitness
        self.best = [0] * self.D  # best solution
        self.eval_flag = True  # evaluations flag
        self.evaluations = 0  # evaluations counter

    def best_flower(self):
        i = 0
        j = 0
        for i in range(self.NP):
            if self.Fitness[i] < self.Fitness[j]:
                j = i
        for i in range(self.D):
            self.best[i] = self.Sol[j][i]
        self.f_min = self.Fitness[j]

    def eval_true(self):
        """Check evaluations."""

        if self.evaluations == self.nFES:
            self.eval_flag = False

    @classmethod
    def simplebounds(cls, val, lower, upper):
        if val < lower:
            val = lower
        if val > upper:
            val = upper
        return val

    def init_flower(self):
        for i in range(self.D):
            self.Lb[i] = self.Lower
            self.Ub[i] = self.Upper

        for i in range(self.NP):
            for j in range(self.D):
                rnd = random.uniform(0, 1)
                self.dS[i][j] = 0.0
                self.Sol[i][j] = self.Lb[j] + (self.Ub[j] - self.Lb[j]) * rnd
            self.Fitness[i] = self.Fun(self.D, self.Sol[i])
            self.evaluations = self.evaluations + 1
        self.best_flower()

    def move_flower(self):
        S = [[0.0 for i in range(self.D)] for j in range(self.NP)]
        self.init_flower()

        while self.eval_flag is not False:
            for i in range(self.NP):
                if random.uniform(0, 1) > self.p:  # probability switch
                    L = self.Levy()
                    for j in range(self.D):
                        self.dS[i][j] = L[j] * (self.Sol[i][j] - self.best[j])
                        S[i][j] = self.Sol[i][j] + self.dS[i][j]

                        S[i][j] = self.simplebounds(
                            S[i][j], self.Lb[j], self.Ub[j])
                else:
                    epsilon = random.uniform(0, 1)
                    JK = np.random.permutation(self.NP)

                    for j in range(self.D):
                        S[i][j] = S[i][j] + epsilon * \
                            (self.Sol[JK[0]][j] - self.Sol[JK[1]][j])
                        S[i][j] = self.simplebounds(
                            S[i][j], self.Lb[j], self.Ub[j])

                self.eval_true()
                if self.eval_flag is not True:
                    break

                Fnew = self.Fun(self.D, S[i])
                self.evaluations = self.evaluations + 1

                if Fnew <= self.Fitness[i]:
                    for j in range(self.D):
                        self.Sol[i][j] = S[i][j]
                    self.Fitness[i] = Fnew

                if Fnew <= self.f_min:
                    for j in range(self.D):
                        self.best[j] = S[i][j]
                    self.f_min = Fnew

        return self.f_min

    def Levy(self):
        beta = 1.5
        sigma = (Gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (Gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = [[0] for j in range(self.D)]
        v = [[0] for j in range(self.D)]
        step = [[0] for j in range(self.D)]
        L = [[0] for j in range(self.D)]

        for j in range(self.D):
            u[j] = np.random.normal(0, 1) * sigma
            v[j] = np.random.normal(0, 1)
            step[j] = u[j] / abs(v[j])**(1 / beta)
            L[j] = 0.01 * step[j]

        return L

    def run(self):
        return self.move_flower()
