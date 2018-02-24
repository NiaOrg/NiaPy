import random
from NiaPy.benchmarks.utility import Utility

__all__ = ['BatAlgorithm']


class BatAlgorithm(object):
    r"""Implementation of Bat algorithm.

    **Algorithm:** Bat algorithm

    **Date:** 2015

    **Authors:** Iztok Fister Jr. and Marko Burjek

    **License:** MIT

    **Reference paper:**
        Yang, Xin-She. "A new metaheuristic bat-inspired algorithm."
        Nature inspired cooperative strategies for optimization (NICSO 2010).
        Springer, Berlin, Heidelberg, 2010. 65-74.
    """

    def __init__(self, D, NP, nFES, A, r, Qmin, Qmax, benchmark):
        r"""**__init__(self, D, NP, nFES, A, r, Qmin, Qmax, benchmark)**.

        Arguments:
            D {integer} -- dimension of problem

            NP {integer} -- population size

            nFES {integer} -- number of function evaluations

            A {decimal} -- loudness

            r {decimal} -- pulse rate

            Qmin {decimal} -- minimum frequency

            Qmax {decimal } -- maximum frequency

            benchmark {object} -- benchmark implementation object

        Raises:
            TypeError -- Raised when given benchmark function which does not exists.

        """

        self.benchmark = Utility().get_benchmark(benchmark)
        self.D = D  # dimension
        self.NP = NP  # population size
        self.nFES = nFES  # number of function evaluations
        self.A = A  # loudness
        self.r = r  # pulse rate
        self.Qmin = Qmin  # frequency min
        self.Qmax = Qmax  # frequency max
        self.Lower = self.benchmark.Lower  # lower bound
        self.Upper = self.benchmark.Upper  # upper bound

        self.f_min = 0.0  # minimum fitness

        self.Lb = [0] * self.D  # lower bound
        self.Ub = [0] * self.D  # upper bound
        self.Q = [0] * self.NP  # frequency

        self.v = [[0 for _i in range(self.D)]
                  for _j in range(self.NP)]  # velocity
        self.Sol = [[0 for _i in range(self.D)] for _j in range(
            self.NP)]  # population of solutions
        self.Fitness = [0] * self.NP  # fitness
        self.best = [0] * self.D  # best solution
        self.evaluations = 0  # evaluations counter
        self.eval_flag = True  # evaluations flag
        self.Fun = self.benchmark.function()

    def best_bat(self):
        """Find the best bat."""

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

    def init_bat(self):
        """Initialize population."""

        for i in range(self.D):
            self.Lb[i] = self.Lower
            self.Ub[i] = self.Upper

        for i in range(self.NP):
            self.Q[i] = 0
            for j in range(self.D):
                rnd = random.uniform(0, 1)
                self.v[i][j] = 0.0
                self.Sol[i][j] = self.Lb[j] + (self.Ub[j] - self.Lb[j]) * rnd
            self.Fitness[i] = self.Fun(self.D, self.Sol[i])
            self.evaluations = self.evaluations + 1
        self.best_bat()

    @classmethod
    def simplebounds(cls, val, lower, upper):
        if val < lower:
            val = lower
        if val > upper:
            val = upper
        return val

    def move_bat(self):
        """Move bats in search space."""
        S = [[0.0 for i in range(self.D)] for j in range(self.NP)]

        self.init_bat()

        while self.eval_flag is not False:
            for i in range(self.NP):
                rnd = random.uniform(0, 1)
                self.Q[i] = self.Qmin + (self.Qmin - self.Qmax) * rnd
                for j in range(self.D):
                    self.v[i][j] = self.v[i][j] + (self.Sol[i][j] -
                                                   self.best[j]) * self.Q[i]
                    S[i][j] = self.Sol[i][j] + self.v[i][j]

                    S[i][j] = self.simplebounds(S[i][j], self.Lb[j],
                                                self.Ub[j])

                rnd = random.random()

                if rnd > self.r:
                    for j in range(self.D):
                        S[i][j] = self.best[j] + 0.001 * random.gauss(0, 1)
                        S[i][j] = self.simplebounds(S[i][j], self.Lb[j],
                                                    self.Ub[j])

                self.eval_true()

                if self.eval_flag is not True:
                    break

                Fnew = self.Fun(self.D, S[i])
                self.evaluations = self.evaluations + 1
                rnd = random.random()

                if (Fnew <= self.Fitness[i]) and (rnd < self.A):
                    for j in range(self.D):
                        self.Sol[i][j] = S[i][j]
                    self.Fitness[i] = Fnew

                if Fnew <= self.f_min:
                    for j in range(self.D):
                        self.best[j] = S[i][j]
                    self.f_min = Fnew

        return self.f_min

    def run(self):
        """Run algorithm with initialized parameters.

        Return {decimal} - best
        """
        return self.move_bat()
