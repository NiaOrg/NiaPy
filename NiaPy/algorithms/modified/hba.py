import random
from NiaPy.benchmarks.utility import Utility

__all__ = ['HybridBatAlgorithm']


class HybridBatAlgorithm(object):
    r"""Implementation of Hybrid bat algorithm.

    **Algorithm:** Hybrid bat algorithm

    **Date:** 2018

    **Author:** Grega Vrbancic

    **License:** MIT

    **Reference paper:**
        Fister Jr., Iztok and Fister, Dusan and Yang, Xin-She.
        "A Hybrid Bat Algorithm". Elektrotehniski vestnik, 2013. 1-7.
    """

    def __init__(self, D, NP, nFES, A, r, F, CR, Qmin, Qmax, benchmark):
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
        self.F = F  # scaling factor
        self.CR = CR  # crossover rate
        self.Qmin = Qmin  # frequency min
        self.Qmax = Qmax  # frequency max
        self.Lower = self.benchmark.Lower  # lower bound
        self.Upper = self.benchmark.Upper  # upper bound
        self.eval_flag = True  # evaluations flag
        self.Fun = self.benchmark.function()

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
        """Check evauations."""

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
        self.init_bat()

        S = [[self.best[i] for i in range(self.D)] for j in range(self.NP)]

        while self.eval_flag is not False:
            for i in range(self.NP):
                rnd = random.uniform(0, 1)
                self.Q[i] = self.Qmin + (self.Qmin - self.Qmax) * rnd

                j = None
                for j in range(self.D):
                    self.v[i][j] = self.v[i][j] + (self.Sol[i][j] -
                                                   self.best[j]) * self.Q[i]
                    S[i][j] = self.Sol[i][j] + self.v[i][j]

                    S[i][j] = self.simplebounds(S[i][j], self.Lb[j],
                                                self.Ub[j])

                rnd = random.random()

                if rnd > self.r:
                    nums = random.sample(range(0, self.NP), 4)  # DE step
                    for j in range(self.D):
                        if random.random() < self.CR:
                            S[i][j] = self.best[j] + self.F * \
                                (S[nums[0]][j] + S[nums[1]][j] - S[nums[2]][j] - S[nums[3]][j])
                            S[i][j] = self.simplebounds(
                                S[i][j], self.Lb[j], self.Ub[j])

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
        return self.move_bat()
