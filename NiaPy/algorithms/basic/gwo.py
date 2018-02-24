import random
from NiaPy.benchmarks.utility import Utility

__all__ = ['GreyWolfOptimizer']


class GreyWolfOptimizer(object):
    r"""Implementation of Grey wolf optimizer.

    **Algorithm:** Grey wolf optimizer

    **Date:** 2018

    **Author:** Iztok Fister Jr.

    **License:** MIT

    **Reference paper:**
        Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Andrew Lewis.
        "Grey wolf optimizer." Advances in engineering software 69 (2014): 46-61.
        & Grey Wold Optimizer (GWO) source code version 1.0 (MATLAB) from MathWorks
    """

    def __init__(self, D, NP, nFES, benchmark):
        r"""**__init__(self, D, NP, nFES, benchmark)**.

        Arguments:
            D {integer} -- dimension of problem

            NP {integer} -- population size

            nFES {integer} -- number of function evaluations

            benchmark {object} -- benchmark implementation object

        Raises:
            TypeError -- Raised when given benchmark function which does not exists.

        """

        self.benchmark = Utility().get_benchmark(benchmark)
        self.D = D  # dimension of the problem
        self.NP = NP  # population size; number of search agents
        self.nFES = nFES  # number of function evaluations
        self.Lower = self.benchmark.Lower  # lower bound
        self.Upper = self.benchmark.Upper  # upper bound
        self.Fun = self.benchmark.function()

        self.Positions = [[0 for _i in range(self.D)]  # positions of search agents
                          for _j in range(self.NP)]

        self.eval_flag = True  # evaluations flag
        self.evaluations = 0  # evaluations counter

        self.Alpha_pos = [0] * self.D  # init of alpha
        self.Alpha_score = float("inf")

        self.Beta_pos = [0] * self.D  # init of beta
        self.Beta_score = float("inf")

        self.Delta_pos = [0] * self.D  # init of delta
        self.Delta_score = float("inf")

    def initialization(self):
        """Initialize positions."""
        for i in range(self.NP):
            for j in range(self.D):
                self.Positions[i][j] = random.random(
                ) * (self.Upper - self.Lower) + self.Lower

    def eval_true(self):
        """Check evaluations."""

        if self.evaluations == self.nFES:
            self.eval_flag = False

    def bounds(self, position):
        for i in range(self.D):
            if position[i] < self.Lower:
                position[i] = self.Lower
            if position[i] > self.Upper:
                position[i] = self.Upper
        return position

    # pylint: disable=too-many-locals
    def move(self):

        self.initialization()

        while self.eval_flag is not False:

            for i in range(self.NP):
                self.Positions[i] = self.bounds(self.Positions[i])

                self.eval_true()
                if self.eval_flag is not True:
                    break

                Fit = self.Fun(self.D, self.Positions[i])
                self.evaluations = self.evaluations + 1

                if Fit < self.Alpha_score:
                    self.Alpha_score = Fit
                    self.Alpha_pos = self.Positions[i]

                if ((Fit > self.Alpha_score) and (Fit < self.Beta_score)):
                    self.Beta_score = Fit
                    self.Beta_pos = self.Positions[i]

                if ((Fit > self.Alpha_score) and (Fit > self.Beta_score) and
                        (Fit < self.Delta_score)):
                    self.Delta_score = Fit
                    self.Delta_pos = self.Positions[i]

            a = 2 - self.evaluations * ((2) / self.nFES)

            for i in range(self.NP):
                for j in range(self.D):

                    r1 = random.random()
                    r2 = random.random()

                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2

                    D_alpha = abs(
                        C1 * self.Alpha_pos[j] - self.Positions[i][j])
                    X1 = self.Alpha_pos[j] - A1 * D_alpha

                    r1 = random.random()
                    r2 = random.random()

                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2

                    D_beta = abs(C2 * self.Beta_pos[j] - self.Positions[i][j])
                    X2 = self.Beta_pos[j] - A2 * D_beta

                    r1 = random.random()
                    r2 = random.random()

                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2

                    D_delta = abs(
                        C3 * self.Delta_pos[j] - self.Positions[i][j])
                    X3 = self.Delta_pos[j] - A3 * D_delta

                    self.Positions[i][j] = (X1 + X2 + X3) / 3

        return self.Alpha_score

    def run(self):
        return self.move()
