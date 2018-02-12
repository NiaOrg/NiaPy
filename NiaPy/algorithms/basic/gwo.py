"""Grey wolf optimizer.

Date: 11.2.2018

Author : Iztok Fister Jr.

License: MIT

Reference paper: Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Andrew Lewis.
"Grey wolf optimizer." Advances in engineering software 69 (2014): 46-61.
#& Grey Wold Optimizer (GWO) source codes version 1.0 (MATLAB)

TODO: Validation must be conducted! More tests are required!
"""

import random

__all__ = ['GreyWolfOptimizer']


class GreyWolfOptimizer(object):

    # pylint: disable=too-many-instance-attributes
    def __init__(self, D, NP, nFES, Lower, Upper, function):
        self.D = D  # dimension of the problem
        self.NP = NP  # population size; number of search agents
        self.nFES = nFES  # number of function evaluations
        self.Lower = Lower  # lower bound
        self.Upper = Upper  # upper bound
        self.Fun = function

        self.Positions = [[0 for _i in range(self.D)]  # positions of search agents
                          for _j in range(self.NP)]

        self.evaluations = 0  # evaluations counter

        # TODO: "-inf" is in the case of maximization problems
        self.Alpha_pos = [0] * self.D  # init of alpha
        self.Alpha_score = float("inf")

        self.Beta_pos = [0] * self.D  # init of beta
        self.Beta_score = float("inf")

        self.Delta_pos = [0] * self.D  # init of delta
        self.Delta_score = float("inf")

    def initialization(self):
        # initialization of positions
        for i in range(self.NP):
            for j in range(self.D):
                self.Positions[i][j] = random.random(
                ) * (self.Upper - self.Lower) + self.Lower

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

        while True:
            if self.evaluations == self.nFES:
                break

            for i in range(self.NP):
                self.Positions[i] = self.bounds(self.Positions[i])

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
