"""Artificial Bee Colony algorithm.

Date: 12. 2. 2018

Authors : Uros Mlakar

License: MIT

Reference paper: Karaboga, D., and Bahriye B. "A powerful
and efficient algorithm for numerical function optimization: artificial
bee colony (ABC) algorithm." Journal of global optimization 39.3 (2007): 459-471.

"""

import random as rnd
import copy

__all__ = ['ArtificialBeeColonyAlgorithm']


class SolutionABC:
    def __init__(self, D, LB, UB):
        self.D = D
        self.Solution = []
        self.Fitness = float('inf')
        self.LB = LB
        self.UB = UB
        self.generateSolution()

    def generateSolution(self):
        self.Solution = [self.LB + (self.UB - self.LB)
                         * rnd.random() for i in range(self.D)]

    def repair(self):
        for i in range(self.D):
            if (self.Solution[i] > self.UB):
                self.Solution[i] = self.UB
            if self.Solution[i] < self.LB:
                self.Solution[i] = self.LB

    def evaluate(self):
        self.Fitness = SolutionABC.FuncEval(self.D, self.Solution)

    def toString(self):
        pass


class ArtificialBeeColonyAlgorithm:
    def __init__(self, NP, D, nFES, Lower, Upper, function):
        self.NP = NP
        self.D = D
        self.FoodNumber = self.NP / 2
        self.Limit = 100
        self.Trial = []
        self.Foods = []
        self.Probs = []
        self.nFES = nFES
        self.Lower = Lower
        self.Upper = Upper
        SolutionABC.FuncEval = staticmethod(function)
        self.Best = SolutionABC(self.D, Lower, Upper)

    def reset(self):
        self.__init__(self.NP, self.D, self.MaxCycle)

    def init(self):
        self.Probs = [0 for i in range(self.FoodNumber)]
        self.Trial = [0 for i in range(self.FoodNumber)]
        for i in range(self.FoodNumber):
            self.Foods.append(SolutionABC(self.D, self.Lower, self.Upper))
            self.Foods[i].evaluate()
            self.checkForBest(self.Foods[i])

    def CalculateProbs(self):
        self.Probs = [1.0 / (self.Foods[i].Fitness + 0.01)
                      for i in range(self.FoodNumber)]
        s = sum(self.Probs)
        self.Probs = [self.Probs[i] / s for i in range(self.FoodNumber)]

    def checkForBest(self, Solution):
        if Solution.Fitness <= self.Best.Fitness:
            self.Best = copy.deepcopy(Solution)

    def run(self):
        self.init()
        FEs = self.FoodNumber
        while FEs < self.nFES:
            self.Best.toString()
            for i in range(self.FoodNumber):
                newSolution = copy.deepcopy(self.Foods[i])
                param2change = int(rnd.random() * self.D)
                neighbor = int(self.FoodNumber * rnd.random())
                newSolution.Solution[param2change] = self.Foods[i].Solution[param2change] + (-1 + 2 * rnd.random()) * (
                    self.Foods[i].Solution[param2change] - self.Foods[neighbor].Solution[param2change])
                newSolution.repair()
                newSolution.evaluate()
                if newSolution.Fitness < self.Foods[i].Fitness:
                    self.checkForBest(newSolution)
                    self.Foods[i] = newSolution
                    self.Trial[i] = 0
                else:
                    self.Trial[i] += 1
            FEs += self.FoodNumber
            self.CalculateProbs()
            t, s = 0, 0
            while t < self.FoodNumber:
                if rnd.random() < self.Probs[s]:
                    t += 1
                    Solution = copy.deepcopy(self.Foods[s])
                    param2change = int(rnd.random() * self.D)
                    neighbor = int(self.FoodNumber * rnd.random())
                    while neighbor == s:
                        neighbor = int(self.FoodNumber * rnd.random())
                    Solution.Solution[param2change] = self.Foods[s].Solution[param2change] + (-1 + 2 * rnd.random()) * (
                        self.Foods[s].Solution[param2change] - self.Foods[neighbor].Solution[param2change])
                    Solution.repair()
                    Solution.evaluate()
                    FEs += 1
                    if Solution.Fitness < self.Foods[s].Fitness:
                        self.checkForBest(newSolution)
                        self.Foods[s] = Solution
                        self.Trial[s] = 0
                    else:
                        self.Trial[s] += 1
                s += 1
                if s == self.FoodNumber:
                    s = 0

            mi = self.Trial.index(max(self.Trial))
            if self.Trial[mi] >= self.Limit:
                self.Foods[mi] = SolutionABC(self.D, self.Lower, self.Upper)
                self.Foods[mi].evaluate()
                FEs += 1
                self.Trial[mi] = 0
        return self.Best.Fitness
