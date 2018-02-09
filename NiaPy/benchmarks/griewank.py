"""Implementation of Griewank function."""
import math


class Griewank(object):
    def __init__(self, D):
        self.D = D

    def evaluate(self, sol):

        val = 0.0
        val1 = 0.0
        val2 = 1.0

        for i in range(self.D):
            val1 = val1 + math.pow(sol[i], 2)
            val2 = val2 + \
                math.cos((((sol[i]) / math.sqrt(i + 1)) * math.pi) / 180)

        val = (1 / 4000) * val1 - val2 + 1

        return val
