"""Implementation of Rastrigin benchmark function."""
import math


class Rastrigin:

    def __init__(self, D, sol):
        self.D = D
        self.sol = sol

    def evaluate(self):
        val = 0.0

        for i in range(self.D):
            val = val + \
                math.pow(self.sol[i], 2) - 10 * math.cos(
                    2 * math.pi * self.sol[i]) + 10

        return val
