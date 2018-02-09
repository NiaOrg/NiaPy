"""Implementation of Rosenbrock benchmark function."""

import math

__all__ = ['Rosenbrock']


class Rosenbrock(object):

    def __init__(self, D, sol):
        self.D = D
        self.sol = sol

    def evaluate(self):
        val = 0.0

        for i in range(self.D - 1):
            val = val + 100 * \
                math.pow(
                    self.sol[i + 1] - math.pow((self.sol[i]),
                                               2),
                    2) + math.pow((self.sol[i] - 1),
                                  2)

        return val
