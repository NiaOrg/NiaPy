"""Implementation of Rastrigin benchmark function."""
import math

__all__ = ['Rastrigin']


class Rastrigin(object):

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            val = 0.0

            for i in range(D):
                val = val + \
                    math.pow(sol[i], 2) - 10 * math.cos(
                        2 * math.pi * sol[i]) + 10

            return val

        return evaluate
