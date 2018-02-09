"""Implementation of Rosenbrock benchmark function."""

import math

__all__ = ['Rosenbrock']


class Rosenbrock(object):

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            val = 0.0

            for i in range(D - 1):
                val = val + 100 * \
                    math.pow(sol[i + 1] - math.pow((sol[i]), 2),
                             2) + math.pow((sol[i] - 1), 2)

            return val

        return evaluate
