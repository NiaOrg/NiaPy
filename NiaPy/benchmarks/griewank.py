"""Implementation of Griewank function."""
import math

__all__ = ['Griewank']


class Griewank(object):

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0
            val1 = 0.0
            val2 = 1.0

            for i in range(D):
                val1 = val1 + math.pow(sol[i], 2)
                val2 = val2 + \
                    math.cos((((sol[i]) / math.sqrt(i + 1)) * math.pi) / 180)

            val = (1 / 4000) * val1 - val2 + 1

            return val

        return evaluate
