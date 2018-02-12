"""Implementation of Schwefel function."""
import math

__all__ = ['Schwefel']


class Schwefel(object):

    def __init__(self, Lower=-500, Upper=500):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            val = 0.0
            val1 = 0.0

            for i in range(D):
                val1 += (sol[i] * math.sin(math.sqrt(abs(sol[i]))))

            val = 418.9829 * D - val1

            return val

        return evaluate
