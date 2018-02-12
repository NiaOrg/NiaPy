"""Implementation of Ackley function."""
import math

__all__ = ['Ackley']


class Ackley(object):

    def __init__(self, Lower=-100, Upper=100):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):

            a = 20  # Recommended variable value
            b = 0.2  # Recommended variable value
            c = 2 * math.pi  # Recommended variable value

            val = 0.0
            val1 = 0.0
            val2 = 0.0

            for i in range(D):
                val1 += math.pow(sol[i], 2)
                val2 += math.cos(c * sol[i])

            temp1 = -b * math.sqrt(val1 / D)
            temp2 = val2 / D

            val = -a * math.exp(temp1) - math.exp(temp2) + a + math.exp(1)

            return val

        return evaluate
