"""Implementation of Sphere functions."""
import math

__all__ = ['Sphere']


class Sphere(object):
    def __init__(self, Lower=-5.12, Upper=5.12):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val += math.pow(sol[i], 2)
            return val

        return evaluate
