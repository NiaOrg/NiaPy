"""Implementation of Sphere function."""

__all__ = ['Sphere']


class Sphere(object):

    def __init__(self, Lower=-100, Upper=100):
        self.Lower = Lower
        self.Upper = Upper

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val

        return evaluate
