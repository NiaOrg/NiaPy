"""Implementation of Sphere function."""

__all__ = ['Sphere']


class Sphere(object):

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val

        return evaluate
