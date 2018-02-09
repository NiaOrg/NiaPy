"""Implementation of Sphere function."""

__all__ = ['Sphere']


class Sphere:
    def __init__(self, D, sol):
        self.D = D
        self.sol = sol

    def evaluate(self):
        val = 0.0
        for i in range(self.D):
            val = val + self.sol[i] * self.sol[i]
        return val
