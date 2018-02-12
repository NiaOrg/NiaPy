from unittest import TestCase

from NiaPy.algorithms.basic import BatAlgorithm


class BATestCase(TestCase):
    def setUp(self):
        def Fun(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val
        
        self.ba_custom = BatAlgorithm(
            10, 20, 10000, 0.5, 0.5, 0.0, 2.0, -10.0, 10.0, Fun)
        self.ba_griewank = BatAlgorithm(
            10, 40, 10000, 0.5, 0.5, 0.0, 2.0, -10.0, 10.0, 'griewank')

    def test_custom_works_fine(self):
        self.assertTrue(self.ba_custom.move_bat())

    def test_griewank_works_fine(self):
        self.assertTrue(self.ba_griewank.move_bat())
