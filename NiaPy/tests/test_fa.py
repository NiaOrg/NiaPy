from unittest import TestCase

from NiaPy.algorithms.basic import FireflyAlgorithm

class FATestCase(TestCase):
    def setUp(self):
        def Fun(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val

        self.fa = FireflyAlgorithm(
            10, 20, 10000, 0.5, 0.2, 1.0, -2.0, 2.0, Fun)

    def test_works_fine(self):
        self.assertTrue(self.fa.Run())
