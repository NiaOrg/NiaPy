# encoding=utf8
from niapy.algorithms.basic import KrillHerd
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class KHTestCase(AlgorithmTestCase):
    r"""Test case for KrillHerdV1 algorithm.

    Date:
        April 2019

    Author:
        Klemen Berkovič

    See Also:
        * :class:`niapy.tests.test_algorithm.AlgorithmTestCase`
    """

    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = KrillHerd

    def test_custom(self):
        kh_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        kh_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, kh_custom, kh_customc, MyProblem())

    def test_griewank(self):
        kh_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        kh_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, kh_griewank, kh_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
