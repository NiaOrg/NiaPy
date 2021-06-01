# encoding=utf8
from niapy.algorithms.basic import BeesAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class BEATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = BeesAlgorithm

    def test_custom(self):
        bea = self.algo(population_size=10, m=5, e=4, nep=5, nsp=5, ngh=2, seed=self.seed)
        beac = self.algo(population_size=10, m=5, e=4, nep=5, nsp=5, ngh=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, bea, beac, MyProblem())

    def test_griewank(self):
        bea_griewank = self.algo(population_size=10, m=5, e=4, nep=5, nsp=5, ngh=2, seed=self.seed)
        bea_griewankc = self.algo(population_size=10, m=5, e=4, nep=5, nsp=5, ngh=2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, bea_griewank, bea_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
