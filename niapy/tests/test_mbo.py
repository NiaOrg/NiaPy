from niapy.algorithms.basic import MonarchButterflyOptimization
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class MBOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MonarchButterflyOptimization

    def test_type_parameters(self):
        tp = MonarchButterflyOptimization.type_parameters()
        self.assertTrue(tp['population_size'](1))
        self.assertFalse(tp['population_size'](0))
        self.assertFalse(tp['population_size'](-1))
        self.assertFalse(tp['population_size'](1.0))
        self.assertTrue(tp['partition'](1.0))
        self.assertFalse(tp['partition'](0.0))
        self.assertFalse(tp['partition'](-1.0))
        self.assertTrue(tp['period'](1.0))
        self.assertFalse(tp['period'](0.0))
        self.assertFalse(tp['period'](-1.0))

    def test(self):
        mbo = self.algo(NP=20, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
        mboc = self.algo(NP=20, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mbo, mboc, MyBenchmark())

    def test_griewank(self):
        mbo_griewank = self.algo(NP=20, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
        mbo_griewankc = self.algo(NP=20, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mbo_griewank, mbo_griewankc)
