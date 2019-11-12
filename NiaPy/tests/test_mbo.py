from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import MonarchButterflyOptimization


class MBOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MonarchButterflyOptimization

    def test_type_parameters(self):
        tp = MonarchButterflyOptimization.typeParameters()
        self.assertTrue(tp['NP'](1))
        self.assertFalse(tp['NP'](0))
        self.assertFalse(tp['NP'](-1))
        self.assertFalse(tp['NP'](1.0))
        self.assertTrue(tp['PAR'](1.0))
        self.assertFalse(tp['PAR'](0.0))
        self.assertFalse(tp['PAR'](-1.0))
        self.assertTrue(tp['PER'](1.0))
        self.assertFalse(tp['PER'](0.0))
        self.assertFalse(tp['PER'](-1.0))

    def test_works_fine(self):
        mbo = self.algo(NP=20, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
        mboc = self.algo(NP=20, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mbo, mboc, MyBenchmark())

    def test_griewank_works_fine(self):
        mbo_griewank = self.algo(NP=20, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
        mbo_griewankc = self.algo(NP=20, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mbo_griewank, mbo_griewankc)
