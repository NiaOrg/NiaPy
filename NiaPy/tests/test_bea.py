# pylint: disable=line-too-long
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import BeesAlgorithm


class BEATestCase(AlgorithmTestCase):

    def test_type_parameters(self):
        tp = BeesAlgorithm.typeParameters()
        self.assertTrue(tp['NP'](1))
        self.assertFalse(tp['NP'](0))
        self.assertFalse(tp['NP'](-1))
        self.assertFalse(tp['NP'](1.0))
        self.assertTrue(tp['m'](1))
        self.assertFalse(tp['m'](0))
        self.assertFalse(tp['m'](-1))
        self.assertFalse(tp['m'](1.0))
        self.assertTrue(tp['e'](1))
        self.assertFalse(tp['e'](0))
        self.assertFalse(tp['e'](-1))
        self.assertFalse(tp['e'](1.0))
        self.assertTrue(tp['nep'](1))
        self.assertFalse(tp['nep'](0))
        self.assertFalse(tp['nep'](-1))
        self.assertFalse(tp['nep'](1.0))
        self.assertTrue(tp['nsp'](1))
        self.assertFalse(tp['nsp'](0))
        self.assertFalse(tp['nsp'](-1))
        self.assertFalse(tp['nsp'](1.0))
        self.assertTrue(tp['ngh'](1.0))
        self.assertTrue(tp['ngh'](0.5))
        self.assertFalse(tp['ngh'](0.0))
        self.assertFalse(tp['ngh'](-1))

    def test_works_fine(self):
        bea = BeesAlgorithm(NP=20, m=15, e=4, nep=10, nsp=5, ngh=2, seed=self.seed)
        beac = BeesAlgorithm(NP=20, m=15, e=4, nep=10, nsp=5, ngh=2, seed=self.seed)

        AlgorithmTestCase.algorithm_run_test(self, bea, beac, MyBenchmark())

    def test_griewank_works_fine(self):
        bea_griewank = BeesAlgorithm(NP=20, m=15, e=4, nep=10, nsp=5, ngh=2, seed=self.seed)
        bea_griewankc = BeesAlgorithm(NP=20, m=15, e=4, nep=10, nsp=5, ngh=2, seed=self.seed)

        AlgorithmTestCase.algorithm_run_test(self, bea_griewank, bea_griewankc)
