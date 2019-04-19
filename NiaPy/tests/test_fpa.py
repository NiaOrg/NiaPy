# pylint: disable=line-too-long
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import FlowerPollinationAlgorithm


class FPATestCase(AlgorithmTestCase):

    def test_type_parameters(self):
        d = FlowerPollinationAlgorithm.typeParameters()
        self.assertTrue(d['NP'](10))
        self.assertFalse(d['NP'](-10))
        self.assertFalse(d['NP'](0))
        self.assertTrue(d['beta'](10))
        self.assertFalse(d['beta'](0))
        self.assertFalse(d['beta'](-10))
        self.assertTrue(d['p'](0.5))
        self.assertFalse(d['p'](-0.5))
        self.assertFalse(d['p'](1.5))

    def test_custom_works_fine(self):
        fpa_custom = FlowerPollinationAlgorithm(NP=10, p=0.5, seed=self.seed)
        fpa_customc = FlowerPollinationAlgorithm(NP=10, p=0.5, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, fpa_custom, fpa_customc, MyBenchmark())

    def test_griewank_works_fine(self):
        fpa_griewank = FlowerPollinationAlgorithm(NP=20, p=0.5, seed=self.seed)
        fpa_griewankc = FlowerPollinationAlgorithm(NP=20, p=0.5, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, fpa_griewank, fpa_griewankc)

    def test_griewank_works_fine_with_beta(self):
        fpa_beta_griewank = FlowerPollinationAlgorithm(NP=20, p=0.5, beta=1.2, seed=self.seed)
        fpa_beta_griewankc = FlowerPollinationAlgorithm(NP=20, p=0.5, beta=1.2, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, fpa_beta_griewank, fpa_beta_griewankc)
