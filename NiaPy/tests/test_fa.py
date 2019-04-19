# pylint: disable=line-too-long
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import FireflyAlgorithm


class FATestCase(AlgorithmTestCase):

    def test_type_parameters(self):
        d = FireflyAlgorithm.typeParameters()
        self.assertTrue(d['alpha'](10))
        self.assertFalse(d['alpha'](-10))
        self.assertTrue(d['betamin'](10))
        self.assertFalse(d['betamin'](-10))
        self.assertTrue(d['gamma'](10))
        self.assertFalse(d['gamma'](-10))
        self.assertTrue(d['NP'](1))
        self.assertFalse(d['NP'](0))
        self.assertFalse(d['NP'](-1))

    def test_works_fine(self):
        fa = FireflyAlgorithm(NP=20, alpha=0.5, betamin=0.2, gamma=1.0, seed=self.seed)
        fac = FireflyAlgorithm(NP=20, alpha=0.5, betamin=0.2, gamma=1.0, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, fa, fac, MyBenchmark())

    def test_griewank_works_fine(self):
        fa_griewank = FireflyAlgorithm(NP=20, alpha=0.5, betamin=0.2, gamma=1.0, seed=self.seed)
        fa_griewankc = FireflyAlgorithm(NP=20, alpha=0.5, betamin=0.2, gamma=1.0, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, fa_griewank, fa_griewankc)
