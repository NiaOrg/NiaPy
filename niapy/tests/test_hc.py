# encoding=utf8
from niapy.algorithms.other import HillClimbAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class HCTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = HillClimbAlgorithm

    def test_custom(self):
        ihc_custom = self.algo(delta=0.4, seed=self.seed)
        ihc_customc = self.algo(delta=0.4, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ihc_custom, ihc_customc, MyProblem())

    def test_griewank(self):
        ihc_griewank = self.algo(delta=0.1, seed=self.seed)
        ihc_griewankc = self.algo(delta=0.1, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ihc_griewank, ihc_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
