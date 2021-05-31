# encoding=utf8

from niapy.algorithms.basic import BatAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class BATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = BatAlgorithm

    def test_custom(self):
        ba_custom = self.algo(population_size=10, loudness=0.5, pulse_rate=0.5, min_frequency=0.0, max_frequency=2.0, seed=self.seed)
        ba_customc = self.algo(population_size=10, loudness=0.5, pulse_rate=0.5, min_frequency=0.0, max_frequency=2.0, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ba_custom, ba_customc, MyProblem())

    def test_griewank(self):
        ba_griewank = self.algo(population_size=10, loudness=0.5, pulse_rate=0.5, min_frequency=0.0, max_frequency=2.0, seed=self.seed)
        ba_griewankc = self.algo(population_size=10, loudness=0.5, pulse_rate=0.5, min_frequency=0.0, max_frequency=2.0, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ba_griewank, ba_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
