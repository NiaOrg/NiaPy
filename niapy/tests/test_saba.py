# pylint: disable=line-too-long
from niapy.algorithms.modified import SelfAdaptiveBatAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class HBATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = SelfAdaptiveBatAlgorithm

    def test_custom(self):
        hba_custom = self.algo(population_size=10, starting_loudness=0.5, pulse_rate=0.5, min_frequency=0.0, max_frequency=2.0, seed=self.seed)
        hba_customc = self.algo(population_size=10, starting_loudness=0.5, pulse_rate=0.5, min_frequency=0.0, max_frequency=2.0, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hba_custom, hba_customc, MyProblem())

    def test_griewank(self):
        hba_griewank = self.algo(population_size=10, starting_loudness=0.5, pulse_rate=0.5, min_frequency=0.0, max_frequency=2.0, seed=self.seed)
        hba_griewankc = self.algo(population_size=10, starting_loudness=0.5, pulse_rate=0.5, min_frequency=0.0, max_frequency=2.0, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hba_griewank, hba_griewankc)
