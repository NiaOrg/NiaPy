# encoding=utf8

from niapy.algorithms.other import SimulatedAnnealing
from niapy.algorithms.other.sa import cool_linear
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class SATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = SimulatedAnnealing

    def test_custom(self):
        ca_custom = self.algo(population_size=10, seed=self.seed)
        ca_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyProblem())

    def test_griewank(self):
        ca_griewank = self.algo(population_size=10, seed=self.seed)
        ca_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc)

    def test_custom1(self):
        ca_custom = self.algo(population_size=10, seed=self.seed, cooling_method=cool_linear)
        ca_customc = self.algo(population_size=10, seed=self.seed, cooling_method=cool_linear)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyProblem())

    def test_griewank1(self):
        ca_griewank = self.algo(population_size=10, seed=self.seed, cooling_method=cool_linear)
        ca_griewankc = self.algo(population_size=10, seed=self.seed, cooling_method=cool_linear)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
