# encoding=utf8
from niapy.algorithms.other import AnarchicSocietyOptimization
from niapy.algorithms.other.aso import elitism, sequential, crossover
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class ASOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = AnarchicSocietyOptimization


class ASOElitismTestCase(ASOTestCase):
    def test_custom(self):
        aso_custom = self.algo(population_size=10, combination=elitism, seed=self.seed)
        aso_customc = self.algo(population_size=10, combination=elitism, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyProblem())

    def test_griewank(self):
        aso_griewank = self.algo(population_size=10, combination=elitism, seed=self.seed)
        aso_griewankc = self.algo(population_size=10, combination=elitism, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_griewank, aso_griewankc)


class ASOSequentialTestCase(AlgorithmTestCase):
    def test_custom(self):
        aso_custom = self.algo(population_size=10, combination=sequential, seed=self.seed)
        aso_customc = self.algo(population_size=10, combination=sequential, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyProblem())

    def test_griewank(self):
        aso_griewank = self.algo(population_size=10, combination=sequential, seed=self.seed)
        aso_griewankc = self.algo(population_size=10, combination=sequential, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_griewank, aso_griewankc)


class ASOCrossoverTestCase(AlgorithmTestCase):
    def test_custom(self):
        aso_custom = self.algo(population_size=10, combination=crossover, seed=self.seed)
        aso_customc = self.algo(population_size=10, combination=crossover, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyProblem())

    def test_griewank(self):
        aso_griewank = self.algo(population_size=10, combination=crossover, seed=self.seed)
        aso_griewankc = self.algo(population_size=10, combination=crossover, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_griewank, aso_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
