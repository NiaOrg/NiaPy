# encoding=utf8
from niapy.algorithms.other.aso import AnarchicSocietyOptimization, elitism, sequential, crossover
from tests.test_algorithm import AlgorithmTestCase, MyProblem


class ASOElitismTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = AnarchicSocietyOptimization

    def test_custom(self):
        aso_custom = self.algo(population_size=10, combination=elitism, seed=self.seed)
        aso_customc = self.algo(population_size=10, combination=elitism, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyProblem())

    def test_griewank(self):
        aso_griewank = self.algo(population_size=10, combination=elitism, seed=self.seed)
        aso_griewankc = self.algo(population_size=10, combination=elitism, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_griewank, aso_griewankc)


class ASOSequentialTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = AnarchicSocietyOptimization

    def test_custom(self):
        aso_custom = self.algo(population_size=10, combination=sequential, seed=self.seed)
        aso_customc = self.algo(population_size=10, combination=sequential, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyProblem())

    def test_griewank(self):
        aso_griewank = self.algo(population_size=10, combination=sequential, seed=self.seed)
        aso_griewankc = self.algo(population_size=10, combination=sequential, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_griewank, aso_griewankc)


class ASOCrossoverTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = AnarchicSocietyOptimization

    def test_custom(self):
        aso_custom = self.algo(population_size=10, combination=crossover, seed=self.seed)
        aso_customc = self.algo(population_size=10, combination=crossover, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyProblem())

    def test_griewank(self):
        aso_griewank = self.algo(population_size=10, combination=crossover, seed=self.seed)
        aso_griewankc = self.algo(population_size=10, combination=crossover, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_griewank, aso_griewankc)
