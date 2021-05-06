# encoding=utf8
from niapy.algorithms.other import AnarchicSocietyOptimization
from niapy.algorithms.other.aso import elitism, sequential, crossover
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class ASOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = AnarchicSocietyOptimization

    def test_parameter_types(self):
        d = self.algo.type_parameters()
        self.assertTrue(d['population_size'](1))
        self.assertFalse(d['population_size'](0))
        self.assertFalse(d['population_size'](-1))
        self.assertTrue(d['mutation_rate'](10))
        self.assertFalse(d['mutation_rate'](0))
        self.assertFalse(d['mutation_rate'](-10))
        self.assertTrue(d['crossover_rate'](0.1))
        self.assertFalse(d['crossover_rate'](-19))
        self.assertFalse(d['crossover_rate'](19))
        self.assertTrue(d['alpha'](10))
        self.assertTrue(d['gamma'](10))
        self.assertTrue(d['theta'](10))


class ASOElitismTestCase(ASOTestCase):
    def test_custom(self):
        aso_custom = self.algo(population_size=40, combination=elitism, seed=self.seed)
        aso_customc = self.algo(population_size=40, combination=elitism, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyBenchmark())

    def test_griewank(self):
        aso_griewank = self.algo(population_size=40, combination=elitism, seed=self.seed)
        aso_griewankc = self.algo(population_size=40, combination=elitism, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_griewank, aso_griewankc)


class ASOSequentialTestCase(AlgorithmTestCase):
    def test_custom(self):
        aso_custom = self.algo(population_size=40, combination=sequential, seed=self.seed)
        aso_customc = self.algo(population_size=40, combination=sequential, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyBenchmark())

    def test_griewank(self):
        aso_griewank = self.algo(population_size=40, combination=sequential, seed=self.seed)
        aso_griewankc = self.algo(population_size=40, combination=sequential, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_griewank, aso_griewankc)


class ASOCrossoverTestCase(AlgorithmTestCase):
    def test_custom(self):
        aso_custom = self.algo(population_size=40, combination=crossover, seed=self.seed)
        aso_customc = self.algo(population_size=40, combination=crossover, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_custom, aso_customc, MyBenchmark())

    def test_griewank(self):
        aso_griewank = self.algo(population_size=40, combination=crossover, seed=self.seed)
        aso_griewankc = self.algo(population_size=40, combination=crossover, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, aso_griewank, aso_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
