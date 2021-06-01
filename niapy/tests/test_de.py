# encoding=utf8

from niapy.algorithms.basic import DifferentialEvolution, DynNpDifferentialEvolution, AgingNpDifferentialEvolution, \
    MultiStrategyDifferentialEvolution, DynNpMultiStrategyDifferentialEvolution
from niapy.algorithms.basic.de import cross_rand1, cross_rand2, cross_best1, cross_best2, cross_curr2rand1, cross_curr2best1
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class DETestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = DifferentialEvolution

    def test_Custom(self):
        de_custom = self.algo(population_size=10, differential_weight=0.5, crossover_probability=0.9, seed=self.seed)
        de_customc = self.algo(population_size=10, differential_weight=0.5, crossover_probability=0.9, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_custom, de_customc, MyProblem())

    def test_griewank(self):
        de_griewank = self.algo(population_size=10, crossover_probability=0.5, differential_weight=0.9, seed=self.seed)
        de_griewankc = self.algo(population_size=10, crossover_probability=0.5, differential_weight=0.9, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_griewank, de_griewankc)

    def test_CrossRand1(self):
        de_rand1 = self.algo(population_size=10, strategy=cross_rand1, seed=self.seed)
        de_rand1c = self.algo(population_size=10, strategy=cross_rand1, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_rand1, de_rand1c)

    def test_CrossBest1(self):
        de_best1 = self.algo(population_size=10, strategy=cross_best1, seed=self.seed)
        de_best1c = self.algo(population_size=10, strategy=cross_best1, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_best1, de_best1c)

    def test_CrossRand2(self):
        de_rand2 = self.algo(population_size=10, strategy=cross_rand2, seed=self.seed)
        de_rand2c = self.algo(population_size=10, strategy=cross_rand2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_rand2, de_rand2c)

    def test_CrossBest2(self):
        de_best2 = self.algo(population_size=10, strategy=cross_best2, seed=self.seed)
        de_best2c = self.algo(population_size=10, strategy=cross_best2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_best2, de_best2c)

    def test_CrossCurr2Rand1(self):
        de_curr2rand1 = self.algo(population_size=10, strategy=cross_curr2rand1, seed=self.seed)
        de_curr2rand1c = self.algo(population_size=10, strategy=cross_curr2rand1, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_curr2rand1, de_curr2rand1c)

    def test_CrossCurr2Best1(self):
        de_curr2best1 = self.algo(population_size=10, strategy=cross_curr2best1, seed=self.seed)
        de_curr2best1c = self.algo(population_size=10, strategy=cross_curr2best1, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_curr2best1, de_curr2best1c)


class DynNpDETestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = DynNpDifferentialEvolution

    def test_Custom(self):
        de_custom = self.algo(population_size=10, differential_weight=0.5, crossover_probability=0.9, seed=self.seed)
        de_customc = self.algo(population_size=10, differential_weight=0.5, crossover_probability=0.9, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_custom, de_customc, MyProblem())

    def test_griewank(self):
        de_griewank = self.algo(population_size=10, crossover_probability=0.5, differential_weight=0.9, seed=self.seed)
        de_griewankc = self.algo(population_size=10, crossover_probability=0.5, differential_weight=0.9, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_griewank, de_griewankc, 'griewank')


class ANpDETestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = AgingNpDifferentialEvolution

    def test_Custom(self):
        de_custom = self.algo(population_size=10, differential_weight=0.5, crossover_probability=0.9, seed=self.seed)
        de_customc = self.algo(population_size=10, differential_weight=0.5, crossover_probability=0.9, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_custom, de_customc, MyProblem())

    def test_griewank(self):
        de_griewank = self.algo(population_size=10, crossover_probability=0.5, differential_weight=0.9, seed=self.seed)
        de_griewankc = self.algo(population_size=10, crossover_probability=0.5, differential_weight=0.9, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_griewank, de_griewankc, 'griewank')


class MsDETestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MultiStrategyDifferentialEvolution

    def test_Custom(self):
        de_custom = MultiStrategyDifferentialEvolution(population_size=10, differential_weight=0.5, crossover_probability=0.9, seed=self.seed)
        de_customc = MultiStrategyDifferentialEvolution(population_size=10, differential_weight=0.5, crossover_probability=0.9, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_custom, de_customc, MyProblem())

    def test_griewank(self):
        de_griewank = MultiStrategyDifferentialEvolution(population_size=10, crossover_probability=0.5, differential_weight=0.9, seed=self.seed)
        de_griewankc = MultiStrategyDifferentialEvolution(population_size=10, crossover_probability=0.5, differential_weight=0.9, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_griewank, de_griewankc, 'griewank')


class DynNpMsDETestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = DynNpMultiStrategyDifferentialEvolution

    def test_Custom(self):
        de_custom = self.algo(population_size=10, rp=3, differential_weight=0.5, crossover_probability=0.9, seed=self.seed)
        de_customc = self.algo(population_size=10, rp=3, differential_weight=0.5, crossover_probability=0.9, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_custom, de_customc, MyProblem())

    def test_griewank(self):
        de_griewank = self.algo(population_size=10, crossover_probability=0.5, differential_weight=0.9, seed=self.seed)
        de_griewankc = self.algo(population_size=10, crossover_probability=0.5, differential_weight=0.9, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, de_griewank, de_griewankc, 'griewank')

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
