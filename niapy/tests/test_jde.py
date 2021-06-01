# encoding=utf8
from unittest import TestCase

from numpy.random import default_rng

from niapy.algorithms.modified import SelfAdaptiveDifferentialEvolution, \
    MultiStrategySelfAdaptiveDifferentialEvolution
from niapy.algorithms.modified.jde import SolutionJDE
from niapy.task import Task
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class SolutionJDETestCase(TestCase):
    def setUp(self):
        self.D, self.F, self.CR = 10, 0.9, 0.3
        self.x, self.task = default_rng().uniform(10, 50, self.D), Task(problem=MyProblem(self.D))
        self.s1, self.s2 = SolutionJDE(task=self.task, e=False), SolutionJDE(differential_weight=self.F,
                                                                             crossover_probability=self.CR, x=self.x)

    def test_F(self):
        self.assertAlmostEqual(self.s1.differential_weight, 2)
        self.assertAlmostEqual(self.s2.differential_weight, self.F)

    def test_cr(self):
        self.assertAlmostEqual(self.s1.crossover_probability, 0.5)
        self.assertAlmostEqual(self.s2.crossover_probability, self.CR)


class JDETestCase(AlgorithmTestCase):

    def test_custom(self):
        jde_custom = SelfAdaptiveDifferentialEvolution(f_lower=0.0, f_upper=2.0, tao1=0.9, tao2=0.45,
                                                       population_size=10, differential_weight=0.5,
                                                       crossover_probability=0.1, seed=self.seed)
        jde_customc = SelfAdaptiveDifferentialEvolution(f_lower=0.0, f_upper=2.0, tao1=0.9, tao2=0.45,
                                                        population_size=10, differential_weight=0.5,
                                                        crossover_probability=0.1, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, jde_custom, jde_customc, MyProblem())

    def test_griewank(self):
        jde_griewank = SelfAdaptiveDifferentialEvolution(f_lower=0.0, f_upper=2.0, tao1=0.9, tao2=0.45,
                                                         population_size=10, differential_weight=0.5,
                                                         crossover_probability=0.1, seed=self.seed)
        jde_griewankc = SelfAdaptiveDifferentialEvolution(f_lower=0.0, f_upper=2.0, tao1=0.9, tao2=0.45,
                                                          population_size=10, differential_weight=0.5,
                                                          crossover_probability=0.1, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, jde_griewank, jde_griewankc)


class MsjDETestCase(AlgorithmTestCase):
    def test_custom(self):
        jde_custom = MultiStrategySelfAdaptiveDifferentialEvolution(population_size=10, differential_weight=0.5,
                                                                    f_lower=0.0, f_upper=2.0, tao1=0.9,
                                                                    crossover_probability=0.1,
                                                                    tao2=0.45, seed=self.seed)
        jde_customc = MultiStrategySelfAdaptiveDifferentialEvolution(population_size=10, differential_weight=0.5,
                                                                     f_lower=0.0, f_upper=2.0, tao1=0.9,
                                                                     crossover_probability=0.1,
                                                                     tao2=0.45, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, jde_custom, jde_customc, MyProblem())

    def test_griewank(self):
        jde_griewank = MultiStrategySelfAdaptiveDifferentialEvolution(population_size=10, differential_weight=0.5,
                                                                      f_lower=0.0, f_upper=2.0, tao1=0.9,
                                                                      crossover_probability=0.1,
                                                                      tao2=0.45, seed=self.seed)
        jde_griewankc = MultiStrategySelfAdaptiveDifferentialEvolution(population_size=10, differential_weight=0.5,
                                                                       f_lower=0.0, f_upper=2.0, tao1=0.9,
                                                                       crossover_probability=0.1,
                                                                       tao2=0.45, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, jde_griewank, jde_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
