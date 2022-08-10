# encoding=utf8
from unittest import TestCase

from numpy.random import default_rng

from niapy.algorithms.modified import SuccessHistoryAdaptiveDifferentialEvolution, \
    LpsrSuccessHistoryAdaptiveDifferentialEvolution
from niapy.algorithms.modified.shade import SolutionSHADE
from niapy.task import Task
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class SolutionSHADETestCase(TestCase):
    def setUp(self):
        self.D, self.F, self.CR = 10, 0.9, 0.3
        self.x, self.task = default_rng().uniform(10, 50, self.D), Task(problem=MyProblem(self.D))
        self.s1, self.s2 = SolutionSHADE(task=self.task, e=False), SolutionSHADE(differential_weight=self.F,
                                                                                 crossover_probability=self.CR,
                                                                                 x=self.x)

    def test_F(self):
        self.assertAlmostEqual(self.s1.differential_weight, 0.5)
        self.assertAlmostEqual(self.s2.differential_weight, self.F)

    def test_CR(self):
        self.assertAlmostEqual(self.s1.crossover_probability, 0.5)
        self.assertAlmostEqual(self.s2.crossover_probability, self.CR)


class SHADETestCase(AlgorithmTestCase):

    def test_custom(self):
        shade_custom = SuccessHistoryAdaptiveDifferentialEvolution(init_pop_size=10, extern_arc_rate=2.0,
                                                                   pbest_factor=0.2, hist_mem_size=5, seed=self.seed)
        shade_customc = SuccessHistoryAdaptiveDifferentialEvolution(init_pop_size=10, extern_arc_rate=2.0,
                                                                    pbest_factor=0.2, hist_mem_size=5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, shade_custom, shade_customc, MyProblem())

    def test_griewank(self):
        shade_griewank = SuccessHistoryAdaptiveDifferentialEvolution(init_pop_size=10, extern_arc_rate=2.0,
                                                                     pbest_factor=0.2, hist_mem_size=5, seed=self.seed)
        shade_griewankc = SuccessHistoryAdaptiveDifferentialEvolution(init_pop_size=10, extern_arc_rate=2.0,
                                                                      pbest_factor=0.2, hist_mem_size=5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, shade_griewank, shade_griewankc)


class LSHADETestCase(AlgorithmTestCase):

    def test_custom(self):
        lshade_custom = LpsrSuccessHistoryAdaptiveDifferentialEvolution(init_pop_size=10, extern_arc_rate=2.0,
                                                                        pbest_factor=0.2, hist_mem_size=5,
                                                                        seed=self.seed)
        lshade_customc = LpsrSuccessHistoryAdaptiveDifferentialEvolution(init_pop_size=10, extern_arc_rate=2.0,
                                                                         pbest_factor=0.2, hist_mem_size=5,
                                                                         seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, lshade_custom, lshade_customc, MyProblem())

    def test_griewank(self):
        lshade_griewank = LpsrSuccessHistoryAdaptiveDifferentialEvolution(init_pop_size=10, extern_arc_rate=2.0,
                                                                          pbest_factor=0.2, hist_mem_size=5,
                                                                          seed=self.seed)
        lshade_griewankc = LpsrSuccessHistoryAdaptiveDifferentialEvolution(init_pop_size=10, extern_arc_rate=2.0,
                                                                           pbest_factor=0.2, hist_mem_size=5,
                                                                           seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, lshade_griewank, lshade_griewankc)
