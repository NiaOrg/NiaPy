# encoding=utf8
from unittest import TestCase

from numpy.random import default_rng

from niapy.algorithms.modified import ImprovedLpsrSuccessHistoryAdaptiveDifferentialEvolution
from niapy.algorithms.modified.ilshade import SolutionILSHADE
from niapy.task import Task
from tests.test_algorithm import AlgorithmTestCase, MyProblem


class SolutionILSHADETestCase(TestCase):
    def setUp(self):
        self.D, self.F, self.CR = 10, 0.9, 0.3
        self.x, self.task = default_rng().uniform(10, 50, self.D), Task(problem=MyProblem(self.D))
        self.s1, self.s2 = SolutionILSHADE(task=self.task, e=False), SolutionILSHADE(differential_weight=self.F,
                                                                                 crossover_probability=self.CR,
                                                                                 x=self.x)

    def test_F(self):
        self.assertAlmostEqual(self.s1.differential_weight, 0.5)
        self.assertAlmostEqual(self.s2.differential_weight, self.F)

    def test_CR(self):
        self.assertAlmostEqual(self.s1.crossover_probability, 0.8)
        self.assertAlmostEqual(self.s2.crossover_probability, self.CR)


class ILSHADETestCase(AlgorithmTestCase):

    def test_custom(self):
        ilshade_custom = ImprovedLpsrSuccessHistoryAdaptiveDifferentialEvolution(init_pop_size=10, extern_arc_rate=2.6,
                                                                   pbest_start=0.2, pbest_end=0.1, hist_mem_size=6, seed=self.seed)
        ilshade_customc = ImprovedLpsrSuccessHistoryAdaptiveDifferentialEvolution(init_pop_size=10, extern_arc_rate=2.6,
                                                                    pbest_start=0.2, pbest_end=0.1, hist_mem_size=6, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ilshade_custom, ilshade_customc, MyProblem())

    def test_griewank(self):
        shade_griewank = ImprovedLpsrSuccessHistoryAdaptiveDifferentialEvolution(init_pop_size=10, extern_arc_rate=2.6,
                                                                     pbest_start=0.2, pbest_end=0.1, hist_mem_size=6, seed=self.seed)
        shade_griewankc = ImprovedLpsrSuccessHistoryAdaptiveDifferentialEvolution(init_pop_size=10, extern_arc_rate=2.6,
                                                                      pbest_start=0.2, pbest_end=0.1, hist_mem_size=6, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, shade_griewank, shade_griewankc)
