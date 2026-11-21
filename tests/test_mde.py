# encoding=utf8
import numpy as np

from niapy.algorithms.modified import MultiPopulationDifferentialEvolution
from niapy.task import Task
from niapy.util.factory import get_algorithm
from tests.test_algorithm import AlgorithmTestCase, MyProblem


class MDETestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MultiPopulationDifferentialEvolution

    def test_custom(self):
        mde_custom = self.algo(population_size=30, seed=self.seed)
        mde_customc = self.algo(population_size=30, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mde_custom, mde_customc, MyProblem())

    def test_griewank(self):
        mde_griewank = self.algo(population_size=30, seed=self.seed)
        mde_griewankc = self.algo(population_size=30, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mde_griewank, mde_griewankc)

    def test_minimal_population(self):
        mde_min = self.algo(population_size=9, seed=self.seed)
        mde_minc = self.algo(population_size=9, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mde_min, mde_minc, 'rastrigin')

    def test_population_too_small(self):
        self.assertRaises(ValueError, self.algo, population_size=8)
        mde = self.algo()
        self.assertRaises(ValueError, mde.set_parameters, population_size=6)

    def test_parameters(self):
        mde = self.algo(population_size=60)
        self.assertEqual(mde.get_parameters()['population_size'], 60)
        mde.set_parameters(population_size=30)
        self.assertEqual(mde.get_parameters()['population_size'], 30)

    def test_factory(self):
        self.assertIsInstance(get_algorithm('MultiPopulationDifferentialEvolution'), self.algo)

    def test_run_iteration(self):
        mde = self.algo(population_size=30, seed=self.seed)
        task = Task(problem=MyProblem(dimension=5), max_iters=10)
        pop, fpop, d = mde.init_population(task)
        self.assertEqual(pop.shape, (30, 5))
        self.assertEqual(d['noise'].shape, (10, 5))
        self.assertTrue(np.all(d['noise'] == 0))
        xb, fxb = mde.get_best(pop, fpop)
        evals = task.evals
        pop, fpop, xb_new, fxb_new, d = mde.run_iteration(task, pop, fpop, xb, fxb, **d)
        # one iteration evaluates the sub-population only
        self.assertEqual(task.evals - evals, 10)
        self.assertEqual(pop.shape, (30, 5))
        self.assertEqual(d['noise'].shape, (10, 5))
        self.assertTrue(np.all((pop >= task.lower) & (pop <= task.upper)))
        self.assertLessEqual(fxb_new, fxb)
        self.assertEqual(fxb_new, np.min(fpop))
        self.assertTrue(np.array_equal(xb_new, pop[np.argmin(fpop)]))
