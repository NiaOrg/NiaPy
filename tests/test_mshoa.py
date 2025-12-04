# encoding=utf8
import numbers
import numpy as np
from niapy.algorithms.basic import MantisSearchAlgorithm
from niapy.task import Task
from tests.test_algorithm import AlgorithmTestCase, MyProblem


class MShOATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MantisSearchAlgorithm

    def test_algorithm_info(self):
        al = self.algo.info()
        self.assertIsNotNone(al)

    def test_custom(self):
        mshoa_custom = self.algo(population_size=10, seed=self.seed)
        mshoa_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mshoa_custom, mshoa_customc, MyProblem())

    def test_griewank(self):
        mshoa_griewank = self.algo(population_size=10, seed=self.seed)
        mshoa_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mshoa_griewank, mshoa_griewankc)

    def test_parameters(self):
        mshoa = self.algo(population_size=20, k_value=0.3, use_reflection=True, normalize_space=False, debug_pti=False, seed=self.seed)
        params = mshoa.get_parameters()
        self.assertEqual(params['population_size'], 20)
        self.assertEqual(params['k_value'], 0.3)
        self.assertTrue(params['use_reflection'])
        self.assertFalse(params['normalize_space'])
        self.assertFalse(params['debug_pti'])

    def test_run_iteration(self):
        """Test that run_iteration works correctly and returns proper shapes."""
        mshoa = self.algo(population_size=10, seed=self.seed)
        task = Task(problem=MyProblem(dimension=5), max_iters=2)
        # Initialize population
        pop, fpop, d = mshoa.init_population(task)
        
        # Check initial population shape
        self.assertEqual(pop.shape, (10, 5))
        self.assertEqual(fpop.shape, (10,))
        self.assertIn('pti', d)
        self.assertEqual(d['pti'].shape, (10,))
        
        # Get initial best
        xb, fxb = mshoa.get_best(pop, fpop)
        
        # Check initial best
        self.assertEqual(xb.shape, (5,))
        self.assertIsInstance(fxb, numbers.Real)
        
        # Run one iteration
        pop_new, fpop_new, xb_new, fxb_new, d_new = mshoa.run_iteration(task, pop, fpop, xb, fxb, **d)
        
        # Check shapes after iteration
        self.assertEqual(pop_new.shape, (10, 5))
        self.assertEqual(fpop_new.shape, (10,))
        self.assertEqual(xb_new.shape, (5,))
        self.assertIsInstance(fxb_new, numbers.Real)
        self.assertIn('pti', d_new)
        self.assertEqual(d_new['pti'].shape, (10,))
        
        # Check PTI values are valid (1, 2, or 3)
        self.assertTrue(np.all((d_new['pti'] >= 1) & (d_new['pti'] <= 3)))

