# encoding=utf8

from math import pow
from unittest import TestCase

import numpy as np

from niapy.util.factory import get_problem


class TestProblemFunctions(TestCase):
    """Testing the problems."""

    def setUp(self):
        """Set up the tests."""
        self.dimension = 5
        self.array = np.asarray([0, 0, 0, 0, 0])
        self.array2 = np.asarray([1, 1, 1, 1, 1])
        self.array3 = np.asarray([420.968746, 420.968746, 420.968746, 420.968746, 420.968746])
        self.array4 = np.asarray([-2.903534, -2.903534])
        self.array5 = np.asarray([-0.5, -0.5, -0.5, -0.5, -0.5])
        self.array6 = np.asarray([-1, -1, -1, -1, -1])
        self.array7 = np.asarray([2, 2, 2, 2, 2])
        self.array8 = np.asarray(
            [7.9170526982459462172, 7.9170526982459462172, 7.9170526982459462172, 7.9170526982459462172,
             7.9170526982459462172])
        self.array9 = np.asarray([-5.12, -5.12, -5.12, -5.12, -5.12])
        self.array10 = np.asarray([1, 2, 3, 4, 5])

    def test_rastrigin(self):
        """Test the rastrigin function."""
        rastrigin = get_problem('rastrigin', dimension=self.dimension)
        self.assertEqual(rastrigin(self.array), 0.0)

    def test_rosenbrock(self):
        """Test the rosenbrock function."""
        fun = get_problem('rosenbrock', self.dimension)
        self.assertEqual(fun(self.array2), 0.0)

    def test_griewank(self):
        """Test the griewank function."""
        fun = get_problem('griewank', self.dimension)
        self.assertEqual(fun(self.array), 0.0)

    def test_sphere(self):
        """Test the sphere function."""
        fun = get_problem('sphere', self.dimension)
        self.assertEqual(fun(self.array), 0.0)

    def test_ackley(self):
        """Test the ackley function."""
        fun = get_problem('ackley', self.dimension)
        self.assertAlmostEqual(fun(self.array), 0.0, places=10)

    def test_schwefel(self):
        """Test the schwefel function."""
        fun = get_problem('schwefel', self.dimension)
        self.assertAlmostEqual(fun(self.array3), 0.0, places=3)

    def test_schwefel221(self):
        """Test the schwefel 221 function."""
        fun = get_problem('schwefel221', self.dimension)
        self.assertEqual(fun(self.array), 0.0)

    def test_schwefel222(self):
        """Test the schwefel 222 function."""
        fun = get_problem('schwefel222', self.dimension)
        self.assertEqual(fun(self.array), 0.0)

    def test_whitley(self):
        """Test the whitley function."""
        fun = get_problem('whitley', self.dimension)
        self.assertEqual(fun(self.array2), 0.0)

    def test_styblinski_tang(self):
        """Test the styblinski tang function."""
        fun = get_problem('styblinski_tang', dimension=2)
        self.assertAlmostEqual(fun(self.array4), -78.332, places=3)

    def test_sum_squares(self):
        """Test the sum squares function."""
        fun = get_problem('sum_squares', self.dimension)
        self.assertEqual(fun(self.array), 0.0)

    def test_stepint(self):
        """Test the stepint function."""
        fun = get_problem('stepint', self.dimension)
        self.assertEqual(fun(self.array9), -5.0)

    def test_step(self):
        """Test the step function."""
        fun = get_problem('step', self.dimension)
        self.assertEqual(fun(self.array), 0.0)

    def test_step2(self):
        """Test the step 2 function."""
        fun = get_problem('step2', self.dimension)
        self.assertEqual(fun(self.array5), 0.0)

    def test_step3(self):
        """Test the step3 function."""
        fun = get_problem('step3', self.dimension)
        self.assertEqual(fun(self.array), 0.0)

    def test_schumer_steiglitz(self):
        """Test the schumer steiglitz function."""
        fun = get_problem('schumer_steiglitz', self.dimension, -100, 100)
        self.assertEqual(fun(self.array), 0.0)

    def test_salomon(self):
        """Test the salomon function."""
        fun = get_problem('salomon', self.dimension, -100.0, 100.0)
        self.assertEqual(fun(self.array), 0.0)

    def test_quintic(self):
        """Test the quintic function."""
        fun = get_problem('quintic', self.dimension, -10.0, 10.0)
        self.assertEqual(fun(self.array6), 0.0)

    def test_quintic2(self):
        """Test the quintic function."""
        fun = get_problem('quintic', self.dimension, -10.0, 10.0)
        self.assertEqual(fun(self.array7), 0.0)

    def test_pinter(self):
        """Test the pinter function."""
        fun = get_problem('pinter', self.dimension, -10.0, 10.0)
        self.assertEqual(fun(self.array), 0.0)

    def test_alpine1(self):
        """Test the alpine 1 function."""
        fun = get_problem('alpine1', self.dimension, -10.0, 10.0)
        self.assertEqual(fun(self.array), 0.0)

    def test_alpine2(self):
        """Test the alpine 2 function."""
        fun = get_problem('alpine2', self.dimension, 0.0, 10.0)
        self.assertEqual(fun(self.array8), pow(2.8081311800070053291, self.dimension))

    def test_chung_reynolds(self):
        """Test the chung reynolds function."""
        fun = get_problem('chung_reynolds', self.dimension, -100, 100)
        self.assertEqual(fun(self.array), 0.0)

    def test_csendes(self):
        """Test the csendes function."""
        fun = get_problem('csendes', self.dimension, -1.0, 1.0)
        self.assertEqual(fun(self.array), 0.0)

    def test_bent_cigar(self):
        """Test the bent cigar function."""
        fun = get_problem('bent_cigar', 2, -100, 100)
        self.assertAlmostEqual(fun(np.zeros(2)), 0.0, delta=1e-4)

    def test_discus(self):
        """Test the discus function."""
        fun = get_problem('discus', self.dimension, -100, 100)
        self.assertAlmostEqual(fun(self.array10), 1000054.0, delta=1e-4)

    def test_elliptic(self):
        """Test the elliptic function."""
        fun = get_problem('elliptic', self.dimension, -100, 100)
        self.assertAlmostEqual(fun(self.array10), 5129555.351959938, delta=2e6)

    def test_expanded_griewank_plus_rosenbrock(self):
        """Test the expanded griewank plus rosenbrock function."""
        fun = get_problem('expanded_griewank_plus_rosenbrock', self.dimension, -100, 100)
        self.assertAlmostEqual(fun(self.array), 2.2997, delta=1e2)

    def test_expanded_schaffer(self):
        """Test the expanded schaffer function."""
        fun = get_problem('expanded_schaffer', self.dimension, -100, 100)
        self.assertAlmostEqual(fun(self.array10), 2.616740208857464, delta=1e-4)

    def test_schaffer_n2(self):
        """Test the schaffer n. 2 function."""
        fun = get_problem('schaffer2', lower=-100, upper=100)
        self.assertAlmostEqual(fun(self.array10[:2]), 0.02467, delta=1e-4)

    def test_schaffer_n4(self):
        """Test the schaffer n. 4 function."""
        fun = get_problem('schaffer4', lower=-100, upper=100)
        self.assertAlmostEqual(fun(self.array10[:2]), 0.97545, delta=1e-4)

    def test_hgbat(self):
        """Test the hgbat function."""
        fun = get_problem('hgbat', self.dimension, -100, 100)
        self.assertAlmostEqual(fun(self.array10), 61.91502622129181, delta=60)

    def test_katsuura(self):
        """Test the katsuura function."""
        fun = get_problem('katsuura', self.dimension, -100, 100)
        self.assertAlmostEqual(fun(self.array10), 3837.4739882594373, delta=4000)

    def test_modified_schwefel(self):
        """Test the modified schwefel function."""
        fun = get_problem('modified_schwefel', self.dimension, -100, 100)
        self.assertAlmostEqual(fun(self.array10), 6.9448853328785844, delta=350)

    def test_weierstrass(self):
        """Test the weierstrass function."""
        fun = get_problem('weierstrass', self.dimension, -100, 100)
        self.assertAlmostEqual(fun(self.array10), 0.0, delta=1e-4)

    def test_happy_cat(self):
        """Test the happy cat function."""
        fun = get_problem('happy_cat', self.dimension, alpha=0.125)
        self.assertAlmostEqual(fun(self.array10), 15.1821333, delta=1e-4)

    def test_qing(self):
        """Test the qing function."""
        fun = get_problem('qing', self.dimension, -500, 500)
        self.assertAlmostEqual(fun(self.array10), 584.0, delta=1e-4)

    def test_ridge(self):
        """Test the ridge function."""
        fun = get_problem('ridge', dimension=self.dimension, lower=-64, upper=64)
        self.assertAlmostEqual(fun(self.array10), 371.0, delta=1e-4)

    def test_michalewicz(self):
        """Test the michalewicz function."""
        fun = get_problem('michalewicz', dimension=2, lower=0, upper=np.pi)
        self.assertAlmostEqual(fun(np.asarray([2.20, 1.57])), -1.8013, delta=1e-3)

    def test_levy(self):
        """Test the levy function."""
        fun = get_problem('levy', dimension=2, lower=0, upper=np.pi)
        self.assertAlmostEqual(fun(np.ones(2)), 0.0)

    def test_sphere2(self):
        """Test the sphere 2 function."""
        fun = get_problem('sphere2', dimension=2, lower=-1, upper=1)
        self.assertAlmostEqual(fun(np.zeros(2)), 0.0)

    def test_sphere3(self):
        """Test the sphere 3 function."""
        fun = get_problem('sphere3', dimension=2, lower=-65.536, upper=65.536)
        self.assertAlmostEqual(fun(np.zeros(2)), 0.0)

    def test_trid(self):
        """Test the trid function."""
        fun = get_problem('trid', dimension=2)
        self.assertAlmostEqual(fun(np.array([2.0, 2.0])), -2.0)

    def test_perm(self):
        """Test the perm function."""
        fun = get_problem('perm', dimension=2)
        self.assertAlmostEqual(fun(np.array([1.0, 0.5])), 0.0)

    def test_zakharov(self):
        """Test the zakharov function."""
        fun = get_problem('zakharov', 2, -5, 10)
        self.assertAlmostEqual(fun(np.zeros(2)), 0.0)

    def test_dixon_price(self):
        """Test the dixon price function."""
        fun = get_problem('dixon_price', 2, -10, 10)
        solution = np.array([1.0, 0.70710678])
        self.assertAlmostEqual(fun(solution), 0.0)

    def test_powell(self):
        """Tests the powell function."""
        fun = get_problem('powell', dimension=2, lower=-4, upper=5)
        self.assertAlmostEqual(fun(np.zeros(2)), 0.0)

    def test_cosine_mixture(self):
        """Test the cosine mixture function."""
        fun = get_problem('cosine_mixture', dimension=2, lower=-1, upper=1)
        self.assertAlmostEqual(fun(np.zeros(2)), -0.2)

    def test_infinity(self):
        """Test the infinity function."""
        infinity = get_problem('infinity', dimension=2, lower=-1, upper=1)
        defaults = np.seterr('raise')
        with self.assertRaises(FloatingPointError):
            infinity(np.zeros(2))
        self.assertAlmostEqual(infinity(np.ones(2)), 5.6829419696157935)
        np.seterr(**defaults)
