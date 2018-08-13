# encoding=utf8
# pylint: disable=mixed-indentation, redefined-builtin, too-many-instance-attributes,too-many-public-methods, multiple-statements, no-self-use
from math import pow, isnan
from unittest import TestCase
from numpy import asarray, pi, full
from NiaPy.benchmarks.utility import Utility

class TestBenchmarkFunctions(TestCase):
	def setUp(self):
		self.D = 5
		self.array = asarray([0, 0, 0, 0, 0])
		self.array2 = asarray([1, 1, 1, 1, 1])
		self.array3 = asarray([420.968746, 420.968746, 420.968746, 420.968746, 420.968746])
		self.array4 = asarray([-2.903534, -2.903534])
		self.array5 = asarray([-0.5, -0.5, -0.5, -0.5, -0.5])
		self.array6 = asarray([-1, -1, -1, -1, -1])
		self.array7 = asarray([2, 2, 2, 2, 2])
		self.array8 = asarray([7.9170526982459462172, 7.9170526982459462172, 7.9170526982459462172, 7.9170526982459462172, 7.9170526982459462172])
		self.array9 = asarray([-5.12, -5.12, -5.12, -5.12, -5.12])
		self.array10 = asarray([1, 2, 3, 4, 5])

	def assertBounds(self, bench, l, u):
		b = Utility().get_benchmark(bench)
		self.assertEqual(b.Lower, l)
		self.assertEqual(b.Upper, u)
		return b.function()

	def test_rastrigin(self):
		rastrigin = Utility().get_benchmark('rastrigin')
		fun = rastrigin.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_rosenbrock(self):
		rosenbrock = Utility().get_benchmark('rosenbrock')
		fun = rosenbrock.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array2), 0.0)

	def test_griewank(self):
		griewank = Utility().get_benchmark('griewank')
		fun = griewank.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_sphere(self):
		sphere = Utility().get_benchmark('sphere')
		fun = sphere.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_ackley(self):
		ackley = Utility().get_benchmark('ackley')
		fun = ackley.function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, self.array), 0.0, places=10)

	def test_schwefel(self):
		schwefel = Utility().get_benchmark('schwefel')
		fun = schwefel.function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, self.array3), 0.0, places=3)

	def test_schwefel221(self):
		schwefel221 = Utility().get_benchmark('schwefel221')
		fun = schwefel221.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_schwefel222(self):
		schwefel222 = Utility().get_benchmark('schwefel222')
		fun = schwefel222.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_whitley(self):
		whitley = Utility().get_benchmark('whitley')
		fun = whitley.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array2), 0.0)

	def test_styblinskiTang(self):
		styblinskiTang = Utility().get_benchmark('styblinskiTang')
		fun = styblinskiTang.function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(2, self.array4), -78.332, places=3)

	def test_sumSquares(self):
		sumSquares = Utility().get_benchmark('sumSquares')
		fun = sumSquares.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_stepint(self):
		stepint = Utility().get_benchmark('stepint')
		fun = stepint.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array9), 25.0 - 6 * self.D)

	def test_step(self):
		step = Utility().get_benchmark('step')
		fun = step.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_step2(self):
		step2 = Utility().get_benchmark('step2')
		fun = step2.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array5), 0.0)

	def test_step3(self):
		step3 = Utility().get_benchmark('step3')
		fun = step3.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_schumerSteiglitz(self):
		fun = self.assertBounds('schumerSteiglitz', -100, 100)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_salomon(self):
		fun = self.assertBounds('salomon', -100.0, 100.0)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_quintic(self):
		fun = self.assertBounds('quintic', -10.0, 10.0)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array6), 0.0)

	def test_quintic2(self):
		fun = self.assertBounds('quintic', -10.0, 10.0)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array7), 0.0)

	def test_pinter(self):
		fun = self.assertBounds('pinter', -10.0, 10.0)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_alpine1(self):
		fun = self.assertBounds('alpine1', -10.0, 10.0)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_alpine2(self):
		fun = self.assertBounds('alpine2', 0.0, 10.0)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array8), pow(2.8081311800070053291, self.D))

	def test_chungReynolds(self):
		fun = self.assertBounds('chungReynolds', -100, 100)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_csendes(self):
		fun = self.assertBounds('csendes', -1.0, 1.0)
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_bentcigar(self):
		fun = self.assertBounds('bentcigar', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(2, full(2, 0)), 0.0, delta=1e-4)
		self.assertAlmostEqual(fun(10, full(10, 0)), 0.0, delta=1e-4)
		self.assertAlmostEqual(fun(100, full(100, 0)), 0.0, delta=1e-4)

	def test_discus(self):
		fun = self.assertBounds('discus', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, self.array10), 1000054.0, delta=1e-4)

	def test_elliptic(self):
		fun = self.assertBounds('elliptic', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, self.array10), 5129555.351959938, delta=2e6)

	def test_expanded_griewank_plus_rosnbrock(self):
		fun = self.assertBounds('expandedgriewankplusrosenbrock', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, self.array), 2.2997, delta=1e2)

	def test_expanded_scaffer(self):
		fun = self.assertBounds('expandedscaffer', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, self.array10), 2.616740208857464, delta=1e-4)

	def test_hgbat(self):
		fun = self.assertBounds('hgbat', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, self.array10), 61.91502622129181, delta=60)

	def test_katsuura(self):
		fun = self.assertBounds('katsuura', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, self.array10), 3837.4739882594373, delta=4000)

	def test_modifiedscwefel(self):
		fun = self.assertBounds('modifiedscwefel', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, self.array10), 6.9448853328785844, delta=350)

	def test_weierstrass(self):
		fun = self.assertBounds('weierstrass', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, self.array10), 0.0, delta=1e-4)

	def test_happyCat(self):
		fun = self.assertBounds('happyCat', -100, 100)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, self.array10), 15.1821333, delta=1e-4)

	def test_qing(self):
		fun = self.assertBounds('qing', -500, 500)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, self.array10), 669.0, delta=1e-4)

	def test_ridge(self):
		fun = self.assertBounds('ridge', -64, 64)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, self.array10), 371.0, delta=1e-4)

	def test_michalewicz(self):
		fun = self.assertBounds('michalewicz', 0, pi)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(2, asarray([2.20, 1.57])), -1.8013, delta=1e-3)

	def test_levy(self):
		fun = self.assertBounds('levy', 0, pi)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(2, full(2, 1.)), 0.0)
		self.assertAlmostEqual(fun(10, full(10, 1.)), 0.0)
		self.assertAlmostEqual(fun(100, full(100, 1.)), 0.0)

	def test_sphere2(self):
		fun = self.assertBounds('sphere2', -1, 1)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(2, full(2, 0.)), 0.0)
		self.assertAlmostEqual(fun(10, full(10, 0.)), 0.0)
		self.assertAlmostEqual(fun(100, full(100, 0.)), 0.0)

	def test_sphere3(self):
		fun = self.assertBounds('sphere3', -65.536, 65.536)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(2, full(2, 0.)), 0.0)
		self.assertAlmostEqual(fun(10, full(10, 0.)), 0.0)
		self.assertAlmostEqual(fun(100, full(100, 0.)), 0.0)

	def __trid_opt(self, d): return -d * (d + 4) * (d - 1) / 6

	def __trid_opt_sol(self, d): return asarray([i * (d + 1 - i) for i in range(1, d + 1)])

	def test_trid(self):
		fun = self.assertBounds('trid', -2 ** 2, 2 ** 2)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(2, self.__trid_opt_sol(2)), self.__trid_opt(2))
		self.assertAlmostEqual(fun(10, self.__trid_opt_sol(10)), self.__trid_opt(10))
		self.assertAlmostEqual(fun(100, self.__trid_opt_sol(100)), self.__trid_opt(100))

	def __perm_opt_sol(self, d): return asarray([1 / i for i in range(1, d + 1)])

	def test_perm(self):
		fun = self.assertBounds('perm', -10, 10)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(2, self.__perm_opt_sol(2)), .0)
		self.assertAlmostEqual(fun(10, self.__perm_opt_sol(10)), .0)
		self.assertAlmostEqual(fun(100, self.__perm_opt_sol(100)), .0)

	def test_zakharov(self):
		fun = self.assertBounds('zakharov', -5, 10)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(2, full(2, .0)), .0)
		self.assertAlmostEqual(fun(10, full(10, .0)), .0)
		self.assertAlmostEqual(fun(100, full(100, .0)), .0)

	def __dixonprice_opt_sol(self, d): return asarray([2 ** (-(2 ** i - 2) / 2 ** i) for i in range(1, d + 1)])

	def test_dixonprice(self):
		fun = self.assertBounds('dixonprice', -10, 10)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(2, self.__dixonprice_opt_sol(2)), .0)
		self.assertAlmostEqual(fun(10, self.__dixonprice_opt_sol(10)), .0)
		self.assertAlmostEqual(fun(100, self.__dixonprice_opt_sol(100)), .0)

	def test_powell(self):
		fun = self.assertBounds('powell', -4, 5)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(2, full(2, .0)), .0)
		self.assertAlmostEqual(fun(10, full(10, .0)), .0)
		self.assertAlmostEqual(fun(100, full(100, .0)), .0)

	def test_cosinemixture(self):
		fun = self.assertBounds('cosinemixture', -1, 1)
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(2, full(2, .0)), -.1 * 2)
		self.assertAlmostEqual(fun(10, full(10, .0)), -.1 * 10)
		self.assertAlmostEqual(fun(100, full(100, .0)), -.1 * 100)

	def test_infinity(self):
		fun = self.assertBounds('infinity', -1, 1)
		self.assertTrue(callable(fun))
		self.assertTrue(isnan(fun(2, full(2, .0))))
		self.assertTrue(isnan(fun(10, full(10, .0))))
		self.assertTrue(isnan(fun(100, full(100, .0))))
		self.assertAlmostEqual(fun(2, full(2, 1e-4)), .0)
		self.assertAlmostEqual(fun(10, full(10, 1e-4)), .0)
		self.assertAlmostEqual(fun(100, full(100, 1e-4)), .0)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
