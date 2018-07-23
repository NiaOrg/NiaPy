# encoding=utf8
# pylint: disable=mixed-indentation
from unittest import TestCase
import math

from NiaPy.benchmarks.utility import Utility


class TestBenchmarkFunctions(TestCase):

	# pylint: disable=too-many-instance-attributes,too-many-public-methods
	def setUp(self):
		self.D = 5
		self.array = [0, 0, 0, 0, 0]
		self.array2 = [1, 1, 1, 1, 1]
		self.array3 = [420.968746, 420.968746, 420.968746, 420.968746, 420.968746]
		self.array4 = [-2.903534, -2.903534]
		self.array5 = [-0.5, -0.5, -0.5, -0.5, -0.5]
		self.array6 = [-1, -1, -1, -1, -1]
		self.array7 = [2, 2, 2, 2, 2]
		self.array8 = [7.9170526982459462172, 7.9170526982459462172, 7.9170526982459462172, 7.9170526982459462172, 7.9170526982459462172]
		self.array9 = [-5.12, -5.12, -5.12, -5.12, -5.12]

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
		schumerSteiglitz = Utility().get_benchmark('schumerSteiglitz')
		fun = schumerSteiglitz.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_salomon(self):
		salomon = Utility().get_benchmark('salomon')
		fun = salomon.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_quintic(self):
		quintic = Utility().get_benchmark('quintic')
		fun = quintic.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array6), 0.0)

	def test_quintic2(self):
		quintic = Utility().get_benchmark('quintic')
		fun = quintic.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array7), 0.0)

	def test_pinter(self):
		pinter = Utility().get_benchmark('pinter')
		fun = pinter.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_alpine1(self):
		alpine1 = Utility().get_benchmark('alpine1')
		fun = alpine1.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_alpine2(self):
		alpine2 = Utility().get_benchmark('alpine2')
		fun = alpine2.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array8), math.pow(2.8081311800070053291, self.D))

	def test_chungReynolds(self):
		chungReynolds = Utility().get_benchmark('chungReynolds')
		fun = chungReynolds.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_csendes(self):
		csendes = Utility().get_benchmark('csendes')
		fun = csendes.function()
		self.assertTrue(callable(fun))
		self.assertEqual(fun(self.D, self.array), 0.0)

	def test_bentcigar(self):
		fun = Utility().get_benchmark('bentcigar').function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, [1, 2, 3, 4, 5]), 54000001.0, delta=1e-4)

	def test_discus(self):
		fun = Utility().get_benchmark('discus').function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, [1, 2, 3, 4, 5]), 1000054.0, delta=1e-4)

	def test_elliptic(self):
		fun = Utility().get_benchmark('elliptic').function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, [1, 2, 3, 4, 5]), 5129555.351959938, delta=2e6)

	def test_expanded_griewank_plus_rosnbrock(self):
		fun = Utility().get_benchmark('expandedgriewankplusrosenbrock').function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, [1, 2, 3, 4, 5]), 854123.5271421941, delta=1)

	def test_expanded_scaffer(self):
		fun = Utility().get_benchmark('expandedscaffer').function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, [1, 2, 3, 4, 5]), 2.616740208857464, delta=1e-4)

	def test_hgbat(self):
		fun = Utility().get_benchmark('hgbat').function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, [1, 2, 3, 4, 5]), 61.91502622129181, delta=60)

	def test_katsuura(self):
		fun = Utility().get_benchmark('katsuura').function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, [1, 2, 3, 4, 5]), 3837.4739882594373, delta=4000)

	def test_modifiedscwefel(self):
		fun = Utility().get_benchmark('modifiedscwefel').function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, [1, 2, 3, 4, 5]), 6.9448853328785844, delta=350)

	def test_weierstrass(self):
		fun = Utility().get_benchmark('weierstrass').function()
		self.assertTrue(callable(fun))
		self.assertAlmostEqual(fun(self.D, [1, 2, 3, 4, 5]), 0.0, delta=1e-4)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
