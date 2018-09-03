# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from unittest import TestCase
from NiaPy.algorithms.basic import ArtificialBeeColonyAlgorithm

class MyBenchmark:
	def __init__(self):
		self.Lower = -5.12
		self.Upper = 5.12

	@classmethod
	def function(cls):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D):
				val = val + sol[i] * sol[i]
			return val
		return evaluate

class ABCTestCase(TestCase):
	def setUp(self):
		self.abc_custom = ArtificialBeeColonyAlgorithm(NP=10, D=40, nFES=4000, benchmark=MyBenchmark(), seed=1)
		self.abc_customc = ArtificialBeeColonyAlgorithm(NP=10, D=40, nFES=4000, benchmark=MyBenchmark(), seed=1)
		self.abc_griewank = ArtificialBeeColonyAlgorithm(NP=10, D=40, nFES=4000, benchmark='griewank', seed=1)
		self.abc_griewankc = ArtificialBeeColonyAlgorithm(NP=10, D=40, nFES=4000, benchmark='griewank', seed=1)

	def test_custom_works_fine(self):
		r1 = self.abc_custom.run()
		self.assertTrue(r1)
		r2 = self.abc_customc.run()
		self.assertTrue(r2)
		self.assertEqual(r1[0], r2[0])
		self.assertEqual(r1[1], r2[1])

	def test_griewank_works_fine(self):
		r1 = self.abc_griewank.run()
		self.assertTrue(r1)
		r2 = self.abc_griewankc.run()
		self.assertTrue(r2)
		self.assertEqual(r1[0], r2[0])
		self.assertEqual(r1[1], r2[1])

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
