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
		self.abc_custom = ArtificialBeeColonyAlgorithm(NP=10, D=40, nFES=4000, benchmark=MyBenchmark())
		self.abc_griewank = ArtificialBeeColonyAlgorithm(NP=10, D=40, nFES=4000, benchmark='griewank')

	def test_custom_works_fine(self):
		self.assertTrue(self.abc_custom.run())

	def test_griewank_works_fine(self):
		self.assertTrue(self.abc_griewank.run())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
