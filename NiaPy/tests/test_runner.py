# encoding=utf8
from unittest import TestCase

import NiaPy
from NiaPy.benchmarks import Benchmark

class MyBenchmark(Benchmark):
	def __init__(self):
		Benchmark.__init__(self, -11, 11)

	@classmethod
	def function(cls):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val = val + sol[i] * sol[i]
			return val
		return evaluate

class RunnerTestCase(TestCase):
	def setUp(self):
		self.algorithms = ['DifferentialEvolution', 'GreyWolfOptimizer', 'GeneticAlgorithm']
		self.benchmarks = ['griewank', MyBenchmark()]

	def test_runner_works_fine(self):
		self.assertTrue(NiaPy.Runner(7, 100, 2, self.algorithms, self.benchmarks).run())

	def test_runner_bad_algorithm_thorws_fine(self):
		self.assertRaises(TypeError, lambda: NiaPy.Runner(4, 10, 3, 'EvolutionStrategy', self.benchmarks).run())

	def test_runner_bad_benchmark_throws_fine(self):
		self.assertRaises(TypeError, lambda: NiaPy.Runner(4, 10, 3, 'EvolutionStrategy1p1', 'TesterMan').run())

	def test_runner_bad_export_throws_fine(self):
		self.assertRaises(TypeError, lambda: NiaPy.Runner(4, 10, 3, 'GreyWolfOptimizer', self.benchmarks).run(export="pandas"))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
