# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, line-too-long
import logging
from unittest import TestCase

from numpy import inf

from NiaPy.benchmarks import Benchmark

logging.basicConfig()
logger = logging.getLogger('NiaPy.test')
logger.setLevel('INFO')

class BenchmarkTestCase(TestCase):
	def setUp(self):
		self.Lower, self.Upper = [-19, -10], [19, 5]
		self.b = Benchmark(self.Lower, self.Upper)

	def test_lower_fine(self):
		self.assertEqual(self.Lower, self.b.Lower)

	def test_upper_fine(self):
		self.assertEqual(self.Upper, self.b.Upper)

	def test_function_eval_fine(self):
		f = self.b.function()
		self.assertTrue(callable(f))
		self.assertEqual(inf, f(1, self.Upper))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
