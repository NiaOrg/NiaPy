# encoding=utf8
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
		self.bc = Benchmark
		self.b = self.bc(self.Lower, self.Upper)

	def test_lower_fine(self):
		self.assertEqual(self.Lower, self.b.Lower)

	def test_upper_fine(self):
		self.assertEqual(self.Upper, self.b.Upper)

	def test_function_eval_fine(self):
		f = self.b.function()
		self.assertTrue(callable(f))
		self.assertEqual(inf, f(1, self.Upper))

	def test_latex_code_fine(self):
		info = self.bc.latex_code()
		self.assertIsNotNone(info)

	def test_call_operator_fine(self):
		f = self.b()
		self.assertIsNotNone(f)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
