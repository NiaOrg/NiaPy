# encoding=utf8
import logging
from unittest import TestCase

import numpy as np

from niapy.benchmarks import Benchmark

logging.basicConfig()
logger = logging.getLogger('niapy.test')
logger.setLevel('INFO')


class BenchmarkTestCase(TestCase):
    def setUp(self):
        self.Lower, self.Upper = [-19, -10], [19, 5]
        self.bc = Benchmark
        self.b = self.bc(self.Lower, self.Upper)

    def test_lower(self):
        self.assertEqual(self.Lower, self.b.lower)

    def test_upper(self):
        self.assertEqual(self.Upper, self.b.upper)

    def test_function_eval(self):
        f = self.b.function()
        self.assertTrue(callable(f))
        self.assertEqual(np.inf, f(1, self.Upper))

    def test_latex_code(self):
        info = self.bc.latex_code()
        self.assertIsNotNone(info)

    def test_call_operator(self):
        f = self.b()
        self.assertIsNotNone(f)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
