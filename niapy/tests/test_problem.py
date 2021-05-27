# encoding=utf8
import logging
from unittest import TestCase

import numpy as np

from niapy.problems import Problem

logging.basicConfig()
logger = logging.getLogger('niapy.test')
logger.setLevel('INFO')


class Dummy(Problem):
    def _evaluate(self, x):
        return np.inf


class ProblemTestCase(TestCase):
    def setUp(self):
        self.Lower, self.Upper = np.array([-19, -10]), np.array([19, 5])
        self.bc = Dummy
        self.b = self.bc(2, self.Lower, self.Upper)

    def test_lower(self):
        self.assertTrue(np.array_equal(self.Lower, self.b.lower))

    def test_upper(self):
        self.assertTrue(np.array_equal(self.Upper, self.b.upper))

    def test_function_eval(self):
        self.assertEqual(np.inf, self.b.evaluate(self.Upper))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
