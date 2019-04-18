# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from unittest import TestCase

import numpy as np

from NiaPy.algorithms import BasicStatistics

class BasicStatisticsTestCase(TestCase):
	r"""Test case for BasicStatistics class.

	Date:
		April 2019

	Author:
		Klemen Berkovič
	"""
	def setUp(self):
		self.x = np.random.uniform(-100, 100, 1000)
		self.stats = BasicStatistics(self.x)

	def test_min_value(self):
		self.assertEqual(self.x.min(), self.stats.min_value())

	def test_max_value(self):
		self.assertEqual(self.x.max(), self.stats.max_value())

	def test_mean(self):
		self.assertEqual(np.mean(self.x), self.stats.mean())

	def test_standard_deviation(self):
		self.assertEqual(self.x.std(ddof=1), self.stats.standard_deviation())

	def test_generate_standard_report(self):
		self.assertIsNotNone(self.stats.generate_standard_report())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
