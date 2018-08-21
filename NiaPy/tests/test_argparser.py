# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from unittest import TestCase
from NiaPy.util import MakeArgParser, getArgs, getDictArgs
from NiaPy.algorithms.basic import SineCosineAlgorithm

class ArgParserTestCase(TestCase):
	def setUp(self):
		self.parser = MakeArgParser()

	def test_parser_fine(self):
		self.assertTrue(self.parser)

	def test_griewank_works_fine(self):
		args = getArgs(['-D', '10', '-nFES', '100000000', '-a', 'SCA'])
		self.assertTrue(args)
		self.assertEquals(args.D, 10)
		self.assertEquals(args.nFES, 100000000)
		self.assertEquals(args.algo, 'SCA')

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
