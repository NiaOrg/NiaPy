# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, old-style-class
from unittest import TestCase
from NiaPy.argparser import MakeArgParser, getArgs, getDictArgs

class ArgParserTestCase(TestCase):
	def setUp(self):
		self.parser = MakeArgParser()

	def test_parser_fine(self):
		self.assertTrue(self.parser)

	def test_getArgs_fine(self):
		args = getArgs(['-D', '10', '-nFES', '100000000', '-a', 'SCA'])
		self.assertTrue(args)
		self.assertEqual(args.D, 10)
		self.assertEqual(args.nFES, 100000000)
		self.assertEqual(args.algo, 'SCA')

	def test_getDictArgs_fine(self):
		args = getDictArgs(['-D', '10', '-nFES', '100000000', '-a', 'SCA'])
		self.assertTrue(args)
		self.assertEqual(args['D'], 10)
		self.assertEqual(args['nFES'], 100000000)
		self.assertEqual(args['algo'], 'SCA')

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
