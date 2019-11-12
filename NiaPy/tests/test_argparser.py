# encoding=utf8
from unittest import TestCase
from NiaPy.util import MakeArgParser, getArgs, getDictArgs

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
		self.assertEqual(args['seed'], [None])

	def test_getDictArgs_seed_fine(self):
		args = getDictArgs(['-D', '10', '-nFES', '100000000', '-a', 'SCA', '-seed', '1'])
		self.assertTrue(args)
		self.assertEqual(args['D'], 10)
		self.assertEqual(args['nFES'], 100000000)
		self.assertEqual(args['algo'], 'SCA')
		self.assertEqual(args['seed'], [1])

	def test_getDictArgs_seed_fine_two(self):
		args = getDictArgs(['-D', '10', '-nFES', '100000000', '-a', 'SCA', '-seed', '1', '234', '231523'])
		self.assertTrue(args)
		self.assertEqual(args['D'], 10)
		self.assertEqual(args['nFES'], 100000000)
		self.assertEqual(args['algo'], 'SCA')
		self.assertEqual(args['seed'], [1, 234, 231523])

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
