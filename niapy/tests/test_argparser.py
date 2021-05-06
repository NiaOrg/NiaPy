# encoding=utf8
from unittest import TestCase

from niapy.util import get_argparser, get_args, get_args_dict


class ArgParserTestCase(TestCase):
    def setUp(self):
        self.parser = get_argparser()

    def test_parser(self):
        self.assertTrue(self.parser)

    def test_getArgs(self):
        args = get_args(['-d', '10', '--max-evals', '100000000', '-a', 'SCA'])
        self.assertTrue(args)
        self.assertEqual(args.dimension, 10)
        self.assertEqual(args.max_evals, 100000000)
        self.assertEqual(args.algo, 'SCA')

    def test_getDictArgs(self):
        args = get_args_dict(['-d', '10', '--max-evals', '100000000', '-a', 'SCA'])
        self.assertTrue(args)
        self.assertEqual(args['dimension'], 10)
        self.assertEqual(args['max_evals'], 100000000)
        self.assertEqual(args['algo'], 'SCA')
        self.assertEqual(args['seed'], [None])

    def test_getDictArgs_seed(self):
        args = get_args_dict(['-d', '10', '--max-evals', '100000000', '-a', 'SCA', '--seed', '1'])
        self.assertTrue(args)
        self.assertEqual(args['dimension'], 10)
        self.assertEqual(args['max_evals'], 100000000)
        self.assertEqual(args['algo'], 'SCA')
        self.assertEqual(args['seed'], [1])

    def test_getDictArgs_seed_fine_two(self):
        args = get_args_dict(['-d', '10', '--max-evals', '100000000', '-a', 'SCA', '--seed', '1', '234', '231523'])
        self.assertTrue(args)
        self.assertEqual(args['dimension'], 10)
        self.assertEqual(args['max_evals'], 100000000)
        self.assertEqual(args['algo'], 'SCA')
        self.assertEqual(args['seed'], [1, 234, 231523])

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
