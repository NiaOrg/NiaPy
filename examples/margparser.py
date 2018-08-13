import sys
from argparse import ArgumentParser

def makeArgParser():
	parser = ArgumentParser(description='Runer example.')
	parser.add_argument('-D', dest='D', default=10, type=int)
	parser.add_argument('-nFES', dest='nFES', default=50000, type=int)
	parser.add_argument('-runType', dest='runType', choices=['', 'log', 'plot'], default='log', type=str)
	parser.add_argument('-seed', dest='seed', default=None, type=int)
	parser.add_argument('-optType', dest='optType', default='min', choices=['min', 'max'], type=str)
	return parser

def getArgs(argv):
	parser = makeArgParser()
	args = parser.parse_args(argv)
	return args

if __name__ == '__main__':
	args = getArgs(sys.argv[1:])
	print (args)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
