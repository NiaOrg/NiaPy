# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, bad-continuation, multiple-statements, singleton-comparison, unused-argument, no-self-use, trailing-comma-tuple, logging-not-lazy, no-else-return, unused-variable, no-member, old-style-class
"""Implementation of benchmarks utility function."""
import logging
from numpy import inf, arange, meshgrid, vectorize
from matplotlib import pyplot as plt
from matplotlib import cm

logging.basicConfig()
logger = logging.getLogger('NiaPy.benchmarks.benchmark')
logger.setLevel('INFO')

__all__ = ['Benchmark']

class Benchmark:
	def __init__(self, Lower, Upper, **kwargs):
		self.Lower, self.Upper = Lower, Upper

	def function(self):
		r"""Get the optimization function."""
		def fun(D, X): return inf
		return fun

	def plot2d(self): pass

	def __2dfun(self, x, y, f): return f(2, x, y)

	def plot3d(self, scale=0.32):
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		func = self.function()
		Xr, Yr = arange(self.Lower, self.Upper, scale), arange(self.Lower, self.Upper, scale)
		X, Y = meshgrid(Xr, Yr)
		Z = vectorize(self.__2dfun)(X, Y, func)
		ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
		cset = ax.contourf(X, Y, Z, zdir='z', offset=-10, cmap=cm.coolwarm)
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		plt.show()

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
