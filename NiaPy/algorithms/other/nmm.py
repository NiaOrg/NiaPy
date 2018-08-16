# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, line-too-long, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, arguments-differ, redefined-builtin
import logging
from numpy import apply_along_axis, argsort, argmin, sum
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['NelderMeadMethod']

class NelderMeadMethod(Algorithm):
	r"""Implementation of Nelder Mead method or downhill simplex method or amoeba method.

	**Algorithm:** Nelder Mead Method

	**Date:** 2018

	**Authors:** Klemen Berkovič

	**License:** MIT

	**Reference URL:** https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
	"""
	def __init__(self, **kwargs): Algorithm.__init__(self, name='NelderMeadMethod', sName='NMM', **kwargs)

	def setParameters(self, alpha=1.0, gamma=2.0, rho=-0.5, sigma=0.5, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments:**

		alpha {real} -- Reflection coefficient parameter

		gamma {real} -- Expansion coefficient parameter

		rho {real} -- Contraction coefficient parameter

		sigma {real} -- Shrink coefficient parameter
		"""
		self.alpha, self.gamma, self.rho, self.sigma = alpha, gamma, rho, sigma
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def init(self, task):
		X = self.uniform(task.Lower, task.Upper, [task.D, task.D])
		X_f = apply_along_axis(task.eval, 1, X)
		return X, X_f

	def method(self, X, X_f, task):
		x0 = sum(X[:-1], axis=0) / (len(X) - 1)
		xr = x0 + self.alpha * (x0 - X[-1])
		rs = task.eval(xr)
		if X_f[0] >= rs < X_f[-2]:
			X[-1], X_f[-1] = xr, rs
			return X, X_f
		if rs < X_f[0]:
			xe = x0 + self.gamma * (x0 - X[-1])
			re = task.eval(xe)
			if re < rs: X[-1], X_f[-1] = xe, re
			else: X[-1], X_f[-1] = xr, rs
			return X, X_f
		xc = x0 + self.rho * (x0 - X[-1])
		rc = task.eval(xc)
		if rc < X_f[-1]:
			X[-1], X_f[-1] = xc, rc
			return X, X_f
		Xn = X[0] + self.sigma * (X[1:] - X[0])
		Xn_f = apply_along_axis(task.eval, 1, Xn)
		X[1:], X_f[1:] = Xn, Xn_f
		return X, X_f

	def runTask(self, task):
		X, X_f = self.init(task)
		while not task.stopCond():
			inds = argsort(X_f)
			X, X_f = X[inds], X_f[inds]
			X, X_f = self.method(X, X_f, task)
		ib = argmin(X_f)
		return X[ib], X_f[ib]

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
