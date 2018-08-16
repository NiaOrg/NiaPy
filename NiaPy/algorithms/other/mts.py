# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, len-as-condition, singleton-comparison, arguments-differ, line-too-long, unused-argument, consider-using-enumerate
import logging
import operator as oper
from numpy import random as rand, vectorize, where, copy, apply_along_axis, argmin, argmax, argsort, fmin, fmax, full
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['MultipleTrajectorySearch', 'MultipleTrajectorySearchV1', 'MTS_LS1', 'MTS_LS1v1', 'MTS_LS2', 'MTS_LS3', 'MTS_LS3v1']

def MTS_LS1(Xk, Xk_fit, Xb, Xb_fit, improve, SR, task, BONUS1=10, BONUS2=1, rnd=rand):
	grade = 0.0
	if not improve:
		SR /= 2
		ifix = where(SR < 1e-15)
		SR[ifix] = task.bRange[ifix] * 0.4
	improve = False
	for i in range(len(Xk)):
		Xk_i_old = Xk[i]
		Xk[i] = Xk_i_old - SR[i]
		Xk_fit_new = task.eval(Xk)
		if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk, Xk_fit_new
		if Xk_fit_new == Xk_fit: Xk[i] = Xk_i_old
		else:
			if Xk_fit_new > Xk_fit:
				Xk[i] = Xk_i_old + 0.5 * SR[i]
				Xk_fit_new = task.eval(Xk)
				if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, copy(Xk), Xk_fit_new
				if Xk_fit_new >= Xk_fit: Xk[i] = Xk_i_old
				else: grade, improve, Xk_fit = grade + BONUS2, True, Xk_fit_new
			else: grade, improve, Xk_fit = grade + BONUS2, True, Xk_fit_new
	return Xk, Xk_fit, Xb, Xb_fit, improve, grade, SR

def MTS_LS1v1(Xk, Xk_fit, Xb, Xb_fit, improve, SR, task, BONUS1=10, BONUS2=1, rnd=rand):
	grade = 0.0
	if not improve:
		SR /= 2
		ifix = where(SR < 1e-15)
		SR[ifix] = task.bRange[ifix] * 0.4
	improve, D = False, rnd.uniform(-1, 1, task.D)
	for i in range(len(Xk)):
		Xk_i_old = Xk[i]
		Xk[i] = Xk_i_old - SR[i] * D[i]
		Xk_fit_new = task.eval(Xk)
		if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk, Xk_fit_new
		if Xk_fit_new == Xk_fit: Xk[i] = Xk_i_old
		else:
			if Xk_fit_new > Xk_fit:
				Xk[i] = Xk_i_old + 0.5 * SR[i]
				Xk_fit_new = task.eval(Xk)
				if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk, Xk_fit_new
				if Xk_fit_new >= Xk_fit: Xk[i] = Xk_i_old
				else: grade, improve, Xk_fit = grade + BONUS2, True, Xk_fit_new
			else: grade, improve, Xk_fit = grade + BONUS2, True, Xk_fit_new
	return Xk, Xk_fit, Xb, Xb_fit, improve, grade, SR

def genNewX(x, r, d, SR, op): return op(x, SR * d) if r == 0 else x

def MTS_LS2(Xk, Xk_fit, Xb, Xb_fit, improve, SR, task, BONUS1=10, BONUS2=1, rnd=rand):
	grade = 0.0
	if not improve:
		SR /= 2
		ifix = where(SR < 1e-15)
		SR[ifix] = task.bRange[ifix] * 0.4
	improve = False
	for _ in range(len(Xk)):
		D = -1 + rnd.rand(len(Xk)) * 2
		R = rnd.choice([0, 1, 2, 3], len(Xk))
		Xk_new = vectorize(genNewX)(Xk, R, D, SR, oper.sub)
		Xk_fit_new = task.eval(Xk_new)
		if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk_new, Xk_fit_new
		if Xk_fit_new != Xk_fit:
			if Xk_fit_new > Xk_fit:
				Xk_new = vectorize(genNewX)(Xk, R, D, SR, oper.add)
				Xk_fit_new = task.eval(Xk_new)
				if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk_new, Xk_fit_new
				if Xk_fit_new < Xk_fit: grade, improve, Xk, Xk_fit = grade + BONUS2, True, Xk_new, Xk_fit_new
			else: grade, improve, Xk, Xk_fit = grade + BONUS2, True, Xk_new, Xk_fit_new
	return Xk, Xk_fit, Xb, Xb_fit, improve, grade, SR

def MTS_LS3(Xk, Xk_fit, Xb, Xb_fit, improve, SR, task, BONUS1=10, BONUS2=1, rnd=rand):
	Xk_new, grade = copy(Xk), 0.0
	for i in range(len(Xk)):
		Xk1, Xk2, Xk3 = copy(Xk_new), copy(Xk_new), copy(Xk_new)
		Xk1[i], Xk2[i], Xk3[i] = Xk1[i] + 0.1, Xk2[i] - 0.1, Xk3[i] + 0.2
		Xk1_fit, Xk2_fit, Xk3_fit = task.eval(Xk1), task.eval(Xk2), task.eval(Xk3)
		if Xk1_fit < Xb_fit: grade, Xb, Xb_fit, improve = grade + BONUS1, Xk1, Xk1_fit, True
		if Xk2_fit < Xb_fit: grade, Xb, Xb_fit, improve = grade + BONUS1, Xk2, Xk2_fit, True
		if Xk3_fit < Xb_fit: grade, Xb, Xb_fit, improve = grade + BONUS1, Xk3, Xk3_fit, True
		D1, D2, D3 = Xk_fit - Xk1_fit, Xk_fit - Xk2_fit, Xk_fit - Xk3_fit
		if D1 > 0: grade, improve = grade + BONUS2, True
		if D2 > 0: grade, improve = grade + BONUS2, True
		if D3 > 0: grade, improve = grade + BONUS2, True
		a, b, c = 0.4 + rnd.rand() * 0.1, 0.1 + rnd.rand() * 0.2, rnd.rand()
		Xk_new[i] += a * (D1 - D2) + b * (D3 - 2 * D1) + c
		Xk_fit_new = task.eval(Xk_new)
	if Xk_fit_new < Xk_fit: Xk, Xk_fit = Xk_new, Xk_fit_new
	return Xk, Xk_fit, Xb, Xb_fit, improve, grade, SR

def MTS_LS3v1(Xk, Xk_fit, Xb, Xb_fit, improve, SR, task, phi=3, BONUS1=10, BONUS2=1, rnd=rand):
	grade, Disp = 0.0, task.bRange / 10
	while True in Disp > 1e-3:
		Xn = [rnd.permutation(Xk) + Disp * rnd.uniform(-1, 1, len(Xk)) for _ in range(phi)]
		Xn_f = apply_along_axis(task.eval, 1, Xn)
		iBetter, iBetterBest = where(Xn_f < Xk_fit), where(Xn_f < Xb_fit)
		grade += len(iBetterBest) * BONUS1 + (len(iBetter) - len(iBetterBest)) * BONUS2
		if len(Xn_f[iBetterBest]) > 0:
			ib, improve = argmin(Xn_f[iBetterBest]), True
			Xb, Xb_fit, Xk, Xk_fit = Xn[ib], Xn_f[ib], Xn[ib], Xn_f[ib]
		elif len(Xn_f[iBetter]) > 0:
			ib, improve = argmin(Xn_f[iBetter]), True
			Xk, Xk_fit = Xn[ib], Xn_f[ib]
		Su, Sl = fmin(task.Upper, Xk + 2 * Disp), fmax(task.Lower, Xk - 2 * Disp)
		Disp = (Su - Sl) / 10
	return Xk, Xk_fit, Xb, Xb_fit, improve, grade, SR

class MultipleTrajectorySearch(Algorithm):
	BONUS1, BONUS2 = 10, 1
	r"""Implementation of Multiple trajectory search.

	**Algorithm:** Multiple trajectory search

	**Date:** 2018

	**Authors:** Klemen Berkovic

	**License:** MIT

	**Reference URL:** https://ieeexplore.ieee.org/document/4631210/

	**Reference paper:** Lin-Yu Tseng and Chun Chen, "Multiple trajectory search for Large Scale Global Optimization," 2008 IEEE Congress on Evolutionary Computation (IEEE World Congress on Computational Intelligence), Hong Kong, 2008, pp. 3052-3059. doi: 10.1109/CEC.2008.4631210
	"""
	def __init__(self, **kwargs):
		if kwargs.get('name', None) is None: Algorithm.__init__(self, name='MultipleTrajectorySearch', sName='MTS', **kwargs)
		else: Algorithm.__init__(self, **kwargs)
		self.LSs = [MTS_LS1, MTS_LS2, MTS_LS3]

	def setParameters(self, NP=40, NoLsTests=5, NoLs=5, NoLsBest=5, NoEnabled=17, **ukwargs):
		r"""Set the arguments of the algorithm.

		**Arguments:**

		NP, M {integer} -- population size

		NoLsTests {integer} -- number of test runs on local search algorihms

		NoLs {integer} -- number of local search algoritm runs

		NoLsBest {integer} -- number of locals search algorithm runs on best solution

		NoEnabled {integer} -- number of best solution for testing
		"""
		self.M, self.NoLsTests, self.NoLs, self.NoLsBest, self.NoEnabled = NP, NoLsTests, NoLs, NoLsBest, NoEnabled
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def GradingRun(self, x, x_f, xb, xb_f, improve, SR, task):
		ls_grades, Xn, Xnb = full(3, 0.0), [[x, x_f]] * len(self.LSs), [xb, xb_f]
		for _ in range(self.NoLsTests):
			for k in range(len(self.LSs)):
				Xn[k][0], Xn[k][1], xnb, xnb_f, improve, g, SR = self.LSs[k](Xn[k][0], Xn[k][1], Xnb[0], Xnb[1], improve, SR, task, BONUS1=self.BONUS1, BONUS2=self.BONUS2, rnd=self.Rand)
				if Xnb[1] > xnb_f: Xnb = [xnb, xnb_f]
				ls_grades[k] += g
		xn, k = min(Xn, key=lambda x: x[1]), argmax(ls_grades)
		return xn[0], xn[1], Xnb[0], Xnb[1], k

	def LsRun(self, k, x, x_f, xb, xb_f, improve, SR, g, task):
		XBn = list()
		for _j in range(self.NoLs):
			x, x_f, xnb, xnb_f, improve, grade, SR = self.LSs[k](x, x_f, xb, xb_f, improve, SR, task, BONUS1=self.BONUS1, BONUS2=self.BONUS2, rnd=self.Rand)
			g += grade
			XBn.append((xnb, xnb_f))
		xb, xb_f = min(XBn, key=lambda x: x[1])
		return x, x_f, xb, xb_f, improve, SR, g

	def getBest(self, X, X_f):
		ib = argmin(X_f)
		return X[ib], X_f[ib]

	def runTask(self, task):
		# TODO izgradi simulirano ortogonalno matriko
		SOA = self.rand([self.M, task.D])
		X = task.Lower + task.bRange * SOA / (self.M - 1)
		X_f = apply_along_axis(task.eval, 1, X)
		enable, improve, SR, grades = full(self.M, True), full(self.M, True), full([self.M, task.D], task.bRange / 2), full(self.M, 0.0)
		xb, xb_f = self.getBest(X, X_f)
		while not task.stopCond():
			for i in range(self.M):
				if not enable[i]: continue
				enable[i], grades[i] = False, 0
				X[i], X_f[i], xb, xb_f, k = self.GradingRun(X[i], X_f[i], xb, xb_f, improve[i], SR[i], task)
				X[i], X_f[i], xb, xb_f, improve[i], SR[i], grades[i] = self.LsRun(k, X[i], X_f[i], xb, xb_f, improve[i], SR[i], grades[i], task)
			for _ in range(self.NoLsBest): _, _, xb, xb_f, _, _, _ = MTS_LS1(xb, xb_f, xb, xb_f, False, task.bRange, task, rnd=self.Rand)
			enable[argsort(grades)[:self.NoEnabled]] = True
		return xb, xb_f

class MultipleTrajectorySearchV1(MultipleTrajectorySearch):
	r"""Implementation of Multiple trajectory search.

	**Algorithm:** Multiple trajectory search

	**Date:** 2018

	**Authors:** Klemen Berkovic

	**License:** MIT

	**Reference URL:** https://ieeexplore.ieee.org/document/4983179/

	**Reference paper:** Tseng, Lin-Yu, and Chun Chen. "Multiple trajectory search for unconstrained/constrained multi-objective optimization." Evolutionary Computation, 2009. CEC'09. IEEE Congress on. IEEE, 2009.
	"""
	def __init__(self, **kwargs):
		MultipleTrajectorySearch.__init__(self, name='MultipleTrajectorySearchV1', sName='MTSv1', **kwargs)
		self.LSs = [MTS_LS1v1, MTS_LS2, MTS_LS3v1]

	def runTask(self, task):
		SOA = self.rand([self.M, task.D])
		X = task.Lower + task.bRange * SOA / (self.M - 1)
		X_f = apply_along_axis(task.eval, 1, X)
		enable, improve, SR, grades = full(self.M, True), full(self.M, True), full([self.M, task.D], task.bRange / 2), full(self.M, 0.0)
		xb, xb_f = self.getBest(X, X_f)
		while not task.stopCond():
			for i in range(self.M):
				if not enable[i]: continue
				enable[i], grades[i] = False, 0
				X[i], X_f[i], xb, xb_f, k = self.GradingRun(X[i], X_f[i], xb, xb_f, improve[i], SR[i], task)
				X[i], X_f[i], xb, xb_f, improve[i], SR[i], grades[i] = self.LsRun(k, X[i], X_f[i], xb, xb_f, improve[i], SR[i], grades[i], task)
			enable[argsort(grades)[:self.NoEnabled]] = True
		return xb, xb_f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
