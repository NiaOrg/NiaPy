# encoding=utf8
from numpy import random as rand, vectorize, where, copy, apply_along_axis, argmin, argmax, argsort, fmin, fmax, full
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.benchmarks.utility import Task

__all__ = ['MultipleTrajectorySearch', 'MTS_LS1', 'MTS_LS1v1', 'MTS_LS2', 'MTS_LS3', 'MTS_LS3v1']

def MTS_LS1(Xk, Xk_fit, Xb_fit, improve, SR, task, BONUS1=10, BONUS2=1, rnd=rand):
	Xb, grade = None, 0.0
	if not improve:
		SR /= 2
		ifix = where(SR < 1e-15)
		SR[ifix] = task.bRange[ifix] * 0.4
	improve = False
	for i in range(len(X_k)):
		Xk_i_old = Xk[i]
		Xk[i] = Xk_i_old - SR[i]
		Xk_fit_new = task.eval(Xk)
		if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk_new, Xk_fit_new
		if Xk_fit_new == Xk_fit: Xk[i] = Xk_i_old
		else:
			if Xk_fit_new > Xk_fit:
				Xk[i] = Xk_i_old + 0.5 * SR[i]
				Xk_fit_new = task.eval(X_k)
				if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk_new, Xk_fit_new
				if Xk_fit_new >= Xk_fit: Xk[i] = Xk_i_old
				else: grade, improve_k, Xk_fit = grade + BONUS2, True, Xk_fit_new
			else: grade, improve_k, Xk_fit = grade + BONUS2, True, Xk_fit_new
	return Xb if Xb != None else Xk, Xb_fit if Xb != None else Xk_fit, improve, grade, SR

def MTS_LS1v1(Xk, Xk_fit, Xb_fit, improve, SR, task, BONUS1=10, BONUS2=1, rnd=rand):
	Xb, grade = None, 0.0
	if not improve:
		SR /= 2
		ifix = where(SR < 1e-15)
		SR[ifix] = task.bRange[ifix] * 0.4
	improve, D = False, rnd.uniform(-1, 1, task.D)
	for i in range(len(X_k)):
		Xk_i_old = Xk[i]
		Xk[i] = Xk_i_old - SR[i] * D[i]
		Xk_fit_new = task.eval(Xk)
		if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk_new, Xk_fit_new
		if Xk_fit_new == Xk_fit: Xk[i] = Xk_i_old
		else:
			if Xk_fit_new > Xk_fit:
				Xk[i] = Xk_i_old + 0.5 * SR[i]
				Xk_fit_new = task.eval(X_k)
				if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk_new, Xk_fit_new
				if Xk_fit_new >= Xk_fit: Xk[i] = Xk_i_old
				else: grade, improve_k, Xk_fit = grade + BONUS2, True, Xk_fit_new
			else: grade, improve_k, Xk_fit = grade + BONUS2, True, Xk_fit_new
	return Xb, Xb_fit, improve, grade, SR

def genNewX(x, r, d, SR, op): return op(x, SR * d) if r == 0 else x

def MTS_LS2(Xk, Xk_fit, Xb_fit, improve, SR, task, BONUS1=10, BONUS2=1, rnd=rand):
	Xb, grade = None, 0.0
	if not improve:
		SR /= 2
		ifix = where(SR < 1e-15)
		SR[ifix] = task.bRange[ifix] * 0.4
	improve = False
	for i in range(len(Xk)):
		D, R = -1 + rnd.rand(len(Xk)) * 2, rnd.choice([0, 1, 2, 3], len(Xk))
		Xk_new = vectorize(genNewX)(Xk, R, D, SR, oper.sub)
		Xk_fit_new = task.eval(Xk_new)
		if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk_new, Xk_fit_new
		if Xk_fit_new != Xk_fit:
			if Xk_fit_new > Xk_fit:
				Xk_new = vectorize(genNewX)(Xk, R, D, SR, oper.add)
				Xk_fit_new = task.eval(Xk_new)
				if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk_new, Xk_fit_new
				if Xk_fit_new < Xk_fit: grade, improve, Xk_fit, Xk, Xk_fit = grade + BONUS2, True, Xk_new, Xk_fit_new
			else: grade, improve, Xk, Xk_fit = grade + BONUS2, True, Xk_new, Xk_fit_new
	return Xb, Xb_fit, improve, grade, SR

def MTS_LS3(Xk, Xk_fit, Xb_fit, improve, SR, task, BONUS1=10, BONUS2=1, rnd=rand):
	Xb, Xk_new, grade = None, copy(X_k), 0.0
	for i in range(len(Xk)):
		Xk_1, Xk_2, Xk_3 = copy(Xk_new), copy(Xk_new), copy(Xk_new)
		Xk1[i], Xk2[i], Xk_3[i] = Xk_1[i] + 0.1, Xk_2[i] - 0.1, Xk_3[i] + 0.2
		Xk1_fit, Xk_2_fit, Xk_3_fit = task.eval(Xk_1), task.eval(Xk_2), task.eval(Xk_3)
		if Xk_1_fit < Xb_fit: grade, Xb, Xb_fit, improve = grade + BONUS1, Xk_1, Xk_1_fit, True
		if Xk_2_fit < Xb_fit: grade, Xb, Xb_fit, improve = grade + BONUS1, Xk_2, Xk_2_fit, True
		if Xk_3_fit < Xb_fit: grade, Xb, Xb_fit, improve = grade + BONUS1, Xk_3, Xk_3_fit, True
		D1, D2, D3 = Xk_fit - Xk_1_fit, Xk_fit - Xk_2_fit, Xk_fit - Xk_3_fit
		if D1 > 0: grade, improve = grade + BONUS2, True
		if D2 > 0: grade, improve = grade + BONUS2, True
		if D3 > 0: grade, improve = grade + BONUS2, True
		a, b, c = 0.4 + rnd.rand() * 0.1, 0.1 + rnd.rand() * 0.2, rnd.rand()
		Xk_new[i] += a * (D1 - D2) + b * (D3 - 2 * D1) + c
		Xk_fit_new = task.evel(Xk_new)
	if Xk_fit_new < Xk_fit: Xk, Xk_fit = Xk_new, Xk_fit_new
	return Xb, Xb_fit, improve, grade, SR

def MTS_LS3v1(Xk, Xk_fit, Xb_fit, improve, SR, task, l=3, BONUS1=10, BONUS2=1, rnd=rand):
	Xb, grade, Disp = None, 0.0, task.bRange / 10
	while True in (Disp > 1e-3):
		Xn = [rnd.permutation(Xk) + Disp * rnd.uniform(-1, 1, len(Xk)) for i in l]
		Xn_f = apply_along_axis(task.eval, 1, Xn)
		iBetter, iBetterBest = where(Xn_f < Xk_fit), where(Xn_f < Xb_fit)
		grade += len(iBetterBest) * BONUS1 + (len(iBetter) - len(iBetterBest)) * BONUS2
		if len(iBetterBest) > 0:
			ib, improve = argmin(Xn_f[iBetterBest]), True
			Xb, Xb_fit, Xk, Xk_fit = Xn[ib], Xn_f[ib], Xn[ib], Xn_f[ib]
		elif len(iBetter) > 0:
			ib, improve = argmin(Xn_f[iBetter]), True
			Xk, Xk_fit = Xn[ib], Xn_f[ib]
		Su, Sl = fmin(task.Upper, Xk + 2 * Disp), fmax(task.Lower, Xk - 2 * Disp)
		Disp = (Su - Sl) / 10
	return Xb, Xb_fit, improve, grade, SR

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
		if kwargs.get('name', None) == None: Algorithm.__init__(self, name='MultipleTrajectorySearch', sName='MTS', **kwargs)
		else: Algorithm.__init__(self, **kwargs)
		self.LSs = [MTS_LS1, MTS_LS2, MTS_LS3]

	def setParameters(self, NP, NoLsTests, NoLs, NoLsBest, NoEnabled, **ukwargs):
		r"""Set the arguments of the algorithm.

		Arguments:
		NP, M {integer} -- population size
		NoLsTests {integer} -- number of test runs on local search algorihms
		NoLs {integer} -- number of local search algoritm runs
		NoLsBest {integer} -- number of locals search algorithm runs on best solution
		NoEnabled {integer} -- number of best solution for testing
		"""
		self.M, self.NoLsTests, self.NoLs, self.NoForeground = NP, NoLsTests, NoLs, NoForeground

	def GradingRun(self, x, x_f, improve, SR, task):
		ls_grades, grades[i] = full(3, 0.0), 0.0
		for j in range(self.NoLsTests):
			Xn, Xn_f = list(), list()
			for k in range(len(self.LSs)):
				xn, xn_f, _, ls_grades[k], _ = self.LSs[k](x, x_f, xb_f, improve, SR, task, BONUS1=self.BONUS1, BONUS2=self.BONUS2, rnd=self.Rand)
				Xn.append(xn), Xn_f.append(xn_f)
			ib = argmin(Xn_f)
		return x, x_f, k

	def runTask(self, task):
		SOA = self.rand([self.M, self.N])
		X = taks.Lower + task.bRange * SOA / (self.M - 1)
		X_f = np.apply_along_axis(task.eval, 1, X)
		enable, improve, SearchRanges, grades = full(self.M, True), full(self.M, True), full(self.M, task.bRange / 2), full(self.M, 0.0)
		ix_b = np.argmin(X_fit)
		xb, xb_f = X[ix_b], X_f[ix_b]
		while not task.stopCond():
			for i in range(self.M):
				if not enable[i]: continue
				X[i], X_f[i], k = self.GradingRun(X[i], X_f[i], improve[i], SR[i], task)
				for j in range(self.NoLs):
					X[i], X_f[i], improve[i], grade, SR[i] = self.LSs[k](X[i], X_f[i], xb_f, improve[i], SR[i], task, BONUS1=self.BONUS1, BONUS2=self.BONUS2, rnd=self.Rand)
					grades[i] += grade
				enable[i] = False
			for i in range(self.NoLsBest): xb, xb_f, _, _, _ = MTS_LS1(xb, xb_f, xb_f, improve[], SR[], task, rnd=self.Rand)
			isort = argsort(grades, self.NoForeground)
			enable[isort[:self.NoForeground]] = True
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
		SOA = self.rand([self.M, self.N])
		X = taks.Lower + task.bRange * SOA / (self.M - 1)
		X_fit = apply_along_axis(task.eval, 1, X)
		enable, improve, SearchRanges, grades = full(self.M, True), full(self.M, True), full(self.M, task.bRange / 2), full(self.M, 0.0)
		ix_b = argmin(X_fit)
		xb, xb_fit = X[ix_b], X_fit[ix_b]
		while not task.stopCond():
			for i in range(self.M):
				if not enable[i]: continue
				ls_grades, grades[i] = full(3, 0.0), 0.0
				for j in range(self.NoLsTests):
					# TODO fix variables for return value, and set parameters for ls
					# for k in range(3): _, ls_grades[k], _ += self.LSs[k](X[i], X_fit[i], xb_fit, improve[i], SR[i], task, BONUS1=self.BONUS1, BONUS2=self.BONUS2, rnd=self.Rand)
				# ils_grades_best = argmax(ls_grades)
				# for j in range(self.NoLs):
					# TODO fix variables for return value, and set parameters for ls
					_, grade, _ = self.LSs[ils_grades_best]()
					grades[i] += grade
				enable[i] = False
			isort = argsort(grades, self.NoForeground)
			enable[isort[:self.NoForeground]] = True
		return xb, xb_fit


# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
