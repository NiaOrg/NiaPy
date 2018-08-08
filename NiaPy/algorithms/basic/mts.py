# encoding=utf8
from numpy import random as rand, vectorize
from NiaPy.algorithms.algoritm import Algorithm
from NiaPy.benchmarks.utility import Task

__all__ = ['MultipleTrajectorySearch', 'MTS_LS1', 'MTS_LS1v1', 'MTS_LS2', 'MTS_LS3']

def MTS_LS1(Xk, Xk_fit, Xb_fit, improve_k, SR, task, BONUS1=10, BONUS2=1, rnd=rand):
	Xb = None
	if not improve_k:
		SR /= 2
		if SR < 1e-15: SR = task.bRange * 0.4
	improve_k = False
	for i in range(len(X_k)):
		Xk_i_old = Xk[i]
		Xk[i] = Xk_i_old - SR
		Xk_fit_new = task.eval(Xk)
		if Xk_fit_new < Xb_fit: grade_k, Xb, Xb_fit = grade_k + BONUS1, Xk_new, Xk_fit_new
		if Xk_fit_new == Xk_fit: Xk[i] = Xk_i_old
		else:
			if Xk_fit_new > Xk_fit:
				Xk[i] = Xk_i_old + 0.5 * SR
				Xk_fit_new = task.eval(X_k)
				if Xk_fit_new < Xb_fit: grade_k, Xb, Xb_fit = grade_k + BONUS1, Xk_new, Xk_fit_new
				if Xk_fit_new >= Xk_fit: Xk[i] = Xk_i_old
				else: grade_k, improve_k, Xk_fit = grade_k + BONUS2, True, Xk_fit_new
			else: grade_k, improve_k, Xk_fit = grade_k + BONUS2, True, Xk_fit_new
	return Xk, Xk_fit, Xb, Xb_fit, improve_k, grade_k, SR

def MTS_LS1v1(Xk, Xk_fit, Xb_fit, improve_k, SR, task, BONUS1=10, BONUS2=1, rnd=rand):
	X_b = None
	if not improve_k:
		SR /= 2
		if SR < 1e-8: SR = task.bRange * 0.4
	improve_k = False
	D = rnd.uniform(-1, 1, len(X_k))
	# TODO
	return Xk, Xk_fit, Xb, Xb_fit, improve_k, grade_k, SR

def genNewX(x, r, d, SR, op): return op(x, SR * d) if r == 0 else x

def MTS_LS2(Xk, Xk_fit, Xb_fit, improve_k, SR, task, BONUS1=10, BONUS2=1, rnd=rand):
	Xb = None
	if not improve_k:
		SR /= 2
		if SR < 1e-15: SR = (task.Lower + task.bRange) * 0.4
	improve_k = False
	for i in range(len(Xk)):
		D, R = -1 + rnd.rand(len(Xk)) * 2, rnd.choice([0, 1, 2, 3], len(Xk))
		Xk_new = vectorize(genNewX)(X_k, R, D, SR, oper.sub)
		Xk_fit_new = task.eval(X_k_new)
		if Xk_fit_new < Xb_fit: grade_k, Xb, Xb_fit = grade_k + BONUS1, Xk_new, Xk_fit_new
		if Xk_fit_new == Xk_fit: continue
		else:
			if Xk_fit_new > Xk_fit:
				Xk_new = vectorize(genNewX)(Xk, R, D, SR, oper.add)
				Xk_fit_new = task.eval(Xk_new)
				if Xk_fit_new < Xb_fit: grade_k, Xb, Xb_fit = grade_k + BONUS1, Xk_new, Xk_fit_new
				if Xk_fit_new >= Xk_fit: continue
				else: grade_k, improve_k, Xk_fit = grade_k + BONUS2, True, Xk_fit_new
			else: grade_k, improve_k = grade_k + BONUS2, True
	return Xk, Xk_fit, Xb, Xb_fit, improve_k, grade_k, SR

def MTS_LS3(Xk, Xk_fit, Xb_fit, improve_k, SR, task, BONUS1=10, BONUS2=1, rnd=rand):
	Xk_new = np.copy(X_k)
	for i in range(len(X_k)):
		Xk_1, Xk_2, X_k_3 = np.copy(X_k_new), np.copy(X_k_new), np.copy(X_k_new)
		Xk1[i], Xk2[i], Xk_3[i] = Xk_1[i] + 0.1, Xk_2[i] - 0.1, Xk_3[i] + 0.2
		Xk1_fit, Xk_2_fit, Xk_3_fit = task.eval(X_k_1), task.eval(X_k_2), task.eval(X_k_3)
		if Xk_1_fit < Xb_fit: grade_k, Xk, Xk_fit = grade_k + BONUS1, Xk_1, Xk_1_fit
		if Xk_2_fit < Xb_fit: grade_k, Xk, Xk_fit = grade_k + BONUS1, Xk_2, Xk_2_fit
		if Xk_3_fit < Xb_fit: grade_k, Xk, Xk_fit = grade_k + BONUS1, Xk_3, Xk_3_fit
		D_1, D_2, D_3 = Xk_fit - Xk_1_fit, Xk_fit - Xk_2_fit, Xk_fit - Xk_3_fit
		if D_1 > 0: grade_k += BONUS2
		if D_2 > 0: grade_k += BONUS2
		if D_3 > 0: grade_k += BONUS2
		a, b, c = 0.4 + rnd.rand() * 0.1, 0.1 + rnd.rand() * 0.2, rnd.rand()
		Xk_new[i] += a * (D_1 - D_2) + b * (D_3 - 2 * D_1) + c
		Xk_fit_new = task.evel(X_k_new)
	if Xk_fit_new < Xk_fit: Xk, Xk_fit = Xk_new, Xk_fit_new
	return Xk, Xk_fit, Xb, Xb_fit, improve_k, grade_k, SR

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
	def __init__(self, **kwargs): Algorithm.__init__(self, name='MultipleTrajectorySearch', sName='MTS', **kwargs)

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
		self.LSs = [MTS_LS1, MTS_LS2, MTS_LS3]

	def runTask(self, task): 
		SOA = self.rand([self.M, self.N])
		X = taks.Lower + task.bRange * SOA / (self.M - 1)
		X_fit = np.apply_along_axis(task.eval, 1, X)
		enable, improve, SearchRanges, grades = np.full(self.M, True), np.full(self.M, True), np.full(self.M, (task.Upper - task.Lower) / 2), np.zeros(self.M)
		ix_b = np.argmin(X_fit)
		x_b, x_b_fit = X[ix_b], X_fit[ix_b]
		while not task.stopCond():
			for i in range(self.M):
				if not enable[i]: continue
				ls_grades, grades[i] = [0, 0, 0], 0
				for j in range(self.NoLsTests): 
					# TODO fix variables for return value, and set parameters for ls
					for k in range(3): _, ls_grades[k], _ += self.LSs[k]()
				ils_grades_best = np.argmax(ls_grades)
				for j in range(self.NoLs):
					# TODO fix variables for return value, and set parameters for ls
					_, grade, _ = self.LSs[ils_grades_best]()
					grades[i] += grade
				enable[i] = False
			for i in range(self.NoLsBest): _ = LocalSearchOne(X_b, X_b_fit)
			# TODO get NoForeground best solutions and set value of use to true
			igrades = np.argpartition(grades, self.NoForeground)
		return x_b, x_b_fit

	def run(self): return self.runTask(self.task)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
