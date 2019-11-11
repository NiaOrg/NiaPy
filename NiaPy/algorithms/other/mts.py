# encoding=utf8
import logging
import operator as oper

from numpy import random as rand, vectorize, argwhere, copy, apply_along_axis, argmin, argsort, fmin, fmax, full, asarray, abs, inf

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['MultipleTrajectorySearch', 'MultipleTrajectorySearchV1', 'MTS_LS1', 'MTS_LS1v1', 'MTS_LS2', 'MTS_LS3', 'MTS_LS3v1']

def MTS_LS1(Xk, Xk_fit, Xb, Xb_fit, improve, SR, task, BONUS1=10, BONUS2=1, sr_fix=0.4, rnd=rand, **ukwargs):
	r"""Multiple trajectory local search one.

	Args:
		Xk (numpy.ndarray): Current solution.
		Xk_fit (float): Current solutions fitness/function value.
		Xb (numpy.ndarray): Global best solution.
		Xb_fit (float): Global best solutions fitness/function value.
		improve (bool): Has the solution been improved.
		SR (numpy.ndarray): Search range.
		task (Task): Optimization task.
		BONUS1 (int): Bonus reward for improving global best solution.
		BONUS2 (int): Bonus reward for improving solution.
		sr_fix (numpy.ndarray): Fix when search range is to small.
		rnd (mtrand.RandomState): Random number generator.
		**ukwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray]:
			1. New solution.
			2. New solutions fitness/function value.
			3. Global best if found else old global best.
			4. Global bests function/fitness value.
			5. If solution has improved.
			6. Search range.
	"""
	if not improve:
		SR /= 2
		ifix = argwhere(SR < 1e-15)
		SR[ifix] = task.bRange[ifix] * sr_fix
	improve, grade = False, 0.0
	for i in range(len(Xk)):
		Xk_i_old = Xk[i]
		Xk[i] = Xk_i_old - SR[i]
		Xk = task.repair(Xk, rnd)
		Xk_fit_new = task.eval(Xk)
		if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk.copy(), Xk_fit_new
		if Xk_fit_new == Xk_fit: Xk[i] = Xk_i_old
		elif Xk_fit_new > Xk_fit:
			Xk[i] = Xk_i_old + 0.5 * SR[i]
			Xk = task.repair(Xk, rnd)
			Xk_fit_new = task.eval(Xk)
			if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk.copy(), Xk_fit_new
			if Xk_fit_new >= Xk_fit: Xk[i] = Xk_i_old
			else: grade, improve, Xk_fit = grade + BONUS2, True, Xk_fit_new
		else: grade, improve, Xk_fit = grade + BONUS2, True, Xk_fit_new
	return Xk, Xk_fit, Xb, Xb_fit, improve, grade, SR

def MTS_LS1v1(Xk, Xk_fit, Xb, Xb_fit, improve, SR, task, BONUS1=10, BONUS2=1, sr_fix=0.4, rnd=rand, **ukwargs):
	r"""Multiple trajectory local search one version two.

	Args:
		Xk (numpy.ndarray): Current solution.
		Xk_fit (float): Current solutions fitness/function value.
		Xb (numpy.ndarray): Global best solution.
		Xb_fit (float): Global best solutions fitness/function value.
		improve (bool): Has the solution been improved.
		SR (numpy.ndarray): Search range.
		task (Task): Optimization task.
		BONUS1 (int): Bonus reward for improving global best solution.
		BONUS2 (int): Bonus reward for improving solution.
		sr_fix (numpy.ndarray): Fix when search range is to small.
		rnd (mtrand.RandomState): Random number generator.
		**ukwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray]:
			1. New solution.
			2. New solutions fitness/function value.
			3. Global best if found else old global best.
			4. Global bests function/fitness value.
			5. If solution has improved.
			6. Search range.
	"""
	if not improve:
		SR /= 2
		ifix = argwhere(SR < 1e-15)
		SR[ifix] = task.bRange[ifix] * sr_fix
	improve, D, grade = False, rnd.uniform(-1, 1, task.D), 0.0
	for i in range(len(Xk)):
		Xk_i_old = Xk[i]
		Xk[i] = Xk_i_old - SR[i] * D[i]
		Xk = task.repair(Xk, rnd)
		Xk_fit_new = task.eval(Xk)
		if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk.copy(), Xk_fit_new
		elif Xk_fit_new == Xk_fit: Xk[i] = Xk_i_old
		elif Xk_fit_new > Xk_fit:
			Xk[i] = Xk_i_old + 0.5 * SR[i]
			Xk = task.repair(Xk, rnd)
			Xk_fit_new = task.eval(Xk)
			if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk.copy(), Xk_fit_new
			elif Xk_fit_new >= Xk_fit: Xk[i] = Xk_i_old
			else: grade, improve, Xk_fit = grade + BONUS2, True, Xk_fit_new
		else: grade, improve, Xk_fit = grade + BONUS2, True, Xk_fit_new
	return Xk, Xk_fit, Xb, Xb_fit, improve, grade, SR

def genNewX(x, r, d, SR, op):
	r"""Move solution to other position based on operator.

	Args:
		x (numpy.ndarray): Solution to move.
		r (int): Random number.
		d (float): Scale factor.
		SR (numpy.ndarray): Search range.
		op (operator): Operator to use.

	Returns:
		numpy.ndarray: Moved solution based on operator.
	"""
	return op(x, SR * d) if r == 0 else x

def MTS_LS2(Xk, Xk_fit, Xb, Xb_fit, improve, SR, task, BONUS1=10, BONUS2=1, sr_fix=0.4, rnd=rand, **ukwargs):
	r"""Multiple trajectory local search two.

	Args:
		Xk (numpy.ndarray): Current solution.
		Xk_fit (float): Current solutions fitness/function value.
		Xb (numpy.ndarray): Global best solution.
		Xb_fit (float): Global best solutions fitness/function value.
		improve (bool): Has the solution been improved.
		SR (numpy.ndarray): Search range.
		task (Task): Optimization task.
		BONUS1 (int): Bonus reward for improving global best solution.
		BONUS2 (int): Bonus reward for improving solution.
		sr_fix (numpy.ndarray): Fix when search range is to small.
		rnd (mtrand.RandomState): Random number generator.
		**ukwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray]:
			1. New solution.
			2. New solutions fitness/function value.
			3. Global best if found else old global best.
			4. Global bests function/fitness value.
			5. If solution has improved.
			6. Search range.

	See Also:
		* :func:`NiaPy.algorithms.other.genNewX`
	"""
	if not improve:
		SR /= 2
		ifix = argwhere(SR < 1e-15)
		SR[ifix] = task.bRange[ifix] * sr_fix
	improve, grade = False, 0.0
	for _ in range(len(Xk)):
		D = -1 + rnd.rand(len(Xk)) * 2
		R = rnd.choice([0, 1, 2, 3], len(Xk))
		Xk_new = task.repair(vectorize(genNewX)(Xk, R, D, SR, oper.sub), rnd)
		Xk_fit_new = task.eval(Xk_new)
		if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk_new.copy(), Xk_fit_new
		elif Xk_fit_new != Xk_fit:
			if Xk_fit_new > Xk_fit:
				Xk_new = task.repair(vectorize(genNewX)(Xk, R, D, SR, oper.add), rnd)
				Xk_fit_new = task.eval(Xk_new)
				if Xk_fit_new < Xb_fit: grade, Xb, Xb_fit = grade + BONUS1, Xk_new.copy(), Xk_fit_new
				elif Xk_fit_new < Xk_fit: grade, Xk, Xk_fit, improve = grade + BONUS2, Xk_new.copy(), Xk_fit_new, True
			else: grade, Xk, Xk_fit, improve = grade + BONUS2, Xk_new.copy(), Xk_fit_new, True
	return Xk, Xk_fit, Xb, Xb_fit, improve, grade, SR

def MTS_LS3(Xk, Xk_fit, Xb, Xb_fit, improve, SR, task, BONUS1=10, BONUS2=1, rnd=rand, **ukwargs):
	r"""Multiple trajectory local search three.

	Args:
		Xk (numpy.ndarray): Current solution.
		Xk_fit (float): Current solutions fitness/function value.
		Xb (numpy.ndarray): Global best solution.
		Xb_fit (float): Global best solutions fitness/function value.
		improve (bool): Has the solution been improved.
		SR (numpy.ndarray): Search range.
		task (Task): Optimization task.
		BONUS1 (int): Bonus reward for improving global best solution.
		BONUS2 (int): Bonus reward for improving solution.
		rnd (mtrand.RandomState): Random number generator.
		**ukwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray]:
			1. New solution.
			2. New solutions fitness/function value.
			3. Global best if found else old global best.
			4. Global bests function/fitness value.
			5. If solution has improved.
			6. Search range.
	"""
	Xk_new, grade = copy(Xk), 0.0
	for i in range(len(Xk)):
		Xk1, Xk2, Xk3 = copy(Xk_new), copy(Xk_new), copy(Xk_new)
		Xk1[i], Xk2[i], Xk3[i] = Xk1[i] + 0.1, Xk2[i] - 0.1, Xk3[i] + 0.2
		Xk1, Xk2, Xk3 = task.repair(Xk1, rnd), task.repair(Xk2, rnd), task.repair(Xk3, rnd)
		Xk1_fit, Xk2_fit, Xk3_fit = task.eval(Xk1), task.eval(Xk2), task.eval(Xk3)
		if Xk1_fit < Xb_fit: grade, Xb, Xb_fit, improve = grade + BONUS1, Xk1.copy(), Xk1_fit, True
		if Xk2_fit < Xb_fit: grade, Xb, Xb_fit, improve = grade + BONUS1, Xk2.copy(), Xk2_fit, True
		if Xk3_fit < Xb_fit: grade, Xb, Xb_fit, improve = grade + BONUS1, Xk3.copy(), Xk3_fit, True
		D1, D2, D3 = Xk_fit - Xk1_fit if abs(Xk1_fit) != inf else 0, Xk_fit - Xk2_fit if abs(Xk2_fit) != inf else 0, Xk_fit - Xk3_fit if abs(Xk3_fit) != inf else 0
		if D1 > 0: grade, improve = grade + BONUS2, True
		if D2 > 0: grade, improve = grade + BONUS2, True
		if D3 > 0: grade, improve = grade + BONUS2, True
		a, b, c = 0.4 + rnd.rand() * 0.1, 0.1 + rnd.rand() * 0.2, rnd.rand()
		Xk_new[i] += a * (D1 - D2) + b * (D3 - 2 * D1) + c
		Xk_new = task.repair(Xk_new, rnd)
		Xk_fit_new = task.eval(Xk_new)
		if Xk_fit_new < Xk_fit:
			if Xk_fit_new < Xb_fit: Xb, Xb_fit, grade = Xk_new.copy(), Xk_fit_new, grade + BONUS1
			else: grade += BONUS2
			Xk, Xk_fit, improve = Xk_new, Xk_fit_new, True
	return Xk, Xk_fit, Xb, Xb_fit, improve, grade, SR

def MTS_LS3v1(Xk, Xk_fit, Xb, Xb_fit, improve, SR, task, phi=3, BONUS1=10, BONUS2=1, rnd=rand, **ukwargs):
	r"""Multiple trajectory local search three version one.

	Args:
		Xk (numpy.ndarray): Current solution.
		Xk_fit (float): Current solutions fitness/function value.
		Xb (numpy.ndarray): Global best solution.
		Xb_fit (float): Global best solutions fitness/function value.
		improve (bool): Has the solution been improved.
		SR (numpy.ndarray): Search range.
		task (Task): Optimization task.
		phi (int): Number of new generated positions.
		BONUS1 (int): Bonus reward for improving global best solution.
		BONUS2 (int): Bonus reward for improving solution.
		rnd (mtrand.RandomState): Random number generator.
		**ukwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray]:
			1. New solution.
			2. New solutions fitness/function value.
			3. Global best if found else old global best.
			4. Global bests function/fitness value.
			5. If solution has improved.
			6. Search range.
	"""
	grade, Disp = 0.0, task.bRange / 10
	while True in (Disp > 1e-3):
		Xn = apply_along_axis(task.repair, 1, asarray([rnd.permutation(Xk) + Disp * rnd.uniform(-1, 1, len(Xk)) for _ in range(phi)]), rnd)
		Xn_f = apply_along_axis(task.eval, 1, Xn)
		iBetter, iBetterBest = argwhere(Xn_f < Xk_fit), argwhere(Xn_f < Xb_fit)
		grade += len(iBetterBest) * BONUS1 + (len(iBetter) - len(iBetterBest)) * BONUS2
		if len(Xn_f[iBetterBest]) > 0:
			ib, improve = argmin(Xn_f[iBetterBest]), True
			Xb, Xb_fit, Xk, Xk_fit = Xn[iBetterBest][ib][0].copy(), Xn_f[iBetterBest][ib][0], Xn[iBetterBest][ib][0].copy(), Xn_f[iBetterBest][ib][0]
		elif len(Xn_f[iBetter]) > 0:
			ib, improve = argmin(Xn_f[iBetter]), True
			Xk, Xk_fit = Xn[iBetter][ib][0].copy(), Xn_f[iBetter][ib][0]
		Su, Sl = fmin(task.Upper, Xk + 2 * Disp), fmax(task.Lower, Xk - 2 * Disp)
		Disp = (Su - Sl) / 10
	return Xk, Xk_fit, Xb, Xb_fit, improve, grade, SR

class MultipleTrajectorySearch(Algorithm):
	r"""Implementation of Multiple trajectory search.

	Algorithm:
		Multiple trajectory search

	Date:
		2018

	Authors:
		Klemen Berkovic

	License:
		MIT

	Reference URL:
		https://ieeexplore.ieee.org/document/4631210/

	Reference paper:
		Lin-Yu Tseng and Chun Chen, "Multiple trajectory search for Large Scale Global Optimization," 2008 IEEE Congress on Evolutionary Computation (IEEE World Congress on Computational Intelligence), Hong Kong, 2008, pp. 3052-3059. doi: 10.1109/CEC.2008.4631210

	Attributes:
		Name (List[Str]): List of strings representing algorithm name.
		LSs (Iterable[Callable[[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray, Task, Dict[str, Any]], Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, int, numpy.ndarray]]]): Local searches to use.
		BONUS1 (int): Bonus for improving global best solution.
		BONUS2 (int): Bonus for improving solution.
		NoLsTests (int): Number of test runs on local search algorithms.
		NoLs (int): Number of local search algorithm runs.
		NoLsBest (int): Number of locals search algorithm runs on best solution.
		NoEnabled (int): Number of best solution for testing.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['MultipleTrajectorySearch', 'MTS']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Lin-Yu Tseng and Chun Chen, "Multiple trajectory search for Large Scale Global Optimization," 2008 IEEE Congress on Evolutionary Computation (IEEE World Congress on Computational Intelligence), Hong Kong, 2008, pp. 3052-3059. doi: 10.1109/CEC.2008.4631210"""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* M (Callable[[int], bool])
				* NoLsTests (Callable[[int], bool])
				* NoLs (Callable[[int], bool])
				* NoLsBest (Callable[[int], bool])
				* NoEnabled (Callable[[int], bool])
				* BONUS1 (Callable([[Union[int, float], bool])
				* BONUS2 (Callable([[Union[int, float], bool])
		"""
		return {
			'M': lambda x: isinstance(x, int) and x > 0,
			'NoLsTests': lambda x: isinstance(x, int) and x >= 0,
			'NoLs': lambda x: isinstance(x, int) and x >= 0,
			'NoLsBest': lambda x: isinstance(x, int) and x >= 0,
			'NoEnabled': lambda x: isinstance(x, int) and x > 0,
			'BONUS1': lambda x: isinstance(x, (int, float)) and x > 0,
			'BONUS2': lambda x: isinstance(x, (int, float)) and x > 0,
		}

	def setParameters(self, M=40, NoLsTests=5, NoLs=5, NoLsBest=5, NoEnabled=17, BONUS1=10, BONUS2=1, LSs=(MTS_LS1, MTS_LS2, MTS_LS3), **ukwargs):
		r"""Set the arguments of the algorithm.

		Arguments:
			M (int): Number of individuals in population.
			NoLsTests (int): Number of test runs on local search algorithms.
			NoLs (int): Number of local search algorithm runs.
			NoLsBest (int): Number of locals search algorithm runs on best solution.
			NoEnabled (int): Number of best solution for testing.
			BONUS1 (int): Bonus for improving global best solution.
			BONUS2 (int): Bonus for improving self.
			LSs (Iterable[Callable[[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray, Task, Dict[str, Any]], Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, int, numpy.ndarray]]]): Local searches to use.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=ukwargs.pop('NP', M), **ukwargs)
		self.NoLsTests, self.NoLs, self.NoLsBest, self.NoEnabled, self.BONUS1, self.BONUS2 = NoLsTests, NoLs, NoLsBest, NoEnabled, BONUS1, BONUS2
		self.LSs = LSs

	def getParameters(self):
		r"""Get parameters values for the algorithm.

		Returns:
			Dict[str, Any]:
		"""
		d = Algorithm.getParameters(self)
		d.update({
			'M': d.pop('NP', self.NP),
			'NoLsTests': self.NoLsTests,
			'NoLs': self.NoLs,
			'NoLsBest': self.NoLsBest,
			'BONUS1': self.BONUS1,
			'BONUS2': self.BONUS2,
			'NoEnabled': self.NoEnabled,
			'LSs': self.LSs
		})
		return d

	def GradingRun(self, x, x_f, xb, fxb, improve, SR, task):
		r"""Run local search for getting scores of local searches.

		Args:
			x (numpy.ndarray): Solution for grading.
			x_f (float): Solutions fitness/function value.
			xb (numpy.ndarray): Global best solution.
			fxb (float): Global best solutions function/fitness value.
			improve (bool): Info if solution has improved.
			SR (numpy.ndarray): Search range.
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, float, numpy.ndarray, float]:
				1. New solution.
				2. New solutions function/fitness value.
				3. Global best solution.
				4. Global best solutions fitness/function value.
		"""
		ls_grades, Xn = full(3, 0.0), [[x, x_f]] * len(self.LSs)
		for k in range(len(self.LSs)):
			for _ in range(self.NoLsTests):
				Xn[k][0], Xn[k][1], xb, fxb, improve, g, SR = self.LSs[k](Xn[k][0], Xn[k][1], xb, fxb, improve, SR, task, BONUS1=self.BONUS1, BONUS2=self.BONUS2, rnd=self.Rand)
				ls_grades[k] += g
		xn, xn_f = min(Xn, key=lambda x: x[1])
		return xn, xn_f, xb, fxb, k

	def LsRun(self, k, x, x_f, xb, fxb, improve, SR, g, task):
		r"""Run a selected local search.

		Args:
			k (int): Index of local search.
			x (numpy.ndarray): Current solution.
			x_f (float): Current solutions function/fitness value.
			xb (numpy.ndarray): Global best solution.
			fxb (float): Global best solutions fitness/function value.
			improve (bool): If the solution has improved.
			SR (numpy.ndarray): Search range.
			g (int): Grade.
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray, int]:
				1. New best solution found.
				2. New best solutions found function/fitness value.
				3. Global best solution.
				4. Global best solutions function/fitness value.
				5. If the solution has improved.
				6. Grade of local search run.
		"""
		for _j in range(self.NoLs):
			x, x_f, xb, fxb, improve, grade, SR = self.LSs[k](x, x_f, xb, fxb, improve, SR, task, BONUS1=self.BONUS1, BONUS2=self.BONUS2, rnd=self.Rand)
			g += grade
		return x, x_f, xb, fxb, improve, SR, g

	def initPopulation(self, task):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
				1. Initialized population.
				2. Initialized populations function/fitness value.
				3. Additional arguments:
					* enable (numpy.ndarray): If solution/individual is enabled.
					* improve (numpy.ndarray): If solution/individual is improved.
					* SR (numpy.ndarray): Search range.
					* grades (numpy.ndarray): Grade of solution/individual.
		"""
		X, X_f, d = Algorithm.initPopulation(self, task)
		enable, improve, SR, grades = full(self.NP, True), full(self.NP, True), full([self.NP, task.D], task.bRange / 2), full(self.NP, 0.0)
		d.update({
			'enable': enable,
			'improve': improve,
			'SR': SR,
			'grades': grades
		})
		return X, X_f, d

	def runIteration(self, task, X, X_f, xb, xb_f, enable, improve, SR, grades, **dparams):
		r"""Core function of MultipleTrajectorySearch algorithm.

		Args:
			task (Task): Optimization task.
			X (numpy.ndarray): Current population of individuals.
			X_f (numpy.ndarray): Current individuals function/fitness values.
			xb (numpy.ndarray): Global best individual.
			xb_f (float): Global best individual function/fitness value.
			enable (numpy.ndarray): Enabled status of individuals.
			improve (numpy.ndarray): Improved status of individuals.
			SR (numpy.ndarray): Search ranges of individuals.
			grades (numpy.ndarray): Grades of individuals.
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. Initialized population.
				2. Initialized populations function/fitness value.
				3. New global best solution.
				4. New global best solutions fitness/objective value.
				5. Additional arguments:
					* enable (numpy.ndarray): If solution/individual is enabled.
					* improve (numpy.ndarray): If solution/individual is improved.
					* SR (numpy.ndarray): Search range.
					* grades (numpy.ndarray): Grade of solution/individual.
		"""
		for i in range(len(X)):
			if not enable[i]: continue
			enable[i], grades[i] = False, 0
			X[i], X_f[i], xb, xb_f, k = self.GradingRun(X[i], X_f[i], xb, xb_f, improve[i], SR[i], task)
			X[i], X_f[i], xb, xb_f, improve[i], SR[i], grades[i] = self.LsRun(k, X[i], X_f[i], xb, xb_f, improve[i], SR[i], grades[i], task)
		for _ in range(self.NoLsBest): _, _, xb, xb_f, _, _, _ = MTS_LS1(xb, xb_f, xb, xb_f, False, task.bRange.copy() / 10, task, rnd=self.Rand)
		enable[argsort(grades)[:self.NoEnabled]] = True
		return X, X_f, xb, xb_f, {'enable': enable, 'improve': improve, 'SR': SR, 'grades': grades}

class MultipleTrajectorySearchV1(MultipleTrajectorySearch):
	r"""Implementation of Multiple trajectory search.

	Algorithm:
		Multiple trajectory search

	Date:
		2018

	Authors:
		Klemen Berkovic

	License:
		MIT

	Reference URL:
		https://ieeexplore.ieee.org/document/4983179/

	Reference paper:
		Tseng, Lin-Yu, and Chun Chen. "Multiple trajectory search for unconstrained/constrained multi-objective optimization." Evolutionary Computation, 2009. CEC'09. IEEE Congress on. IEEE, 2009.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.other.MultipleTrajectorySearch``
	"""
	Name = ['MultipleTrajectorySearchV1', 'MTSv1']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Tseng, Lin-Yu, and Chun Chen. "Multiple trajectory search for unconstrained/constrained multi-objective optimization." Evolutionary Computation, 2009. CEC'09. IEEE Congress on. IEEE, 2009."""

	def setParameters(self, **kwargs):
		r"""Set core parameters of MultipleTrajectorySearchV1 algorithm.

		Args:
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.other.MultipleTrajectorySearch.setParameters`
		"""
		kwargs.pop('NoLsBest', None)
		MultipleTrajectorySearch.setParameters(self, NoLsBest=0, LSs=(MTS_LS1v1, MTS_LS2), **kwargs)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
