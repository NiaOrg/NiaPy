# encoding=utf8
import logging

from numpy import apply_along_axis, asarray, argmin, argmax, sum, full

from NiaPy.algorithms.algorithm import Algorithm

__all__ = ['GravitationalSearchAlgorithm']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

class GravitationalSearchAlgorithm(Algorithm):
	r"""Implementation of gravitational search algorithm.

	Algorithm:
		Gravitational Search Algorithm

	Date:
		2018

	Author:
		Klemen Berkoivč

	License:
		MIT

	Reference URL:
		https://doi.org/10.1016/j.ins.2009.03.004

	Reference paper:
		Esmat Rashedi, Hossein Nezamabadi-pour, Saeid Saryazdi, GSA: A Gravitational Search Algorithm, Information Sciences, Volume 179, Issue 13, 2009, Pages 2232-2248, ISSN 0020-0255

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.algorithm.Algorithm`
	"""
	Name = ['GravitationalSearchAlgorithm', 'GSA']

	@staticmethod
	def algorithmInfo():
		r"""Get algorithm information.

		Returns:
			str: Algorithm information.
		"""
		return r"""Esmat Rashedi, Hossein Nezamabadi-pour, Saeid Saryazdi, GSA: A Gravitational Search Algorithm, Information Sciences, Volume 179, Issue 13, 2009, Pages 2232-2248, ISSN 0020-0255"""

	@staticmethod
	def typeParameters():
		r"""TODO.

		Returns:
			Dict[str, Callable]:
				* G_0 (Callable[[Union[int, float]], bool]): TODO
				* epsilon (Callable[[float], bool]): TODO

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'G_0': lambda x: isinstance(x, (int, float)) and x >= 0,
			'epsilon': lambda x: isinstance(x, float) and 0 < x < 1
		})
		return d

	def setParameters(self, NP=40, G_0=2.467, epsilon=1e-17, **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
			G_0 (float): Starting gravitational constant.
			epsilon (float): TODO.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.G_0, self.epsilon = G_0, epsilon

	def getParameters(self):
		r"""Get algorithm parameters values.

		Returns:
			Dict[str, Any]: TODO.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.getParameters`
		"""
		d = Algorithm.getParameters(self)
		d.update({
			'G_0': self.G_0,
			'epsilon': self.epsilon
		})
		return d

	def G(self, t):
		r"""TODO.

		Args:
			t (int): TODO

		Returns:
			float: TODO
		"""
		return self.G_0 / t

	def d(self, x, y, ln=2):
		r"""TODO.

		Args:
			x:
			y:
			ln:

		Returns:
			TODO
		"""
		return sum((x - y) ** ln) ** (1 / ln)

	def initPopulation(self, task):
		r"""Initialize staring population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized population.
				2. Initialized populations fitness/function values.
				3. Additional arguments:
					* v (numpy.ndarray[float]): TODO

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.initPopulation`
		"""
		X, X_f, _ = Algorithm.initPopulation(self, task)
		v = full([self.NP, task.D], 0.0)
		return X, X_f, {'v': v}

	def runIteration(self, task, X, X_f, xb, fxb, v, **dparams):
		r"""Core function of GravitationalSearchAlgorithm algorithm.

		Args:
			task (Task): Optimization task.
			X (numpy.ndarray): Current population.
			X_f (numpy.ndarray): Current populations fitness/function values.
			xb (numpy.ndarray): Global best solution.
			fxb (float): Global best fitness/function value.
			v (numpy.ndarray): TODO
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population.
				2. New populations fitness/function values.
				3. New global best solution
				4. New global best solutions fitness/objective value
				5. Additional arguments:
					* v (numpy.ndarray): TODO
		"""
		ib, iw = argmin(X_f), argmax(X_f)
		m = (X_f - X_f[iw]) / (X_f[ib] - X_f[iw])
		M = m / sum(m)
		Fi = asarray([[self.G(task.Iters) * ((M[i] * M[j]) / (self.d(X[i], X[j]) + self.epsilon)) * (X[j] - X[i]) for j in range(len(M))] for i in range(len(M))])
		F = sum(self.rand([self.NP, task.D]) * Fi, axis=1)
		a = F.T / (M + self.epsilon)
		v = self.rand([self.NP, task.D]) * v + a.T
		X = apply_along_axis(task.repair, 1, X + v, self.Rand)
		X_f = apply_along_axis(task.eval, 1, X)
		xb, fxb = self.getBest(X, X_f, xb, fxb)
		return X, X_f, xb, fxb, {'v': v}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
