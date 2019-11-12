# encoding=utf8
import logging

from numpy import argmax, log, exp, full

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger("NiaPy.algorithms.basic")
logger.setLevel("INFO")

__all__ = ["HarmonySearch", "HarmonySearchV1"]


class HarmonySearch(Algorithm):
	r"""Implementation of harmony search algorithm.

	Algorithm:
			  Harmony Search Algorithm

	Date:
			  2018

	Authors:
			  Klemen Berkovič

	License:
			  MIT

	Reference URL:
			  https://link.springer.com/chapter/10.1007/978-3-642-00185-7_1

	Reference paper:
			  Yang, Xin-She. "Harmony search as a metaheuristic algorithm." Music-inspired harmony search algorithm. Springer, Berlin, Heidelberg, 2009. 1-14.

	Attributes:
			  Name (List[str]): List of strings representing algorithm names
			  r_accept (float): Probability of accepting new bandwidth into harmony.
			  r_pa (float): Probability of accepting random bandwidth into harmony.
			  b_range (float): Range of bandwidth.

	See Also:
			  * :class:`NiaPy.algorithms.algorithm.Algorithm`

	"""

	Name = ["HarmonySearch", "HS"]

	@staticmethod
	def algorithmInfo():
		r"""Get basic information about the algorithm.

		Returns:
				  str: Basic information.
		"""
		return r"""Yang, Xin-She. "Harmony search as a metaheuristic algorithm." Music-inspired harmony search algorithm. Springer, Berlin, Heidelberg, 2009. 1-14."""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
				  Dict[str, Callable]:
							 * HMS (Callable[[int], bool])
							 * r_accept (Callable[[float], bool])
							 * r_pa (Callable[[float], bool])
							 * b_range (Callable[[float], bool])
		"""
		return {
			"HMS": lambda x: isinstance(x, int) and x > 0,
			"r_accept": lambda x: isinstance(x, float) and 0 < x < 1,
			"r_pa": lambda x: isinstance(x, float) and 0 < x < 1,
			"b_range": lambda x: isinstance(x, (int, float)) and x > 0
		}

	def setParameters(self, HMS=30, r_accept=0.7, r_pa=0.35, b_range=1.42, **ukwargs):
		r"""Set the arguments of the algorithm.

		Arguments:
				  HMS (Optional[int]): Number of harmony in the memory
				  r_accept (Optional[float]): Probability of accepting new bandwidth to harmony.
				  r_pa (Optional[float]): Probability of accepting random bandwidth into harmony.
				  b_range (Optional[float]): Bandwidth range.

		See Also:
				  * :func:`NiaPy.algorithms.algorithm.Algorithm.setParameters`
		"""
		ukwargs.pop('NP', None)
		Algorithm.setParameters(self, NP=HMS, **ukwargs)
		self.r_accept, self.r_pa, self.b_range = r_accept, r_pa, b_range

	def getParameters(self):
		d = Algorithm.getParameters(self)
		d.pop('NP', None)
		d.update({
			'HMS': self.NP,
			'r_accept': self.r_accept,
			'r_pa': self.r_pa,
			'b_range': self.b_range
		})
		return d

	def bw(self, task):
		r"""Get bandwidth.

		Args:
				  task (Task): Optimization task.

		Returns:
				  float: Bandwidth.
		"""
		return self.uniform(-1, 1) * self.b_range

	def adjustment(self, x, task):
		r"""Adjust value based on bandwidth.

		Args:
				  x (Union[int, float]): Current position.
				  task (Task): Optimization task.

		Returns:
				  float: New position.
		"""
		return x + self.bw(task)

	def improvize(self, HM, task):
		r"""Create new individual.

		Args:
				  HM (numpy.ndarray): Current population.
				  task (Task): Optimization task.

		Returns:
				  numpy.ndarray: New individual.
		"""
		H = full(task.D, .0)
		for i in range(task.D):
			r, j = self.rand(), self.randint(self.NP)
			H[i] = HM[j, i] if r > self.r_accept else self.adjustment(HM[j, i], task) if r > self.r_pa else self.uniform(task.Lower[i], task.Upper[i])
		return H

	def initPopulation(self, task):
		r"""Initialize first population.

		Args:
				  task (Task): Optimization task.

		Returns:
				  Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
							 1. New harmony/population.
							 2. New population fitness/function values.
							 3. Additional parameters.

		See Also:
				  * :func:`NiaPy.algorithms.algorithm.Algorithm.initPopulation`
		"""
		return Algorithm.initPopulation(self, task)

	def runIteration(self, task, HM, HM_f, xb, fxb, **dparams):
		r"""Core function of HarmonySearch algorithm.

		Args:
				  task (Task): Optimization task.
				  HM (numpy.ndarray): Current population.
				  HM_f (numpy.ndarray): Current populations function/fitness values.
				  xb (numpy.ndarray): Global best individual.
				  fxb (float): Global best fitness/function value.
				  **dparams (Dict[str, Any]): Additional arguments.

		Returns:
				  Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
							 1. New harmony/population.
							 2. New populations function/fitness values.
							 3. New global best solution
							 4. New global best solution fitness/objective value
							 5. Additional arguments.
		"""
		H = self.improvize(HM, task)
		H_f = task.eval(task.repair(H, self.Rand))
		iw = argmax(HM_f)
		if H_f <= HM_f[iw]: HM[iw], HM_f[iw] = H, H_f
		xb, fxb = self.getBest(H, H_f, xb, fxb)
		return HM, HM_f, xb, fxb, {}

class HarmonySearchV1(HarmonySearch):
	r"""Implementation of harmony search algorithm.

	Algorithm:
			  Harmony Search Algorithm

	Date:
			  2018

	Authors:
			  Klemen Berkovič

	License:
			  MIT

	Reference URL:
			  https://link.springer.com/chapter/10.1007/978-3-642-00185-7_1

	Reference paper:
			  Yang, Xin-She. "Harmony search as a metaheuristic algorithm." Music-inspired harmony search algorithm. Springer, Berlin, Heidelberg, 2009. 1-14.

	Attributes:
			  Name (List[str]): List of strings representing algorithm name.
			  bw_min (float): Minimal bandwidth.
			  bw_max (float): Maximal bandwidth.

	See Also:
			  * :class:`NiaPy.algorithms.basic.hs.HarmonySearch`
	"""
	Name = ["HarmonySearchV1", "HSv1"]

	@staticmethod
	def algorithmInfo():
		r"""Get basic information about algorihtm.

		Returns:
				  str: Basic information.
		"""
		return r"""Yang, Xin-She. "Harmony search as a metaheuristic algorithm." Music-inspired harmony search algorithm. Springer, Berlin, Heidelberg, 2009. 1-14."""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
				  Dict[str, Callable]: Function for testing correctness of parameters.

		See Also:
				  * :func:`NiaPy.algorithms.basic.HarmonySearch.typeParameters`
		"""
		d = HarmonySearch.typeParameters()
		del d["b_range"]
		d.update({
			"dw_min": lambda x: isinstance(x, (float, int)) and x >= 1,
			"dw_max": lambda x: isinstance(x, (float, int)) and x >= 1
		})
		return d

	def setParameters(self, bw_min=1, bw_max=2, **kwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
				  bw_min (Optional[float]): Minimal bandwidth
				  bw_max (Optional[float]): Maximal bandwidth
				  kwargs (Dict[str, Any]): Additional arguments.

		See Also:
				  * :func:`NiaPy.algorithms.basic.hs.HarmonySearch.setParameters`
		"""
		HarmonySearch.setParameters(self, **kwargs)
		self.bw_min, self.bw_max = bw_min, bw_max

	def getParameters(self):
		d = HarmonySearch.getParameters(self)
		d.update({
			'bw_min': self.bw_min,
			'bw_max': self.bw_max
		})
		return d

	def bw(self, task):
		r"""Get new bandwidth.

		Args:
				  task (Task): Optimization task.

		Returns:
				  float: New bandwidth.
		"""
		return self.bw_min * exp(log(self.bw_min / self.bw_max) * task.Iters / task.nGEN)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
