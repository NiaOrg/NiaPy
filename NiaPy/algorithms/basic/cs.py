# encoding=utf8
import logging
from numpy import apply_along_axis, argsort
from scipy.stats import levy
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CuckooSearch']

class CuckooSearch(Algorithm):
	r"""Implementation of Cuckoo behaviour and levy flights.

	Algorithm:
		Cuckoo Search

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference:
		Yang, Xin-She, and Suash Deb. "Cuckoo search via Lévy flights." Nature & Biologically Inspired Computing, 2009. NaBIC 2009. World Congress on. IEEE, 2009.

	Attributes:
		Name (List[str]): list of strings representing algorithm names.
		N (int): Population size.
		pa (float): Proportion of worst nests.
		alpha (float): Scale factor for levy flight.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['CuckooSearch', 'CS']

	@staticmethod
	def algorithmInfo():
		r"""Get algorithms information.

		Returns:
			str: Algorithm information.
		"""
		return r"""Yang, Xin-She, and Suash Deb. "Cuckoo search via Lévy flights." Nature & Biologically Inspired Computing, 2009. NaBIC 2009. World Congress on. IEEE, 2009."""

	@staticmethod
	def typeParameters():
		r"""TODO.

		Returns:
			Dict[str, Callable]:
				* N (Callable[[int], bool]): TODO
				* pa (Callable[[float], bool]): TODO
				* alpha (Callable[[Union[int, float]], bool]): TODO
		"""
		return {
			'N': lambda x: isinstance(x, int) and x > 0,
			'pa': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'alpha': lambda x: isinstance(x, (float, int)),
		}

	def setParameters(self, N=50, pa=0.2, alpha=0.5, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			N (int): Population size :math:`\in [1, \infty)`
			pa (float): factor :math:`\in [0, 1]`
			alpah (float): TODO
			**ukwargs (Dict[str, Any]): Additional arguments

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		ukwargs.pop('NP', None)
		Algorithm.setParameters(self, NP=N, **ukwargs)
		self.pa, self.alpha = pa, alpha

	def getParameters(self):
		d = Algorithm.getParameters(self)
		d.pop('NP', None)
		d.update({
			'N': self.NP,
			'pa': self.pa,
			'alpha': self.alpha
		})
		return d

	def emptyNests(self, pop, fpop, pa_v, task):
		r"""Empty ensts.

		Args:
			pop (numpy.ndarray): Current population
			fpop (numpy.ndarray[float]): Current population fitness/funcion values
			pa_v (): TODO.
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float]]:
				1. New population
				2. New population fitness/function values
		"""
		si = argsort(fpop)[:int(pa_v):-1]
		pop[si] = task.Lower + self.rand(task.D) * task.bRange
		fpop[si] = apply_along_axis(task.eval, 1, pop[si])
		return pop, fpop

	def initPopulation(self, task):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized population.
				2. Initialized populations fitness/function values.
				3. Additional arguments:
					* pa_v (float): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		N, N_f, d = Algorithm.initPopulation(self, task)
		d.update({'pa_v': self.NP * self.pa})
		return N, N_f, d

	def runIteration(self, task, pop, fpop, xb, fxb, pa_v, **dparams):
		r"""Core function of CuckooSearch algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Current populations fitness/function values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individual function/fitness values.
			pa_v (float): TODO
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. Initialized population.
				2. Initialized populations fitness/function values.
				3. New global best solution
				4. New global best solutions fitness/objective value
				5. Additional arguments:
					* pa_v (float): TODO
		"""
		i = self.randint(self.NP)
		Nn = task.repair(pop[i] + self.alpha * levy.rvs(size=[task.D], random_state=self.Rand), rnd=self.Rand)
		Nn_f = task.eval(Nn)
		j = self.randint(self.NP)
		while i == j: j = self.randint(self.NP)
		if Nn_f <= fpop[j]: pop[j], fpop[j] = Nn, Nn_f
		pop, fpop = self.emptyNests(pop, fpop, pa_v, task)
		xb, fxb = self.getBest(pop, fpop, xb, fxb)
		return pop, fpop, xb, fxb, {'pa_v': pa_v}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
