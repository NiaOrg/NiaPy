# encoding=utf8
import logging

from numpy import exp, random as rand, asarray

from NiaPy.algorithms.algorithm import Algorithm, Individual
from NiaPy.util.utility import objects2array

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CamelAlgorithm']

class Camel(Individual):
	r"""Implementation of population individual that is a camel for Camel algorithm.

	Algorithm:
		Camel algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		E (float): Camel endurance.
		S (float): Camel supply.
		x_past (numpy.ndarray): Camel's past position.
		f_past (float): Camel's past funciton/fitness value.
		steps (int): Age of camel.

	See Also:
		* :class:`NiaPy.algorithms.Individual`
	"""
	def __init__(self, E_init=None, S_init=None, **kwargs):
		r"""Initialize the Camel.

		Args:
			E_init (Optional[float]): Starting endurance of Camel.
			S_init (Optional[float]): Stating supply of Camel.
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Individual.__init__`
		"""
		Individual.__init__(self, **kwargs)
		self.E, self.E_past = E_init, E_init
		self.S, self.S_past = S_init, S_init
		self.x_past, self.f_past = self.x, self.f
		self.steps = 0

	def nextT(self, T_min, T_max, rnd=rand):
		r"""Apply nextT function on Camel.

		Args:
			T_min (float): TODO
			T_max (float): TODO
			rnd (Optional[mtrand.RandomState]): Random number generator.
		"""
		self.T = (T_max - T_min) * rnd.rand() + T_min

	def nextS(self, omega, n_gens):
		r"""Apply nextS on Camel.

		Args:
			omega (float): TODO.
			n_gens (int): Number of Camel Algorithm iterations/generations.
		"""
		self.S = self.S_past * (1 - omega * self.steps / n_gens)

	def nextE(self, n_gens, T_max):
		r"""Apply function nextE on function on Camel.

		Args:
			n_gens (int): Number of Camel Algorithm iterations/generations
			T_max (float): Maximum temperature of environment
		"""
		self.E = self.E_past * (1 - self.T / T_max) * (1 - self.steps / n_gens)

	def nextX(self, cb, E_init, S_init, task, rnd=rand):
		r"""Apply function nextX on Camel.

		This method/function move this Camel to new position in search space.

		Args:
			cb (Camel): Best Camel in population.
			E_init (float): Starting endurance of camel.
			S_init (float): Starting supply of camel.
			task (Task): Optimization task.
			rnd (Optional[mtrand.RandomState]): Random number generator.
		"""
		delta = -1 + rnd.rand() * 2
		self.x = self.x_past + delta * (1 - (self.E / E_init)) * exp(1 - self.S / S_init) * (cb - self.x_past)
		if not task.isFeasible(self.x): self.x = self.x_past
		else: self.f = task.eval(self.x)

	def next(self):
		r"""Save new position of Camel to old position."""
		self.x_past, self.f_past, self.E_past, self.S_past = self.x.copy(), self.f, self.E, self.S
		self.steps += 1
		return self

	def refill(self, S=None, E=None):
		r"""Apply this function to Camel.

		Args:
			S (float): New value of Camel supply.
			E (float): New value of Camel endurance.
		"""
		self.S, self.E = S, E

class CamelAlgorithm(Algorithm):
	r"""Implementation of Camel traveling behavior.

	Algorithm:
		Camel algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.iasj.net/iasj?func=fulltext&aId=118375

	Reference paper:
		Ali, Ramzy. (2016). Novel Optimization Algorithm Inspired by Camel Traveling Behavior. Iraq J. Electrical and Electronic Engineering. 12. 167-177.

	Attributes:
		Name (List[str]): List of strings representing name of the algorithm.
		T_min (float): Minimal temperature of environment.
		T_max (float): Maximal temperature of environment.
		E_init (float): Starting value of energy.
		S_init (float): Starting value of supplys.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['CamelAlgorithm', 'CA']

	@staticmethod
	def algorithmInfo():
		r"""Get information about algorithm.

		Returns:
			str: Algorithm information
		"""
		return r'''Ali, Ramzy. (2016). Novel Optimization Algorithm Inspired by Camel Traveling Behavior. Iraq J. Electrical and Electronic Engineering. 12. 167-177.'''

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* omega (Callable[[Union[int, float]], bool])
				* mu (Callable[[float], bool])
				* alpha (Callable[[float], bool])
				* S_init (Callable[[Union[float, int]], bool])
				* E_init (Callable[[Union[float, int]], bool])
				* T_min (Callable[[Union[float, int], bool])
				* T_max (Callable[[Union[float, int], bool])

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'omega': lambda x: isinstance(x, (float, int)),
			'mu': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'alpha': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'S_init': lambda x: isinstance(x, (float, int)) and x > 0,
			'E_init': lambda x: isinstance(x, (float, int)) and x > 0,
			'T_min': lambda x: isinstance(x, (float, int)) and x > 0,
			'T_max': lambda x: isinstance(x, (float, int)) and x > 0
		})
		return d

	def setParameters(self, NP=50, omega=0.25, mu=0.5, alpha=0.5, S_init=10, E_init=10, T_min=-10, T_max=10, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			NP (Optional[int]): Population size :math:`\in [1, \infty)`.
			T_min (Optional[float]): Minimum temperature, must be true :math:`$T_{min} < T_{max}`.
			T_max (Optional[float]): Maximum temperature, must be true :math:`T_{min} < T_{max}`.
			omega (Optional[float]): Burden factor :math:`\in [0, 1]`.
			mu (Optional[float]): Dying rate :math:`\in [0, 1]`.
			S_init (Optional[float]): Initial supply :math:`\in (0, \infty)`.
			E_init (Optional[float]): Initial endurance :math:`\in (0, \infty)`.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, itype=Camel, InitPopFunc=ukwargs.pop('InitPopFunc', self.initPop), **ukwargs)
		self.omega, self.mu, self.alpha, self.S_init, self.E_init, self.T_min, self.T_max = omega, mu, alpha, S_init, E_init, T_min, T_max

	def getParameters(self):
		r"""Get parameters of the algorithm.

		Returns:
			Dict[str, Any]:
		"""
		d = Algorithm.getParameters(self)
		d.update({
			'omega': self.omega,
			'mu': self.mu,
			'alpha': self.alpha,
			'S_init': self.S_init,
			'E_init': self.E_init,
			'T_min': self.T_min,
			'T_max': self.T_max
		})
		return d

	def initPop(self, task, NP, rnd, itype, **kwargs):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.
			NP (int): Number of camels in population.
			rnd (mtrand.RandomState): Random number generator.
			itype (Individual): Individual type.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray[Camel], numpy.ndarray[float]]:
				1. Initialize population of camels.
				2. Initialized populations function/fitness values.
		"""
		caravan = objects2array([itype(E_init=self.E_init, S_init=self.S_init, task=task, rnd=rnd, e=True) for _ in range(NP)])
		return caravan, asarray([c.f for c in caravan])

	def walk(self, c, cb, task):
		r"""Move the camel in search space.

		Args:
			c (Camel): Camel that we want to move.
			cb (Camel): Best know camel.
			task (Task): Optimization task.

		Returns:
			Camel: Camel that moved in the search space.
		"""
		c.nextT(self.T_min, self.T_max, self.Rand)
		c.nextS(self.omega, task.nGEN)
		c.nextE(task.nGEN, self.T_max)
		c.nextX(cb, self.E_init, self.S_init, task, self.Rand)
		return c

	def oasis(self, c, rn, alpha):
		r"""Apply oasis function to camel.

		Args:
			c (Camel): Camel to apply oasis on.
			rn (float): Random number.
			alpha (float): View range of Camel.

		Returns:
			Camel: Camel with appliyed oasis on.
		"""
		if rn > 1 - alpha and c.f < c.f_past: c.refill(self.S_init, self.E_init)
		return c

	def lifeCycle(self, c, mu, task):
		r"""Apply life cycle to Camel.

		Args:
			c (Camel): Camel to apply life cycle.
			mu (float): Vision range of camel.
			task (Task): Optimization task.

		Returns:
			Camel: Camel with life cycle applyed to it.
		"""
		if c.f_past < mu * c.f: return Camel(self.E_init, self.S_init, rnd=self.Rand, task=task)
		else: return c.next()

	def initPopulation(self, task):
		r"""Initialize population.

		Args:
			task (Task): Optimization taks.

		Returns:
			Tuple[numpy.ndarray[Camel], numpy.ndarray[float], dict]:
				1. New population of Camels.
				2. New population fitness/function values.
				3. Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		caravan, fcaravan, _ = Algorithm.initPopulation(self, task)
		return caravan, fcaravan, {}

	def runIteration(self, task, caravan, fcaravan, cb, fcb, **dparams):
		r"""Core function of Camel Algorithm.

		Args:
			task (Task): Optimization task.
			caravan (numpy.ndarray[Camel]): Current population of Camels.
			fcaravan (numpy.ndarray[float]): Current population fitness/function values.
			cb (Camel): Current best Camel.
			fcb (float): Current best Camel fitness/function value.
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, folat, dict]:
				1. New population
				2. New population function/fitness value
				3. New global best solution
				4. New global best fitness/objective value
				5. Additional arguments
		"""
		ncaravan = objects2array([self.walk(c, cb, task) for c in caravan])
		ncaravan = objects2array([self.oasis(c, self.rand(), self.alpha) for c in ncaravan])
		ncaravan = objects2array([self.lifeCycle(c, self.mu, task) for c in ncaravan])
		fncaravan = asarray([c.f for c in ncaravan])
		cb, fcb = self.getBest(ncaravan, fncaravan, cb, fcb)
		return ncaravan, fncaravan, cb, fcb, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
