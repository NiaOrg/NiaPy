# encoding=utf8
import logging

import numpy as np
from numpy import random as rand

from NiaPy.algorithms.algorithm import Algorithm, Individual, defaultIndividualInit

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['BacterialForagingOptimizationAlgorithm']


class Cell(Individual):
	r"""Representation of a single bacterium.

	Date:
		2021

	Author:
		Žiga Stupan

	License:
		MIT

	Attributes:
		x (numpy.ndarray): Coordinates of cell.
		f (float): Function/fitness value of cell.
		cost (float): Cost of cell i.e. the sum of the fitness value and cell to cell ineraction J_cc.
		health (float): Health of cell i.e. the sum of nutrients (cost) the cell got over it's lifetime.
	"""

	def __init__(self, x=None, task=None, e=True, rnd=rand, **kwargs):
		r"""Initialize new cell.

		Args:
			x (Optional[numpy.ndarray]): Coordinates of cell, if None, a random solution will be generated in the search space defined in task.
			task (Optional[Task]): Optimization task.
			e (bool): True if cell is to be evaluated at initialization.
			rand (Optional[mtrand.RandomState]): Random generator.
		"""
		super().__init__(x=x, task=task, e=e, rnd=rnd, **kwargs)
		self.cost = 0.0  # J(i, j, k, l) = J(i, j, k, l) + J_cc
		self.health = 0.0

	def __lt__(self, other):
		r"""Less than operator.

		Args:
			other (Cell): Cell to compare to

		Returns:
			bool: True if self.health < other.health
		"""
		return self.health < other.health


class BacterialForagingOptimizationAlgorithm(Algorithm):
	r"""Implementation of the Bacterial foraging optimization algorithm.

	Date:
		2021

	Author:
		Žiga Stupan

	License:
		MIT

	Reference paper:
		K. M. Passino, "Biomimicry of bacterial foraging for distributed optimization and control," in IEEE Control Systems Magazine, vol. 22, no. 3, pp. 52-67, June 2002, doi: 10.1109/MCS.2002.1004010.

	Attributes:
		Name (List[str]): list of strings representing algorithm names.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""

	Name = ['BacterialForagingOptimizationAlgorithm', 'BFOA', 'BFO']

	@staticmethod
	def algorithmInfo():
		r"""Get algorithm information.

		Returns:
			str: Bit item.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""

		return r"""K. M. Passino, "Biomimicry of bacterial foraging for distributed optimization and control," in IEEE Control Systems Magazine, vol. 22, no. 3, pp. 52-67, June 2002, doi: 10.1109/MCS.2002.1004010."""

	@staticmethod
	def typeParameters():
		r"""Return functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""

		d = Algorithm.typeParameters()
		d.update({
			'n_chemotactic': lambda x: isinstance(x, int) and x > 0,
			'n_swim': lambda x: isinstance(x, int) and x > 0,
			'n_reproduction': lambda x: isinstance(x, int) and x > 0,
			'n_elimination': lambda x: isinstance(x, int) and x > 0,
			'prob_elimination': lambda x: isinstance(x, float) and 0 <= x <= 1,
			'step_size': lambda x: isinstance(x, float) and x > 0,
			'd_attract': lambda x: isinstance(x, float) and x > 0,
			'w_attract': lambda x: isinstance(x, float) and x > 0,
			'h_repel': lambda x: isinstance(x, float) and x > 0,
			'w_repel': lambda x: isinstance(x, float) and x > 0
		})

		return d

	def __init__(self, **kwargs):
		r"""Initialize algorithm.

		Args:
			seed (int): Starting seed for random generator.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.__init__`
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""

		super().__init__(**kwargs)

		self.i = 0  # elimination and dispersal step counter
		self.j = 0  # reproduction step counter
		self.k = 0  # chemotaxis step counter

	def setParameters(self, NP=50, n_chemotactic=100, n_swim=4, n_reproduction=4, n_elimination=2, prob_elimination=0.25, step_size=0.1, d_attract=0.1, w_attract=0.2, h_repel=0.1, w_repel=10.0, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		Args:
			NP (Optional[int]): Number of individuals in population :math:`\in [1, \infty]`.
			n_chemotactic (Optional[int]): Number of chemotactic steps.
			n_swim (Optional[int]): Number of swim steps.
			n_reproduction (Optional[int]): Number of reproduction steps.
			n_elimination (Optional[int]): Number of elimination and dispersal steps.
			prob_elimination (Optional[float]): Probability of a bacterium being eliminated and a new one being created at a random location in the search space.
			step_size (Optional[float]): Size of a chemotactic step.
			d_attract (Optional[float]): Depth of the attractant released by the cell (a quantification of how much attractant is released).
			w_attract (Optional[float]): Width of the attractant signal (a quantification of the diffusion rate of the chemical).
			h_repel (Optional[float]): Height of the repellant effect (magnitude of its effect).
			w_repel (Optional[float]): Width of the repellant.
			**kwargs (Dict[str, Any]): Additional arguments.
		"""

		super().setParameters(NP=NP, InitPopFunc=defaultIndividualInit, itype=Cell, **kwargs)
		self.n_chemotactic = n_chemotactic
		self.n_swim = n_swim
		self.n_reproduction = n_reproduction
		self.n_elimination = n_elimination
		self.prob_elimination = prob_elimination
		self.step_size = step_size
		self.d_attract = d_attract
		self.w_attract = w_attract
		self.h_repel = h_repel
		self.w_repel = w_repel

	def getParameters(self):
		r"""Get parameters of the algorithm.

		Returns:
			Dict[str, Any]:
			* Parameter name (str): Represents a parameter name
			* Value of parameter (Any): Represents the value of the parameter
		"""

		params = super().getParameters()
		params.update({
			'n_chemotactic': self.n_chemotactic,
			'n_swim': self.n_swim,
			'n_reproduction': self.n_reproduction,
			'n_elimination': self.n_elimination,
			'prob_elimination': self.prob_elimination,
			'step_size': self.step_size,
			'd_attract': self.d_attract,
			'w_attract': self.w_attract,
			'h_repel': self.h_repel,
			'w_repel': self.w_repel
		})

		return params

	def interaction(self, cell, population):
		r"""Compute cell to cell interaction J_cc.

		Args:
			cell (Cell): Cell to compute interaction for.
			population (numpy.ndarray[Cell]): Population

		Returns:
			float: Cell to cell interaction J_cc
		"""

		attract = 0.0
		repel = 0.0

		for c in population:
			diff = np.sum(np.square(cell.x - c.x))
			attract += -1.0 * self.d_attract * np.exp(-1.0 * self.w_attract * diff)
			repel += self.h_repel * np.exp(-1.0 * self.w_repel * diff)
		return attract + repel

	def random_direction(self, dimension):
		r"""Generate a random direction vector.

		Args:
			dimension (int): Problem dimension

		Returns:
			numpy.ndarray: Normalised random direction vector
		"""
		delta = self.uniform(-1.0, 1.0, dimension)
		return delta / np.linalg.norm(delta)

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core function of Bacterial Foraging Optimization algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Current populations fitness/function values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individuals function/fitness value.
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population.
				2. New populations function/fitness values.
				3. New global best solution,
				4. New global best solutions fitness/objective value.
				5. Additional arguments.
		"""

		# Chemotaxis

		for i in range(len(pop)):
			pop[i].cost = pop[i].f + self.interaction(pop[i], pop)
			j_last = pop[i].cost
			step_direction = self.random_direction(task.D)
			pop[i].x = pop[i].x + self.step_size * step_direction
			pop[i].evaluate(task)

			if pop[i].f < fxb:
				xb = pop[i].x.copy()
				fxb = pop[i].f

			pop[i].cost = pop[i].f + self.interaction(pop[i], pop)
			pop[i].health += pop[i].cost

			for _ in range(self.n_swim):
				if pop[i].cost < j_last:
					pop[i].x = pop[i].x + self.step_size * step_direction
					pop[i].evaluate(task)

					if pop[i].f < fxb:
						xb = pop[i].x.copy()
						fxb = pop[i].f

					pop[i].cost = pop[i].f + self.interaction(pop[i], pop)
					pop[i].health += pop[i].cost

		self.k += 1

		if self.k >= self.n_chemotactic:

			self.k = 0
			self.j += 1

			# Reproduction
			pop = np.sort(pop)  # sort by health
			pop = np.tile(pop[:self.NP // 2], 2)  # keep half of the healthiest cells and duplicate them
			for i in range(len(pop)):
				pop[i].health = 0.0

		if self.j >= self.n_reproduction:
			self.j = 0
			self.i += 1

			# Elimination and dispersal
			for i in range(len(pop)):
				if self.rand() <= self.prob_elimination:  # Eliminate i-th bacterium with the probability 'prob_elimination' and replace it with a new one at a random location
					pop[i].generateSolution(task, self.Rand)
					pop[i].evaluate(task)
					if pop[i].f < fxb:
						xb = pop[i].x.copy()
						fxb = pop[i].f

		if self.i == self.n_elimination - 1 and self.j == self.n_reproduction - 1 and self.k == self.n_chemotactic - 1:  # if last iteration, reset counters
			self.i = 0
			self.j = 0
			self.k = 0

		return pop, np.asarray([c.f for c in pop]), xb, fxb, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
