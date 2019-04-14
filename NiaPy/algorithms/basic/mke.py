# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, len-as-condition, singleton-comparison, arguments-differ, bad-continuation
import logging
from math import ceil

from numpy import apply_along_axis, vectorize, argmin, argmax, full, tril

from NiaPy.algorithms.algorithm import Algorithm, Individual, defaultIndividualInit, defaultNumPyInit

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['MonkeyKingEvolutionV1', 'MonkeyKingEvolutionV2', 'MonkeyKingEvolutionV3']

class MkeSolution(Individual):
	r"""Implementation of Monkey King Evolution individual.

	Data:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		x_pb (array of (float or int)): Persional best position of Monkey patricle
		f_pb (float): Personal best fitness/function value
		MonkeyKing (bool): Boolean value indicating if particle is Monkey King paticle

	See Also:
		* :class:`NiaPy.algorithms.Individual`
	"""
	def __init__(self, **kwargs):
		r"""Initialize Monkey particle.

		Args:
			**kwargs: Additional arguments

		See Also:
			* :class:`NiaPy.algorithms.Individual.__init__()`
		"""
		Individual.__init__(self, **kwargs)
		self.f_pb, self.x_pb = self.f, self.x
		self.MonkeyKing = False

	def uPersonalBest(self):
		r"""Update presonal best position of particle."""
		if self.f < self.f_pb: self.x_pb, self.f_pb = self.x, self.f

class MonkeyKingEvolutionV1(Algorithm):
	r"""Implementation of monkey king evolution algorithm version 1.

	Algorithm:
		Monkey King Evolution version 1

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.sciencedirect.com/science/article/pii/S0950705116000198

	Reference paper:
		Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009.

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		F (float): Scale factor for normal particles.
		R (float): TODO.
		C (int): Number of new particles generated by Monkey King particle.
		FC (float): Scale factor for Monkey King particles.

	See Also:
		* :class:`NiaPy.algorithms.algorithm.Algorithm`
	"""
	Name = ['MonkeyKingEvolutionV1', 'MKEv1']

	@staticmethod
	def typeParameters():
		r"""TODO.

		Returns:
			Dict[str, Callable]:
				* F (func): TODO
				* R (func): TODO
				* C (func): TODO
				* FC (func): TODO
		"""
		d = Algorithm.typeParameters()
		d.update({
			'NP': lambda x: isinstance(x, int) and x > 0,
			'F': lambda x: isinstance(x, (float, int)) and x > 0,
			'R': lambda x: isinstance(x, (float, int)) and x > 0,
			'C': lambda x: isinstance(x, int) and x > 0,
			'FC': lambda x: isinstance(x, (float, int)) and x > 0
		})
		return d

	def setParameters(self, NP=40, F=0.7, R=0.3, C=3, FC=0.5, **ukwargs):
		r"""Set Monkey King Evolution v1 algorithms static parameters.

		Args:
			NP (int): Population size.
			F (float): Scale factor for normal particle.
			R (float): Procentual value of now many new particle Monkey King particle creates. Value in rage [0, 1].
			C (int): Number of new particles generated by Monkey King particle.
			FC (float): Scale factor for Monkey King particles.
			**ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, itype=ukwargs.pop('itype', MkeSolution), InitPopFunc=ukwargs.pop('InitPopFunc', defaultIndividualInit), **ukwargs)
		self.F, self.R, self.C, self.FC = F, R, C, FC
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def moveP(self, x, x_pb, x_b, task):
		r"""Move normal particle in search space.

		For moving particles algorithm uses next formula:
		:math:`\mathbf{x_{pb} - \mathit{F} \odot \mathbf{r} \odot (\mathbf{x_b} - \mathbf{x})`
		where
		:math:`\mathbf{r}` is one dimension array with `D` components. Components in this vector are in range [0, 1].

		Args:
			x (numpy.ndarray): Paticle position.
			x_pb (numpy.ndarray): Particle best position.
			x_b (numpy.ndarray): Best particle position.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: Particle new position.
		"""
		return x_pb + self.F * self.rand(task.D) * (x_b - x)

	def moveMK(self, x, task):
		r"""Move Mokey King paticle.

		For moving Monkey King particles algorithm uses next formula:
		:math:`\mathbf{x} + \mathit{FC} \odot \mathbf{R} \odot \mathbf{x}`
		where
		:math:`\mathbf{R}` is two dimensional array with shape `{C * D, D}`. Componentes of this array are in range [0, 1]

		Args:
			x (numpy.ndarray): Monkey King patricle position.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: New particles generated by Monkey King particle.
		"""
		return x + self.FC * self.rand([int(self.C * task.D), task.D]) * x

	def movePartice(self, p, p_b, task):
		r"""Move patricles.

		Args:
			p (MkeSolution): Monke particle.
			p_b (MkeSolution): Population best particle.
			task (Task): Optimization task.
		"""
		p.x = self.moveP(p.x, p.x_pb, p_b.x, task)
		p.evaluate(task, rnd=self.Rand)

	def moveMokeyKingPartice(self, p, task):
		r"""Move Monky King Particles.

		Args:
			p (MkeSolution): Monkey King particle to apply this function on.
			task (Task): Optimization task
		"""
		p.MonkeyKing = False
		A = apply_along_axis(task.repair, 1, self.moveMK(p.x, task), self.Rand)
		A_f = apply_along_axis(task.eval, 1, A)
		ib = argmin(A_f)
		p.x, p.f = A[ib], A_f[ib]

	def movePopulation(self, pop, xb, task):
		r"""Move population.

		Args:
			pop (numpy.ndarray[MkeSolution]): Current population.
			xb (MkeSolution): Current best solution.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray[MkeSolution]: New particles.
		"""
		for p in pop:
			if p.MonkeyKing: self.moveMokeyKingPartice(p, task)
			else: self.movePartice(p, xb, task)
			p.uPersonalBest()
		return pop

	def initPopulation(self, task):
		r"""Init population.

		Args:
			task (Task): Optimization task

		Returns:
			Tuple(numpy.ndarray[MkeSolution], numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized solutions
				2. Fitness/function values of solution
				3. Additional arguments
		"""
		pop, fpop, _ = Algorithm.initPopulation(self, task)
		for i in self.Rand.choice(self.NP, int(self.R * len(pop)), replace=False): pop[i].MonkeyKing = True
		return pop, fpop, {}

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core function of Monkey King Evolution v1 algorithm.

		Args:
			task (Task): Optimization task
			pop (numpy.ndarray[MkeSolution]): Current population
			fpop (numpy.ndarray[float]): Current population fitness/function values
			xb (MkeSolution): Current best solution.
			fxb (float): Current best solutions function/fitness value.
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple(numpy.ndarray[MkeSolution], numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized solutions.
				2. Fitness/function values of solution.
				3. Additional arguments.
		"""
		pop = self.movePopulation(pop, xb, task)
		for i in self.Rand.choice(self.NP, int(self.R * len(pop)), replace=False): pop[i].MonkeyKing = True
		return pop, [m.f for m in pop], {}

class MonkeyKingEvolutionV2(MonkeyKingEvolutionV1):
	r"""Implementation of monkey king evolution algorithm version 2.

	Algorithm:
		Monkey King Evolution version 2

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.sciencedirect.com/science/article/pii/S0950705116000198

	Reference paper:
		Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009.

	Attributes:
		Name (List[str]): List of strings representing algorithm names.

	See Also:
		* :class:`NiaPy.algorithms.basic.mke.MonkeyKingEvolutionV1`
	"""
	Name = ['MonkeyKingEvolutionV2', 'MKEv2']

	def moveMK(self, x, dx, task):
		r"""Move Monkey King particle.

		For movment of particles algorithm uses next formula:
		:math:`\mathbf{x} - \mathit{FC} \odot \mathbf{dx}`

		Args:
			x (numpy.ndarray): Particle to apply movment on.
			dx (numpy.ndarray): Difference between to random paricles in population.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: Moved particles.
		"""
		return x - self.FC * dx

	def moveMokeyKingPartice(self, p, pop, task):
		r"""Move Monkey King particles.

		Args:
			p (MkeSolution): Monkey King particle to move.
			pop (numpy.ndarray[MkeSolution]): Current population.
			task (Task): Optimization task.
		"""
		p.MonkeyKing = False
		p_b, p_f = p.x, p.f
		for _i in range(int(self.C * self.NP)):
			r = self.Rand.choice(self.NP, 2, replace=False)
			a = task.repair(self.moveMK(p.x, pop[r[0]].x - pop[r[1]].x, task), self.Rand)
			a_f = task.eval(a)
			if a_f < p_f: p_b, p_f = a, a_f
		p.x, p.f = p_b, p_f

	def movePopulation(self, pop, xb, task):
		r"""Move population.

		Args:
			pop (numpy.ndarray[MkeSolution]): Current population.
			xb (MkeSolution): Current best solution.
			task (Task): Optimization task.

		Returns:
			numpy.ndarray[MkeSolution]: Moved population.
		"""
		for p in pop:
			if p.MonkeyKing: self.moveMokeyKingPartice(p, pop, task)
			else: self.movePartice(p, xb, task)
			p.uPersonalBest()
		return pop

class MonkeyKingEvolutionV3(MonkeyKingEvolutionV1):
	r"""Implementation of monkey king evolution algorithm version 3.

	Algorithm:
		Monkey King Evolution version 3

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.sciencedirect.com/science/article/pii/S0950705116000198

	Reference paper:
		Zhenyu Meng, Jeng-Shyang Pan, Monkey King Evolution: A new memetic evolutionary algorithm and its application in vehicle fuel consumption optimization, Knowledge-Based Systems, Volume 97, 2016, Pages 144-157, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2016.01.009.

	Attributes:
		Name (List[str]): List of strings that represent algorithm names.

	See Also:
		* :class:`NiaPy.algorithms.basic.mke.MonkeyKingEvolutionV1`
	"""
	Name = ['MonkeyKingEvolutionV3', 'MKEv3']

	def setParameters(self, **ukwargs):
		r"""Set core parameters of MonkeyKingEvolutionV3 algorithm.

		Args:
			**ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.basic.MonkeyKingEvolutionV1.setParameters`
		"""
		MonkeyKingEvolutionV1.setParameters(self, itype=ukwargs.pop('itype', None), InitPopFunc=ukwargs.pop('InitPopFunc', defaultNumPyInit), **ukwargs)

	def neg(self, x):
		r"""Transform function.

		Args:
			x (Union[int, float]): Sould be 0 or 1.

		Returns:
			float: If 0 thet 1 else 1 then 0.
		"""
		return 0.0 if x == 1.0 else 1.0

	def initPopulation(self, task):
		r"""Initialize the population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized population.
				2. Initialized population function/fitness values.
				3. Additional arguments:
					* k (int): TODO.
					* c (int): TODO.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.initPopulation`
		"""
		X, X_f, d = Algorithm.initPopulation(self, task)
		k, c = int(ceil(self.NP / task.D)), int(ceil(self.C * task.D))
		d.update({'k': k, 'c': c})
		return X, X_f, d

	def runIteration(self, task, X, X_f, xb, fxb, k, c, **dparams):
		r"""Core funciton of Monkey King Evolution v3 algorithm.

		Args:
			task (Task): Optimization task
			X (numpy.ndarray): Current population
			X_f (numpy.ndarray[float]): Current population fitness/function values
			xb (numpy.ndarray): Current best individual
			fxb (float): Current best individual function/fitness value
			k (int): TODO
			c (int: TODO
			**dparams: Additional arguments

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized population.
				2. Initialized population function/fitness values.
				3. Additional arguments:
					* k (int): TODO.
					* c (int): TODO.
		"""
		X_gb = apply_along_axis(task.repair, 1, xb + self.FC * X[self.Rand.choice(len(X), c)] - X[self.Rand.choice(len(X), c)], self.Rand)
		X_gb_f = apply_along_axis(task.eval, 1, X_gb)
		M = full([self.NP, task.D], 1.0)
		for i in range(k): M[i * task.D:(i + 1) * task.D] = tril(M[i * task.D:(i + 1) * task.D])
		for i in range(self.NP): self.Rand.shuffle(M[i])
		X = apply_along_axis(task.repair, 1, M * X + vectorize(self.neg)(M) * xb, self.Rand)
		X_f = apply_along_axis(task.eval, 1, X)
		iw, ib_gb = argmax(X_f), argmin(X_gb_f)
		if X_gb_f[ib_gb] <= X_f[iw]: X[iw], X_f[iw] = X_gb[ib_gb], X_gb_f[ib_gb]
		return X, X_f, {'k': k, 'c': c}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
