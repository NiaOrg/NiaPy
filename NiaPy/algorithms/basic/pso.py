# encoding=utf8
import logging

import numpy as np

from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.util import fullArray
from NiaPy.util.utility import reflectRepair

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['ParticleSwarmAlgorithm', 'ParticleSwarmOptimization', 'OppositionVelocityClampingParticleSwarmOptimization', 'CenterParticleSwarmOptimization', 'MutatedParticleSwarmOptimization', 'MutatedCenterParticleSwarmOptimization', 'MutatedCenterUnifiedParticleSwarmOptimization', 'ComprehensiveLearningParticleSwarmOptimizer']

class ParticleSwarmAlgorithm(Algorithm):
	r"""Implementation of Particle Swarm Optimization algorithm.

	Algorithm:
		Particle Swarm Optimization algorithm

	Date:
		2018

	Authors:
		Lucija Brezočnik, Grega Vrbančič, Iztok Fister Jr. and Klemen Berkovič

	License:
		MIT

	Reference paper:
		TODO: Find the right paper

	Attributes:
		Name (List[str]): List of strings representing algorithm names
		C1 (float): Cognitive component.
		C2 (float): Social component.
		w (Union[float, numpy.ndarray[float]]): Inertial weight.
		vMin (Union[float, numpy.ndarray[float]]): Minimal velocity.
		vMax (Union[float, numpy.ndarray[float]]): Maximal velocity.
		Repair (Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, mtrnd.RandomState], numpy.ndarray]): Repair method for velocity.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['WeightedVelocityClampingParticleSwarmAlgorithm', 'WVCPSO']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""TODO find one"""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable[[Union[int, float]], bool]]:
			* NP (Callable[[int], bool])
			* C1 (Callable[[Union[int, float]], bool])
			* C2 (Callable[[Union[int, float]], bool])
			* w (Callable[[float], bool])
			* vMin (Callable[[Union[int, float]], bool])
			* vMax (Callable[[Union[int, float], bool])
		"""
		d = Algorithm.typeParameters()
		d.update({
			'C1': lambda x: isinstance(x, (int, float)) and x >= 0,
			'C2': lambda x: isinstance(x, (int, float)) and x >= 0,
			'w': lambda x: isinstance(x, float) and x >= 0,
			'vMin': lambda x: isinstance(x, (int, float)),
			'vMax': lambda x: isinstance(x, (int, float))
		})
		return d

	def setParameters(self, NP=25, C1=2.0, C2=2.0, w=0.7, vMin=-1.5, vMax=1.5, Repair=reflectRepair, **ukwargs):
		r"""Set Particle Swarm Algorithm main parameters.

		Args:
			NP (int): Population size
			C1 (float): Cognitive component.
			C2 (float): Social component.
			w (Union[float, numpy.ndarray]): Inertial weight.
			vMin (Union[float, numpy.ndarray]): Minimal velocity.
			vMax (Union[float, numpy.ndarray]): Maximal velocity.
			Repair (Callable[[np.ndarray, np.ndarray, np.ndarray, dict], np.ndarray]): Repair method for velocity.
			**ukwargs: Additional arguments

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.C1, self.C2, self.w, self.vMin, self.vMax, self.Repair = C1, C2, w, vMin, vMax, Repair

	def getParameters(self):
		r"""Get value of parametrs for this instance of algorithm.

		Returns:
			Dict[str, Union[int, float, np.ndarray]]: Dictionari which has parameters maped to values.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.getParameters`
		"""
		d = Algorithm.getParameters(self)
		d.update({
			'C1': self.C1,
			'C2': self.C2,
			'w': self.w,
			'vMin': self.vMin,
			'vMax': self.vMax
		})
		return d

	def init(self, task):
		r"""Initialize dynamic arguments of Particle Swarm Optimization algorithm.

		Args:
			task (Task): Optimization task.

		Returns:
			Dict[str, Union[float, np.ndarray]]:
				* w (numpy.ndarray): Inertial weight.
				* vMin (numpy.ndarray): Mininal velocity.
				* vMax (numpy.ndarray): Maximal velocity.
				* V (numpy.ndarray): Initial velocity of particle.
		"""
		return {
			'w': fullArray(self.w, task.D),
			'vMin': fullArray(self.vMin, task.D),
			'vMax': fullArray(self.vMax, task.D),
			'V': np.full([self.NP, task.D], 0.0)
		}

	def initPopulation(self, task):
		r"""Initialize population and dynamic arguments of the Particle Swarm Optimization algorithm.

		Args:
			task: Optimization task.

		Returns:
			Tuple[np.ndarray, np.ndarray, dict]:
			1. Initial population.
			2. Initial population fitness/function values.
			3. Additional arguments:
				* popb (numpy.ndarray): particles best population.
				* fpopb (numpy.ndarray[float]): particles best positions function/fitness value.
				* w (numpy.ndarray): Inertial weight.
				* vMin (numpy.ndarray): Minimal velocity.
				* vMax (numpy.ndarray): Maximal velocity.
				* V (numpy.ndarray): Initial velocity of particle.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		pop, fpop, d = Algorithm.initPopulation(self, task)
		d.update(self.init(task))
		d.update({'popb': pop.copy(), 'fpopb': fpop.copy()})
		return pop, fpop, d

	def updateVelocity(self, V, p, pb, gb, w, vMin, vMax, task, **kwargs):
		r"""Update particle velocity.

		Args:
			V (numpy.ndarray): Current velocity of particle.
			p (numpy.ndarray): Current position of particle.
			pb (numpy.ndarray): Personal best position of particle.
			gb (numpy.ndarray): Global best position of particle.
			w (numpy.ndarray): Weights for velocity adjustment.
			vMin (numpy.ndarray): Minimal velocity allowed.
			vMax (numpy.ndarray): Maximal velocity allowed.
			task (Task): Optimization task.
			kwargs: Additional arguments.

		Returns:
			numpy.ndarray: Updated velocity of particle.
		"""
		return self.Repair(w * V + self.C1 * self.rand(task.D) * (pb - p) + self.C2 * self.rand(task.D) * (gb - p), vMin, vMax)

	def runIteration(self, task, pop, fpop, xb, fxb, popb, fpopb, w, vMin, vMax, V, **dparams):
		r"""Core function of Particle Swarm Optimization algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current populations.
			fpop (numpy.ndarray): Current population fitness/function values.
			xb (numpy.ndarray): Current best particle.
			fxb (float): Current best particle fitness/function value.
			popb (numpy.ndarray): Particles best position.
			fpopb (numpy.ndarray): Particles best positions fitness/function values.
			w (numpy.ndarray): Inertial weights.
			vMin (numpy.ndarray): Minimal velocity.
			vMax (numpy.ndarray): Maximal velocity.
			V (numpy.ndarray): Velocity of particles.
			**dparams: Additional function arguments.

		Returns:
			Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
			1. New population.
			2. New population fitness/function values.
			3. New global best position.
			4. New global best positions function/fitness value.
			5. Additional arguments:
				* popb (numpy.ndarray): Particles best population.
				* fpopb (numpy.ndarray[float]): Particles best positions function/fitness value.
				* w (numpy.ndarray): Inertial weight.
				* vMin (numpy.ndarray): Minimal velocity.
				* vMax (numpy.ndarray): Maximal velocity.
				* V (numpy.ndarray): Initial velocity of particle.

		See Also:
			* :class:`NiaPy.algorithms.algorithm.runIteration`
		"""
		for i in range(len(pop)):
			V[i] = self.updateVelocity(V[i], pop[i], popb[i], xb, w, vMin, vMax, task)
			pop[i] = task.repair(pop[i] + V[i], rnd=self.Rand)
			fpop[i] = task.eval(pop[i])
			if fpop[i] < fpopb[i]:
				popb[i], fpopb[i] = pop[i].copy(), fpop[i]
				if fpop[i] < fxb: xb, fxb = pop[i].copy(), fpop[i]
		return pop, fpop, xb, fxb, {'popb': popb, 'fpopb': fpopb, 'w': w, 'vMin': vMin, 'vMax': vMax, 'V': V}

class ParticleSwarmOptimization(ParticleSwarmAlgorithm):
	r"""Implementation of Particle Swarm Optimization algorithm.

	Algorithm:
		Particle Swarm Optimization algorithm

	Date:
		2018

	Authors:
		Lucija Brezočnik, Grega Vrbančič, Iztok Fister Jr. and Klemen Berkovič

	License:
		MIT

	Reference paper:
		Kennedy, J. and Eberhart, R. "Particle Swarm Optimization". Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942--1948, 1995.

	Attributes:
		Name (List[str]): List of strings representing algorithm names
		C1 (float): Cognitive component.
		C2 (float): Social component.
		Repair (Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, mtrnd.RandomState], numpy.ndarray]): Repair method for velocity.

	See Also:
		* :class:`NiaPy.algorithms.basic.WeightedVelocityClampingParticleSwarmAlgorithm`
	"""
	Name = ['ParticleSwarmAlgorithm', 'PSO']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Kennedy, J. and Eberhart, R. "Particle Swarm Optimization". Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942--1948, 1995."""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable[[Union[int, float]], bool]]:
			* NP: Population size.
			* C1: Cognitive component.
			* C2: Social component.
		"""
		d = ParticleSwarmAlgorithm.typeParameters()
		d.pop('w', None), d.pop('vMin', None), d.pop('vMax', None)
		return d

	def setParameters(self, **ukwargs):
		r"""Set core parameters of algorithm.

		Args:
			**ukwargs (Dict[str, Any]): Additional parameters.

		See Also:
			* :func:`NiaPy.algorithms.basic.WeightedVelocityClampingParticleSwarmAlgorithm.setParameters`
		"""
		ukwargs.pop('w', None), ukwargs.pop('vMin', None), ukwargs.pop('vMax', None)
		ParticleSwarmAlgorithm.setParameters(self, w=1, vMin=-np.inf, vMax=np.inf, **ukwargs)

	def getParameters(self):
		r"""Get value of parametrs for this instance of algorithm.

		Returns:
			Dict[str, Union[int, float, np.ndarray]]: Dictionari which has parameters maped to values.

		See Also:
			* :func:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm.getParameters`
		"""
		d = ParticleSwarmAlgorithm.getParameters(self)
		d.pop('w', None), d.pop('vMin', None), d.pop('vMax', None)
		return d

class OppositionVelocityClampingParticleSwarmOptimization(ParticleSwarmAlgorithm):
	r"""Implementation of Opposition-Based Particle Swarm Optimization with Velocity Clamping.

	Algorithm:
		Opposition-Based Particle Swarm Optimization with Velocity Clamping

	Date:
		2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		Shahzad, Farrukh, et al. "Opposition-based particle swarm optimization with velocity clamping (OVCPSO)." Advances in Computational Intelligence. Springer, Berlin, Heidelberg, 2009. 339-348

	Attributes:
		p0: Probability of opposite learning phase.
		w_min: Minimum inertial weight.
		w_max: Maximum inertial weight.
		sigma: Velocity scaling factor.

	See Also:
		* :class:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm`
	"""
	Name = ['OppositionVelocityClampingParticleSwarmOptimization', 'OVCPSO']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Shahzad, Farrukh, et al. "Opposition-based particle swarm optimization with velocity clamping (OVCPSO)." Advances in Computational Intelligence. Springer, Berlin, Heidelberg, 2009. 339-348"""

	def setParameters(self, p0=.3, w_min=.4, w_max=.9, sigma=.1, C1=1.49612, C2=1.49612, **kwargs):
		r"""Set core algorithm parameters.

		Args:
			p0 (float): Probability of running Opposite learning.
			w_min (numpy.ndarray): Minimal value of weights.
			w_max (numpy.ndarray): Maximum value of weights.
			sigma (numpy.ndarray): Velocity range factor.
			**kwargs: Additional arguments.

		See Also:
			* :func:`NiaPy.algorithm.basic.ParticleSwarmAlgorithm.setParameters`
		"""
		kwargs.pop('w', None)
		ParticleSwarmAlgorithm.setParameters(self, w=w_max, C1=C1, C2=C2, **kwargs)
		self.p0, self.w_min, self.w_max, self.sigma = p0, w_min, w_max, sigma

	def getParameters(self):
		r"""Get value of parametrs for this instance of algorithm.

		Returns:
			Dict[str, Union[int, float, np.ndarray]]: Dictionari which has parameters maped to values.

		See Also:
			* :func:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm.getParameters`
		"""
		d = ParticleSwarmAlgorithm.getParameters(self)
		d.pop('vMin', None), d.pop('vMax', None)
		d.update({
			'p0': self.p0, 'w_min': self.w_min, 'w_max': self.w_max, 'sigma': self.sigma
		})
		return d

	def oppositeLearning(self, S_l, S_h, pop, fpop, task):
		r"""Run opposite learning phase.

		Args:
			S_l (numpy.ndarray): Lower limit of opposite particles.
			S_h (numpy.ndarray): Upper limit of opposite particles.
			pop (numpy.ndarray): Current populations positions.
			fpop (numpy.ndarray): Current populations functions/fitness values.
			task (Task): Optimization task.

		Returns:
			Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
			1. New particles position
			2. New particles function/fitness values
			3. New best position of opposite learning phase
			4. new best function/fitness value of opposite learning phase
		"""
		S_r = S_l + S_h
		S = np.asarray([S_r - e for e in pop])
		S_f = np.asarray([task.eval(e) for e in S])
		S, S_f = np.concatenate([pop, S]), np.concatenate([fpop, S_f])
		sinds = np.argsort(S_f)
		return S[sinds[:len(pop)]], S_f[sinds[:len(pop)]], S[sinds[0]], S_f[sinds[0]]

	def initPopulation(self, task):
		r"""Init starting population and dynamic parameters.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[np.ndarray, np.ndarray, dict]:
			1. Initialized population.
			2. Initialized populations function/fitness values.
			3. Additional arguments:
				* popb (numpy.ndarray): particles best population.
				* fpopb (numpy.ndarray[float]): particles best positions function/fitness value.
				* vMin (numpy.ndarray): Minimal velocity.
				* vMax (numpy.ndarray): Maximal velocity.
				* V (numpy.ndarray): Initial velocity of particle.
				* S_u (numpy.ndarray): Upper bound for opposite learning.
				* S_l (numpy.ndarray): Lower bound for opposite learning.
		"""
		pop, fpop, d = ParticleSwarmAlgorithm.initPopulation(self, task)
		S_l, S_h = task.Lower, task.Upper
		pop, fpop, _, _ = self.oppositeLearning(S_l, S_h, pop, fpop, task)
		pb_inds = np.where(fpop < d['fpopb'])
		d['popb'][pb_inds], d['fpopb'][pb_inds] = pop[pb_inds], fpop[pb_inds]
		d['vMin'], d['vMax'] = self.sigma * (task.Upper - task.Lower), self.sigma * (task.Lower - task.Upper)
		d.update({'S_l': S_l, 'S_h': S_h})
		return pop, fpop, d

	def runIteration(self, task, pop, fpop, xb, fxb, popb, fpopb, vMin, vMax, V, S_l, S_h, **dparams):
		r"""Core function of Opposite-based Particle Swarm Optimization with velocity clamping algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population.
			fpop (numpy.ndarray): Current populations function/fitness values.
			xb (numpy.ndarray): Current global best position.
			fxb (float): Current global best positions function/fitness value.
			popb (numpy.ndarray): Personal best position.
			fpopb (numpy.ndarray): Personal best positions function/fitness values.
			vMin (numpy.ndarray): Minimal allowed velocity.
			vMax (numpy.ndarray): Maximal allowed velocity.
			V (numpy.ndarray): Populations velocity.
			S_l (numpy.ndarray): Lower bound of opposite learning.
			S_h (numpy.ndarray): Upper bound of opposite learning.
			**dparams: Additional arguments.

		Returns:
			Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
			1. New population.
			2. New populations function/fitness values.
			3. New global best position.
			4. New global best positions function/fitness value.
			5. Additional arguments:
				* popb: particles best population.
				* fpopb: particles best positions function/fitness value.
				* vMin: Minimal velocity.
				* vMax: Maximal velocity.
				* V: Initial velocity of particle.
				* S_u: Upper bound for opposite learning.
				* S_l: Lower bound for opposite learning.
		"""
		if self.rand() < self.p0:
			pop, fpop, nb, fnb = self.oppositeLearning(S_l, S_h, pop, fpop, task)
			pb_inds = np.where(fpop < fpopb)
			popb[pb_inds], fpopb[pb_inds] = pop[pb_inds], fpop[pb_inds]
			if fnb < fxb: xb, fxb = nb.copy(), fnb
		else:
			w = self.w_max - ((self.w_max - self.w_min) / task.nGEN) * task.Iters
			for i in range(len(pop)):
				V[i] = self.updateVelocity(V[i], pop[i], popb[i], xb, w, vMin, vMax, task)
				pop[i] = task.repair(pop[i] + V[i], rnd=self.Rand)
				fpop[i] = task.eval(pop[i])
				if fpop[i] < fpopb[i]:
					popb[i], fpopb[i] = pop[i].copy(), fpop[i]
					if fpop[i] < fxb: xb, fxb = pop[i].copy(), fpop[i]
			vMin, vMax = self.sigma * np.min(pop, axis=0), self.sigma * np.max(pop, axis=0)
		return pop, fpop, xb, fxb, {'popb': popb, 'fpopb': fpopb, 'vMin': vMin, 'vMax': vMax, 'V': V, 'S_l': S_l, 'S_h': S_h}

class CenterParticleSwarmOptimization(ParticleSwarmAlgorithm):
	r"""Implementation of Center Particle Swarm Optimization.

	Algorithm:
		Center Particle Swarm Optimization

	Date:
		2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		H.-C. Tsai, Predicting strengths of concrete-type specimens using hybrid multilayer perceptrons with center-Unified particle swarm optimization, Adv. Eng. Softw. 37 (2010) 1104–1112.

	See Also:
		* :class:`NiaPy.algorithms.basic.WeightedVelocityClampingParticleSwarmAlgorithm`
	"""
	Name = ['CenterParticleSwarmOptimization', 'CPSO']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""H.-C. Tsai, Predicting strengths of concrete-type specimens using hybrid multilayer perceptrons with center-Unified particle swarm optimization, Adv. Eng. Softw. 37 (2010) 1104–1112."""

	def setParameters(self, **kwargs):
		r"""Set core algorithm parameters.

		Args:
			**kwargs: Additional arguments.

		See Also:
			:func:`NiaPy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.setParameters`
		"""
		kwargs.pop('vMin', None), kwargs.pop('vMax', None)
		ParticleSwarmAlgorithm.setParameters(self, vMin=-np.inf, vMax=np.inf, **kwargs)

	def getParameters(self):
		r"""Get value of parametrs for this instance of algorithm.

		Returns:
			Dict[str, Union[int, float, np.ndarray]]: Dictionari which has parameters maped to values.

		See Also:
			* :func:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm.getParameters`
		"""
		d = ParticleSwarmAlgorithm.getParameters(self)
		d.pop('vMin', None), d.pop('vMax', None)
		return d

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core function of algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population of particles.
			fpop (numpy.ndarray): Current particles function/fitness values.
			xb (numpy.ndarray): Current global best particle.
			fxb (numpy.ndarray): Current global best particles function/fitness value.
			**dparams: Additional arguments.

		Returns:
			Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
			1. New population of particles.
			2. New populations function/fitness values.
			3. New global best particle.
			4. New global best particle function/fitness value.
			5. Additional arguments.

		See Also:
			* :func:`NiaPy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.runIteration`
		"""
		pop, fpop, xb, fxb, d = ParticleSwarmAlgorithm.runIteration(self, task, pop, fpop, xb, fxb, **dparams)
		c = np.sum(pop, axis=0) / len(pop)
		fc = task.eval(c)
		if fc <= fxb: xb, fxb = c, fc
		return pop, fpop, xb, fxb, d

class MutatedParticleSwarmOptimization(ParticleSwarmAlgorithm):
	r"""Implementation of Mutated Particle Swarm Optimization.

	Algorithm:
		Mutated Particle Swarm Optimization

	Date:
		2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		H. Wang, C. Li, Y. Liu, S. Zeng, A hybrid particle swarm algorithm with cauchy mutation, Proceedings of the 2007 IEEE Swarm Intelligence Symposium (2007) 356–360.

	Attributes:
		nmutt (int): Number of mutations of global best particle.

	See Also:
		* :class:`NiaPy.algorithms.basic.WeightedVelocityClampingParticleSwarmAlgorithm`
	"""
	Name = ['MutatedParticleSwarmOptimization', 'MPSO']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""H. Wang, C. Li, Y. Liu, S. Zeng, A hybrid particle swarm algorithm with cauchy mutation, Proceedings of the 2007 IEEE Swarm Intelligence Symposium (2007) 356–360."""

	def setParameters(self, nmutt=10, **kwargs):
		r"""Set core algorithm parameters.

		Args:
			nmutt (int): Number of mutations of global best particle.
			**kwargs: Additional arguments.

		See Also:
			:func:`NiaPy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.setParameters`
		"""
		kwargs.pop('vMin', None), kwargs.pop('vMax', None)
		ParticleSwarmAlgorithm.setParameters(self, vMin=-np.inf, vMax=np.inf, **kwargs)
		self.nmutt = nmutt

	def getParameters(self):
		r"""Get value of parametrs for this instance of algorithm.

		Returns:
			Dict[str, Union[int, float, np.ndarray]]: Dictionari which has parameters maped to values.

		See Also:
			* :func:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm.getParameters`
		"""
		d = ParticleSwarmAlgorithm.getParameters(self)
		d.pop('vMin', None), d.pop('vMax', None)
		d.update({'nmutt': self.nmutt})
		return d

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core function of algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population of particles.
			fpop (numpy.ndarray): Current particles function/fitness values.
			xb (numpy.ndarray): Current global best particle.
			fxb (float): Current global best particles function/fitness value.
			**dparams: Additional arguments.

		Returns:
			Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
			1. New population of particles.
			2. New populations function/fitness values.
			3. New global best particle.
			4. New global best particle function/fitness value.
			5. Additional arguments.

		See Also:
			* :func:`NiaPy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.runIteration`
		"""
		pop, fpop, xb, fxb, d = ParticleSwarmAlgorithm.runIteration(self, task, pop, fpop, xb, fxb, **dparams)
		v = d['V']
		v_a = (np.sum(v, axis=0) / len(v))
		v_a = v_a / np.max(np.abs(v_a))
		for _ in range(self.nmutt):
			g = task.repair(xb + v_a * self.uniform(task.Lower, task.Upper), self.Rand)
			fg = task.eval(g)
			if fg <= fxb: xb, fxb = g, fg
		return pop, fpop, xb, fxb, d

class MutatedCenterParticleSwarmOptimization(CenterParticleSwarmOptimization):
	r"""Implementation of Mutated Particle Swarm Optimization.

	Algorithm:
		Mutated Center Particle Swarm Optimization

	Date:
		2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		TODO find one

	Attributes:
		nmutt (int): Number of mutations of global best particle.

	See Also:
		* :class:`NiaPy.algorithms.basic.CenterParticleSwarmOptimization`
	"""
	Name = ['MutatedCenterParticleSwarmOptimization', 'MCPSO']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""TODO find one"""

	def setParameters(self, nmutt=10, **kwargs):
		r"""Set core algorithm parameters.

		Args:
			nmutt (int): Number of mutations of global best particle.
			**kwargs: Additional arguments.

		See Also:
			:func:`NiaPy.algorithm.basic.CenterParticleSwarmOptimization.setParameters`
		"""
		kwargs.pop('vMin', None), kwargs.pop('vMax', None)
		ParticleSwarmAlgorithm.setParameters(self, vMin=-np.inf, vMax=np.inf, **kwargs)
		self.nmutt = nmutt

	def getParameters(self):
		r"""Get value of parametrs for this instance of algorithm.

		Returns:
			Dict[str, Union[int, float, np.ndarray]]: Dictionari which has parameters maped to values.

		See Also:
			* :func:`NiaPy.algorithms.basic.CenterParticleSwarmOptimization.getParameters`
		"""
		d = CenterParticleSwarmOptimization.getParameters(self)
		d.update({'nmutt': self.nmutt})
		return d

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core function of algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population of particles.
			fpop (numpy.ndarray): Current particles function/fitness values.
			xb (numpy.ndarray): Current global best particle.
			fxb (float: Current global best particles function/fitness value.
			**dparams: Additional arguments.

		Returns:
			Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
			1. New population of particles.
			2. New populations function/fitness values.
			3. New global best particle.
			4. New global best particle function/fitness value.
			5. Additional arguments.

		See Also:
			* :func:`NiaPy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.runIteration`
		"""
		pop, fpop, xb, fxb, d = CenterParticleSwarmOptimization.runIteration(self, task, pop, fpop, xb, fxb, **dparams)
		v = d['V']
		v_a = (np.sum(v, axis=0) / len(v))
		v_a = v_a / np.max(np.abs(v_a))
		for _ in range(self.nmutt):
			g = task.repair(xb + v_a * self.uniform(task.Lower, task.Upper), self.Rand)
			fg = task.eval(g)
			if fg <= fxb: xb, fxb = g, fg
		return pop, fpop, xb, fxb, d

class MutatedCenterUnifiedParticleSwarmOptimization(MutatedCenterParticleSwarmOptimization):
	r"""Implementation of Mutated Particle Swarm Optimization.

	Algorithm:
		Mutated Center Unified Particle Swarm Optimization

	Date:
		2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		Tsai, Hsing-Chih. "Unified particle swarm delivers high efficiency to particle swarm optimization." Applied Soft Computing 55 (2017): 371-383.

	Attributes:
		nmutt (int): Number of mutations of global best particle.

	See Also:
		* :class:`NiaPy.algorithms.basic.CenterParticleSwarmOptimization`
	"""
	Name = ['MutatedCenterUnifiedParticleSwarmOptimization', 'MCUPSO']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""Tsai, Hsing-Chih. "Unified particle swarm delivers high efficiency to particle swarm optimization." Applied Soft Computing 55 (2017): 371-383."""

	def setParameters(self, **kwargs):
		r"""Set core algorithm parameters.

		Args:
			**kwargs: Additional arguments.

		See Also:
			:func:`NiaPy.algorithm.basic.MutatedCenterParticleSwarmOptimization.setParameters`
		"""
		kwargs.pop('vMin', None), kwargs.pop('vMax', None)
		MutatedCenterParticleSwarmOptimization.setParameters(self, vMin=-np.inf, vMax=np.inf, **kwargs)

	def updateVelocity(self, V, p, pb, gb, w, vMin, vMax, task, **kwargs):
		r"""Update particle velocity.

		Args:
			V (numpy.ndarray): Current velocity of particle.
			p (numpy.ndarray): Current position of particle.
			pb (numpy.ndarray): Personal best position of particle.
			gb (numpy.ndarray): Global best position of particle.
			w (numpy.ndarray): Weights for velocity adjustment.
			vMin (numpy.ndarray): Minimal velocity allowed.
			vMax (numpy.ndarray): Maxmimal velocity allowed.
			task (Task): Optimization task.
			kwargs: Additional arguments.

		Returns:
			numpy.ndarray: Updated velocity of particle.
		"""
		r3 = self.rand(task.D)
		return self.Repair(w * V + self.C1 * self.rand(task.D) * (pb - p) * r3 + self.C2 * self.rand(task.D) * (gb - p) * (1 - r3), vMin, vMax)

class ComprehensiveLearningParticleSwarmOptimizer(ParticleSwarmAlgorithm):
	r"""Implementation of Mutated Particle Swarm Optimization.

	Algorithm:
		Comprehensive Learning Particle Swarm Optimizer

	Date:
		2019

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		J. J. Liang, A. K. Qin, P. N. Suganthan and S. Baskar, "Comprehensive learning particle swarm optimizer for global optimization of multimodal functions," in IEEE Transactions on Evolutionary Computation, vol. 10, no. 3, pp. 281-295, June 2006. doi: 10.1109/TEVC.2005.857610

	Reference URL:
		http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1637688&isnumber=34326

	Attributes:
		w0 (float): Inertia weight.
		w1 (float): Inertia weight.
		C (float): Velocity constant.
		m (int): Refresh rate.

	See Also:
		* :class:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm`
	"""
	Name = ['ComprehensiveLearningParticleSwarmOptimizer', 'CLPSO']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information of algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.algorithmInfo`
		"""
		return r"""J. J. Liang, A. K. Qin, P. N. Suganthan and S. Baskar, "Comprehensive learning particle swarm optimizer for global optimization of multimodal functions," in IEEE Transactions on Evolutionary Computation, vol. 10, no. 3, pp. 281-295, June 2006. doi: 10.1109/TEVC.2005.857610	"""

	def setParameters(self, m=10, w0=.9, w1=.4, C=1.49445, **ukwargs):
		r"""Set Particle Swarm Algorithm main parameters.

		Args:
			w0 (int): Inertia weight.
			w1 (float): Inertia weight.
			C (float): Velocity constant.
			m (float): Refresh rate.
			**ukwargs: Additional arguments

		See Also:
			* :func:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm.setParameters`
		"""
		ParticleSwarmAlgorithm.setParameters(self, **ukwargs)
		self.m, self.w0, self.w1, self.C = m, w0, w1, C

	def getParameters(self):
		r"""Get value of parametrs for this instance of algorithm.

		Returns:
			Dict[str, Union[int, float, np.ndarray]]: Dictionari which has parameters maped to values.

		See Also:
			* :func:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm.getParameters`
		"""
		d = ParticleSwarmAlgorithm.getParameters(self)
		d.update({
			'm': self.m, 'w0': self.w0, 'w1': self.w1, 'C': self.C
		})
		return d

	def init(self, task):
		r"""Initialize dynamic arguments of Particle Swarm Optimization algorithm.

		Args:
			task (Task): Optimization task.

		Returns:
			Dict[str, np.ndarray]:
			* vMin: Mininal velocity.
			* vMax: Maximal velocity.
			* V: Initial velocity of particle.
			* flag: Refresh gap counter.
		"""
		return {'vMin': fullArray(self.vMin, task.D), 'vMax': fullArray(self.vMax, task.D), 'V': np.full([self.NP, task.D], 0.0), 'flag': np.full(self.NP, 0), 'Pc': np.asarray([.05 + .45 * (np.exp(10 * (i - 1) / (self.NP - 1)) - 1) / (np.exp(10) - 1) for i in range(self.NP)])}

	def generatePbestCL(self, i, Pc, pbs, fpbs):
		r"""Generate new personal best position for learning.

		Args:
			i (int): Current particle.
			Pc (float): Learning probability.
			pbs (numpy.ndarray): Personal best positions for population.
			fpbs (numpy.ndarray): Personal best positions function/fitness values for persolan best position.

		Returns:
			numpy.ndarray: Personal best for learning.
		"""
		pbest = []
		for j in range(len(pbs[i])):
			if self.rand() > Pc: pbest.append(pbs[i, j])
			else:
				r1, r2 = int(self.rand() * len(pbs)), int(self.rand() * len(pbs))
				if fpbs[r1] < fpbs[r2]: pbest.append(pbs[r1, j])
				else: pbest.append(pbs[r2, j])
		return np.asarray(pbest)

	def updateVelocityCL(self, V, p, pb, w, vMin, vMax, task, **kwargs):
		r"""Update particle velocity.

		Args:
			V (numpy.ndarray): Current velocity of particle.
			p (numpy.ndarray): Current position of particle.
			pb (numpy.ndarray): Personal best position of particle.
			w (numpy.ndarray): Weights for velocity adjustment.
			vMin (numpy.ndarray): Minimal velocity allowed.
			vMax (numpy.ndarray): Maxmimal velocity allowed.
			task (Task): Optimization task.
			kwargs: Additional arguments.

		Returns:
			numpy.ndarray: Updated velocity of particle.
		"""
		return self.Repair(w * V + self.C * self.rand(task.D) * (pb - p), vMin, vMax)

	def runIteration(self, task, pop, fpop, xb, fxb, popb, fpopb, vMin, vMax, V, flag, Pc, **dparams):
		r"""Core function of algorithm.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current populations.
			fpop (numpy.ndarray): Current population fitness/function values.
			xb (numpy.ndarray): Current best particle.
			fxb (float): Current best particle fitness/function value.
			popb (numpy.ndarray): Particles best position.
			fpopb (numpy.ndarray): Particles best positions fitness/function values.
			vMin (numpy.ndarray): Minimal velocity.
			vMax (numpy.ndarray): Maximal velocity.
			V (numpy.ndarray): Velocity of particles.
			flag (numpy.ndarray): Refresh rate counter.
			Pc (numpy.ndarray): Learning rate.
			**dparams (Dict[str, Any]): Additional function arguments.

		Returns:
			Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
			1. New population.
			2. New population fitness/function values.
			3. New global best position.
			4. New global best positions function/fitness value.
			5. Additional arguments:
				* popb: Particles best population.
				* fpopb: Particles best positions function/fitness value.
				* vMin: Minimal velocity.
				* vMax: Maximal velocity.
				* V: Initial velocity of particle.
				* flag: Refresh gap counter.
				* Pc: Learning rate.

		See Also:
			* :class:`NiaPy.algorithms.basic.ParticleSwarmAlgorithm.runIteration`
		"""
		w = self.w0 * (self.w0 - self.w1) * task.Iters / task.nGEN
		for i in range(len(pop)):
			if flag[i] >= self.m:
				V[i] = self.updateVelocity(V[i], pop[i], popb[i], xb, 1, vMin, vMax, task)
				pop[i] = task.repair(pop[i] + V[i], rnd=self.Rand)
				fpop[i] = task.eval(pop[i])
				if fpop[i] < fpopb[i]:
					popb[i], fpopb[i] = pop[i].copy(), fpop[i]
					if fpop[i] < fxb: xb, fxb = pop[i].copy(), fpop[i]
				flag[i] = 0
			pbest = self.generatePbestCL(i, Pc[i], popb, fpopb)
			V[i] = self.updateVelocityCL(V[i], pop[i], pbest, w, vMin, vMax, task)
			pop[i] = pop[i] + V[i]
			if not ((pop[i] < task.Lower).any() or (pop[i] > task.Upper).any()):
				fpop[i] = task.eval(pop[i])
				if fpop[i] < fpopb[i]:
					popb[i], fpopb[i] = pop[i].copy(), fpop[i]
					if fpop[i] < fxb: xb, fxb = pop[i].copy(), fpop[i]
		return pop, fpop, xb, fxb, {'popb': popb, 'fpopb': fpopb, 'vMin': vMin, 'vMax': vMax, 'V': V, 'flag': flag, 'Pc': Pc}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
