# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, arguments-differ, line-too-long, unused-argument, no-self-use, redefined-builtin
import copy

from numpy import nan, asarray, zeros, float, full

from NiaPy.util.utility import objects2array
from NiaPy.algorithms.algorithm import Algorithm, Individual

class Fish(Individual):
	r"""Fish individual class.

	Attributes:
		weight (float): Weight of fish.
		delta_pos (float): TODO.
		delta_cost (float): TODO.
		has_improved (bool): If the fish has improved.

	See Also:
		* :class:`NiaPy.algorithms.algorithm.Individual`
	"""
	def __init__(self, weight, **kwargs):
		r"""Initialize fish individual.

		Args:
			weight (float): Weight of fish.
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Individual`
		"""
		Individual.__init__(self, **kwargs)
		self.weight = weight
		self.delta_pos = nan
		self.delta_cost = nan
		self.has_improved = False

class FishSchoolSearch(Algorithm):
	r"""Implementation of Fish School Search algorithm.

	Algorithm:
		Fish School Search algorithm

	Date:
		2019

	Authors:
		Clodomir Santana Jr, Elliackin Figueredo, Mariana Maceds, Pedro Santos.
		Ported to NiaPy with small changes by Kristian Järvenpää (2018).
		Ported to the NiaPy 2.0 by Klemen Berkovič (2019).

	License:
		MIT

	Reference paper:
		Bastos Filho, Lima Neto, Lins, D. O. Nascimento and P. Lima, “A novel search algorithm based on fish school behavior,” in 2008 IEEE International Conference on Systems, Man and Cybernetics, Oct 2008, pp. 2646–2651.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		SI_init (int): Length of initial individual step.
		SI_final (int): Length of final individual step.
		SV_init (int): Length of initial volatile step.
		SV_final (int): Length of final volatile step.
		min_w (float): Minimum weight of a fish.
		w_scale (float): Maximum weight of a fish.

	See Also:
		* :class:`NiaPy.algorithms.algorithm.Algorithm`
	"""
	Name = ['FSS', 'FishSchoolSearch']

	@staticmethod
	def typeParameters():
		# TODO
		return {'school_size': lambda x: False, 'SI_final': lambda x: False}

	def setParameters(self, NP=25, SI_init=3, SI_final=10, SV_init=3, SV_final=13, min_w=0.3, w_scale=0.7, **ukwargs):
		r"""Set core arguments of FishSchoolSearch algorithm.

		Arguments:
			NP (Optional[int]): Number of fishes in school.
			SI_init (Optional[int]): Length of initial individual step.
			SI_final (Optional[int]): Length of final individual step.
			SV_init (Optional[int]): Length of initial volatile step.
			SV_final (Optional[int]): Length of final volatile step.
			min_w (Optional[float]): Minimum weight of a fish.
			w_scale (Optional[float]): Maximum weight of a fish.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.step_individual_init = SI_init
		self.step_individual_final = SI_final
		self.step_volitive_init = SV_init
		self.step_volitive_final = SV_final
		self.min_w = min_w
		self.w_scale = w_scale

	def generate_uniform_coordinates(self, task):
		r"""Return Numpy array with uniform distribution.

		Args:
			task (Task): Optimization task.

		Returns:
			numpy.ndarray: Array with uniform distribution.
		"""
		return asarray([self.uniform(task.Lower, task.Upper, task.D) for _ in range(self.NP)])

	def gen_weight(self):
		r"""Get initial weight for fish.

		Returns:
			float: Weight for fish.
		"""
		return self.w_scale / 2.0

	def init_fish(self, pos, task):
		"""Create a new fish to a given position."""
		return Fish(x=pos, weight=self.gen_weight(), task=task, e=True)

	def init_school(self, task):
		"""Initialize fish school with uniform distribution."""
		curr_step_individual = self.step_individual_init * (task.Upper - task.Lower)
		curr_step_volitive = self.step_volitive_init * (task.Upper - task.Lower)
		curr_weight_school = 0.0
		prev_weight_school = 0.0
		school = []
		positions = self.generate_uniform_coordinates(task)
		for idx in range(self.NP):
			fish = self.init_fish(positions[idx], task)
			school.append(fish)
			curr_weight_school += fish.weight
		prev_weight_school = curr_weight_school
		return curr_step_individual, curr_step_volitive, curr_weight_school, prev_weight_school, objects2array(school)

	def max_delta_cost(self, school):
		"""Find maximum delta cost - return 0 if none of the fishes moved."""
		max_ = 0
		for fish in school:
			if max_ < fish.delta_cost: max_ = fish.delta_cost
		return max_

	def total_school_weight(self, school, prev_weight_school, curr_weight_school):
		"""Calculate and update current weight of fish school."""
		prev_weight_school = curr_weight_school
		curr_weight_school = sum(fish.weight for fish in school)
		return prev_weight_school, curr_weight_school

	def calculate_barycenter(self, school, task):
		"""Calculate barycenter of fish school."""
		barycenter = zeros((task.D,), dtype=float)
		density = 0.0
		for fish in school:
			density += fish.weight
			for dim in range(task.D): barycenter[dim] += (fish.x[dim] * fish.weight)
		for dim in range(task.D): barycenter[dim] = barycenter[dim] / density
		return barycenter

	def update_steps(self, task):
		"""Update step length for individual and volatile steps."""
		curr_step_individual = full(task.D, self.step_individual_init - task.Iters * float(self.step_individual_init - self.step_individual_final) / task.nGEN)
		curr_step_volitive = full(task.D, self.step_volitive_init - task.Iters * float(self.step_volitive_init - self.step_volitive_final) / task.nGEN)
		return curr_step_individual, curr_step_volitive

	def update_best_fish(self, school, best_fish):
		"""Find and update current best fish."""
		if best_fish is None: best_fish = copy.copy(school[0])
		for fish in school:
			if best_fish.f > fish.f: best_fish = copy.copy(fish)
		return best_fish

	def feeding(self, school):
		r"""Feed all fishes.

		Args:
			school (numpy.ndarray[Fish]): Current school fish population.

		Returns:
			numpy.ndarray[Fish]: New school fish population.
		"""
		for fish in school:
			if self.max_delta_cost(school): fish.weight = fish.weight + (fish.delta_cost / self.max_delta_cost(school))
			if fish.weight > self.w_scale: fish.weight = self.w_scale
			elif fish.weight < self.min_w: fish.weight = self.min_w
		return school

	def individual_movement(self, school, curr_step_individual, task):
		r"""Perform individual movement for each fish.

		Args:
			school (numpy.ndarray): School fish population.
			curr_step_individual (numpy.ndarray[float]): TODO
			task (Task): Optimization task.

		Returns:
			numpy.ndarray[Fish]: New school of fishes.
		"""
		for fish in school:
			new_pos = task.repair(fish.x + (curr_step_individual * self.uniform(-1, 1, task.D)), rnd=self.Rand)
			cost = task.eval(new_pos)
			if cost < fish.f:
				fish.delta_cost = abs(cost - fish.f)
				fish.f = cost
				delta_pos = zeros((task.D,), dtype=float)
				for idx in range(task.D): delta_pos[idx] = new_pos[idx] - fish.x[idx]
				fish.delta_pos = delta_pos
				fish.x = new_pos
			else:
				fish.delta_pos = zeros((task.D,), dtype=float)
				fish.delta_cost = 0
		return school

	def collective_instinctive_movement(self, school, task):
		"""Perform collective instictive movement."""
		cost_eval_enhanced = zeros((task.D,), dtype=float)
		density = sum([f.delta_cost for f in school])
		for fish in school: cost_eval_enhanced += (fish.delta_pos * fish.delta_cost)
		if density != 0: cost_eval_enhanced = cost_eval_enhanced / density
		for fish in school: fish.x = task.repair(fish.x + cost_eval_enhanced, rnd=self.Rand)
		return school

	def collective_volitive_movement(self, school, curr_step_volitive, prev_weight_school, curr_weight_school, task):
		"""Perform collective volitive movement."""
		prev_weight_school, curr_weight_school = self.total_school_weight(school=school, prev_weight_school=prev_weight_school, curr_weight_school=curr_weight_school)
		barycenter = self.calculate_barycenter(school, task)
		for fish in school:
			if curr_weight_school > prev_weight_school: fish.x -= (fish.x - barycenter) * curr_step_volitive * self.uniform(0, 1, task.D)
			else: fish.x += (fish.x - barycenter) * curr_step_volitive * self.uniform(0, 1, task.D)
			fish.evaluate(task, rnd=self.Rand)
		return school

	def initPopulation(self, task):
		curr_step_individual, curr_step_volitive, curr_weight_school, prev_weight_school, school = self.init_school(task)
		return school, asarray([f.f for f in school]), {'curr_step_individual': curr_step_individual, 'curr_step_volitive': curr_step_volitive, 'curr_weight_school': curr_weight_school, 'prev_weight_school': prev_weight_school}

	def runIteration(self, task, school, fschool, best_fish, fxb, curr_step_individual, curr_step_volitive, curr_weight_school, prev_weight_school, **dparams):
		school = self.individual_movement(school, curr_step_individual, task)
		school = self.feeding(school)
		school = self.collective_instinctive_movement(school, task)
		school = self.collective_volitive_movement(school=school, curr_step_volitive=curr_step_volitive, prev_weight_school=prev_weight_school, curr_weight_school=curr_weight_school, task=task)
		curr_step_individual, curr_step_volitive = self.update_steps(task)
		return school, asarray([f.f for f in school]), {'curr_step_individual': curr_step_individual, 'curr_step_volitive': curr_step_volitive, 'curr_weight_school': curr_weight_school, 'prev_weight_school': prev_weight_school}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
