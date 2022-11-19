# encoding=utf8
import copy
import logging
import numpy
import numpy as np

from niapy.algorithms.algorithm import Algorithm, Individual, default_individual_init
from niapy.util import objects_to_array

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['LionOptimizationAlgorithm']


class Lion(Individual):
    r"""Implementation of population individual that is a lion for Lion Optimization Algorithm.

    Algorithm:
        Lion Optimization Algorithm

    Date:
        2021

    Authors:
        Aljoša Mesarec

    License:
        MIT

    Attributes:
        gender (string): Lion gender.
        has_pride (bool): Lion has a pride.
        pride (int): Lion's pride id.
        hunting_group (int): Lion's hunting group.
        current_x (numpy.ndarray): Lion's current position
        current_f (float): Lion's current fitness
        previous_iter_best_f (float): Lion's fitness at end of previous iteration.
        has_improved (bool): Lion has improved fitness since last iteration.

    See Also:
        * :class:`niapy.algorithms.Individual`

    """

    def __init__(self, gender="m", has_pride=False, pride=-1, hunting_group=0, has_improved=True, **kwargs):
        r"""Initialize the Lion.

        Args:
            gender (Optional[string]): Lion's gender.
            has_pride (Optional[bool]): Lion has a pride.
            pride (Optional[int]): Lion's pride id.
            hunting_group (Optional[int]): Lion's hunting group id.
            has_improved (Optional[bool]): Lion has improved fitness since last iteration.

        See Also:
            * :func:`niapy.algorithms.Individual.__init__`

        """
        super().__init__(**kwargs)
        self.gender = gender
        self.has_pride = has_pride
        self.pride = pride
        self.hunting_group = hunting_group
        self.current_x = np.copy(self.x)
        self.current_f = self.f
        self.previous_iter_best_f = self.f
        self.has_improved = has_improved


class LionOptimizationAlgorithm(Algorithm):
    r"""Implementation of lion optimization algorithm.

    Algorithm:
        Lion Optimization algorithm

    Date:
        2021

    Authors:
        Aljoša Mesarec

    License:
        MIT

    Reference URL:
        https://doi.org/10.1016/j.jcde.2015.06.003

    Reference paper:
        Yazdani, Maziar, Jolai, Fariborz. Lion Optimization Algorithm (LOA): A nature-inspired metaheuristic algorithm. Journal of Computational Design and Engineering, Volume 3, Issue 1, Pages 24-36. 2016.

    Attributes:
        Name (List[str]): List of strings representing name of the algorithm.
        population_size (Optional[int]): Population size :math:`\in [1, \infty)`.
        nomad_ratio (Optional[float]): Ratio of nomad lions :math:`\in [0, 1]`.
        num_of_prides = Number of prides :math:`\in [1, \infty)`.
        female_ratio = Ratio of female lions in prides :math:`\in [0, 1]`.
        roaming_factor = Roaming factor :math:`\in [0, 1]`.
        mating_factor = Mating factor :math:`\in [0, 1]`.
        mutation_factor = Mutation factor :math:`\in [0, 1]`.
        immigration_factor = Immigration factor :math:`\in [0, 1]`.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['LionOptimizationAlgorithm', 'LOA']

    @staticmethod
    def info():
        r"""Get information about algorithm.

        Returns:
            str: Algorithm information

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r'''Yazdani, Maziar, Jolai, Fariborz. Lion Optimization Algorithm (LOA): A nature-inspired metaheuristic algorithm. Journal of Computational Design and Engineering, Volume 3, Issue 1, Pages 24-36. 2016.'''

    def __init__(self, population_size=50, nomad_ratio=0.2, num_of_prides=5, female_ratio=0.8, roaming_factor=0.2, mating_factor=0.3, mutation_factor=0.2, immigration_factor=0.4, *args, **kwargs):
        r"""Initialize LionOptimizationAlgorithm.

        Args:
            population_size (Optional[int]): Population size :math:`\in [1, \infty)`.
            nomad_ratio (Optional[float]): Ratio of nomad lions :math:`\in [0, 1]`.
            num_of_prides = Number of prides :math:`\in [1, \infty)`.
            female_ratio = Ratio of female lions in prides :math:`\in [0, 1]`.
            roaming_factor = Roaming factor :math:`\in [0, 1]`.
            mating_factor = Mating factor :math:`\in [0, 1]`.
            mutation_factor = Mutation factor :math:`\in [0, 1]`.
            immigration_factor = Immigration factor :math:`\in [0, 1]`.


        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, individual_type=kwargs.pop('individual_type', Lion),
                         initialization_function=kwargs.pop('initialization_function', default_individual_init), *args, **kwargs)
        self.nomad_ratio = nomad_ratio
        self.num_of_prides = num_of_prides
        self.female_ratio = female_ratio
        self.roaming_factor = roaming_factor
        self.mating_factor = mating_factor
        self.mutation_factor = mutation_factor
        self.immigration_factor = immigration_factor

    def set_parameters(self, population_size=50, nomad_ratio=0.2, num_of_prides=5, female_ratio=0.8, roaming_factor=0.2, mating_factor=0.3, mutation_factor=0.2, immigration_factor=0.4, **kwargs):
        r"""Set the arguments of an algorithm.

        Args:
            population_size (Optional[int]): Population size :math:`\in [1, \infty)`.
            nomad_ratio (Optional[float]): Ratio of nomad lions :math:`\in [0, 1]`.
            num_of_prides = Number of prides :math:`\in [1, \infty)`.
            female_ratio = Ratio of female lions in prides :math:`\in [0, 1]`.
            roaming_factor = Roaming factor :math:`\in [0, 1]`.
            mating_factor = Mating factor :math:`\in [0, 1]`.
            mutation_factor = Mutation factor :math:`\in [0, 1]`.
            immigration_factor = Immigration factor :math:`\in [0, 1]`.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, individual_type=Lion,
                               initialization_function=kwargs.pop('initialization_function', default_individual_init()), **kwargs)
        self.nomad_ratio = nomad_ratio
        self.num_of_prides = num_of_prides
        self.female_ratio = female_ratio
        self.roaming_factor = roaming_factor
        self.mating_factor = mating_factor
        self.mutation_factor = mutation_factor
        self.immigration_factor = immigration_factor

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm Parameters.

        """
        d = super().get_parameters()
        d.update({
            'nomad_ratio': self.nomad_ratio,
            'num_of_prides': self.num_of_prides,
            'female_ratio': self.female_ratio,
            'roaming_factor': self.roaming_factor,
            'mating_factor': self.mating_factor,
            'mutation_factor': self.mutation_factor,
            'immigration_factor': self.immigration_factor
        })
        return d

    def init_population(self, task):
        r"""Initialize starting population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray[Lion], numpy.ndarray[float], Dict[str, Any]]:
                1. Initialized population of lions.
                2. Initialized populations function/fitness values.
                3. Additional arguments:
                    * pride_size (numpy.ndarray): Pride and nomad sizes.
                    * gender_distribution (numpy.ndarray): Pride and nomad gender distributions.

        """
        pop, fpop, d = super().init_population(task)
        pop, d = self.init_population_data(pop, d)
        return pop, fpop, d

    def init_population_data(self, pop, d):
        r"""Initialize data of starting population.

        Args:
            pop (numpy.ndarray[Lion]: Starting lion population
            d (Dict[str, Any]): Additional arguments

        Returns:
            Tuple[numpy.ndarray[Lion], Dict[str, Any]]:
                1. Initialized population of lions.
                2. Additional arguments:
                    * pride_size (numpy.ndarray): Pride and nomad sizes.
                    * gender_distribution (numpy.ndarray): Pride and nomad gender distributions.

        """
        nomad_size = round(self.nomad_ratio * self.population_size)

        # Creating array of pride sizes.
        pride_size = np.zeros(self.num_of_prides + 1, dtype=int)
        pride_size[-1] = nomad_size
        remaining_lions = self.population_size - nomad_size
        while remaining_lions > 0:
            if remaining_lions >= self.num_of_prides:
                pride_size[:self.num_of_prides] += 1
            else:
                pride_size[:remaining_lions] += 1
            remaining_lions -= self.num_of_prides

        # Setting gender and pride id of pride members.
        index_counter = 0
        for i in range(self.num_of_prides):
            curr_pride_size = pride_size[i]
            num_of_females = round(self.female_ratio * curr_pride_size)
            for lion in pop[index_counter:index_counter + curr_pride_size]:
                lion.has_pride = True
                lion.pride = i
                if num_of_females > 0:
                    lion.gender = "f"
                    num_of_females -= 1
            index_counter += curr_pride_size

        # Setting gender of nomads
        num_of_females = round((1 - self.female_ratio) * pride_size[-1])
        for lion in pop[index_counter:index_counter + num_of_females]:
            lion.gender = "f"

        # Creating array of pride gender quantities
        gender_distribution = np.zeros((self.num_of_prides + 1, 2), dtype=int)
        index_counter = 0
        for i in range(self.num_of_prides):
            curr_pride_size = pride_size[i]
            for lion in pop[index_counter:index_counter + curr_pride_size]:
                if lion.gender == "f":
                    gender_distribution[i][0] += 1
                elif lion.gender == "m":
                    gender_distribution[i][1] += 1
            index_counter += curr_pride_size
        # Creating array of nomad gender quantities
        for lion in pop[index_counter:]:
            if lion.gender == "f":
                gender_distribution[self.num_of_prides][0] += 1
            elif lion.gender == "m":
                gender_distribution[self.num_of_prides][1] += 1

        # Updating params
        d.update({'pride_size': pride_size, 'gender_distribution': gender_distribution})
        return pop, d

    def hunting(self, population, pride_size, task):
        r"""Pride female hunters go hunting.

        Args:
            population (numpy.ndarray[Lion]): Lion population.
            pride_size (numpy.ndarray[int]): Pride and nomad sizes.
            task (Task): Optimization task.

        Returns:
            population (numpy.ndarray[Lion]): Lion population that finished with hunting.

        """
        num_of_prides = len(pride_size) - 1
        index_counter_pride = 0
        for pride_i in range(num_of_prides):
            curr_pride_size = pride_size[pride_i]
            prey_x = np.zeros(task.dimension, dtype=float)
            num_of_hunters = 0
            hunting_group_fitness = np.zeros(4)
            for lion in population[index_counter_pride:index_counter_pride + curr_pride_size]:
                if lion.gender == "f":
                    lion.hunting_group = self.integers(0, 4)
                    hunting_group_fitness[lion.hunting_group] += lion.current_f
                if lion.hunting_group != 0:
                    prey_x += lion.current_x
                    num_of_hunters += 1

            # Group with highest fitness becomes center group, the rest become left and right groups
            sorted_hunting_group_indices = np.argsort(hunting_group_fitness[1:])
            right_group, left_group, center_group = sorted_hunting_group_indices + 1

            # Prey's position is average position of hunters.
            if num_of_hunters != 0:
                prey_x /= num_of_hunters

            # Check if prey's new position is in limits.
            prey_x = task.repair(prey_x)

            # Calculate new positions of hunters with "Opposition-Based Learning method".
            for lion in population[index_counter_pride:index_counter_pride + curr_pride_size]:
                if lion.hunting_group == left_group or lion.hunting_group == right_group:
                    for i in range(task.dimension):
                        if (2 * prey_x[i] - lion.current_x[i]) < prey_x[i]:
                            lion.current_x[i] = self.uniform((2 * prey_x[i] - lion.current_x[i]), prey_x[i])
                        elif (2 * prey_x[i] - lion.current_x[i]) > prey_x[i]:
                            lion.current_x[i] = self.uniform(prey_x[i], (2 * prey_x[i] - lion.current_x[i]))
                if lion.hunting_group == center_group:
                    for i in range(task.dimension):
                        if lion.current_x[i] < prey_x[i]:
                            lion.current_x[i] = self.uniform(lion.current_x[i], prey_x[i])
                        elif lion.current_x[i] > prey_x[i]:
                            lion.current_x[i] = self.uniform(prey_x[i], lion.current_x[i])

                if lion.hunting_group != 0:
                    # Check if lion's new position is in limits.
                    lion.current_x = task.repair(lion.current_x)
                    lion.current_f = task.eval(lion.current_x)
                    # If hunter's new fitness is better then change prey's position.
                    if lion.current_f < lion.f:
                        lion.x = np.copy(lion.current_x)
                        lion.f = lion.current_f
                        percentage_of_improvement = 1 - lion.f / lion.previous_iter_best_f
                        prey_x = prey_x + self.random() * percentage_of_improvement * (prey_x - lion.current_x)
                        # Check if prey's new position is in limits.
                        prey_x = task.repair(prey_x)

            index_counter_pride += curr_pride_size

        return population

    def move_to_safe_place(self, population, pride_size, task):
        r"""Female pride lions move towards position with good fitness.

        Args:
            population (numpy.ndarray[Lion]): Lion population.
            pride_size (numpy.ndarray[int]): Pride and nomad sizes.
            task (Task): Optimization task.

        Returns:
            population (numpy.ndarray[Lion]): Lion population that finished with moving to safe place.

        """
        num_of_prides = len(pride_size) - 1
        index_counter_pride = 0
        for pride_i in range(num_of_prides):
            curr_pride_size = pride_size[pride_i]
            num_of_improvements = 0
            pride_territory = []
            for lion in population[index_counter_pride:index_counter_pride + curr_pride_size]:
                lion_copy = copy.deepcopy(lion)
                pride_territory = np.append(pride_territory, objects_to_array([lion_copy]))
                if lion.has_improved:
                    num_of_improvements += 1
            # Tournament selection to select places in territory if there's more than 2 places
            if len(pride_territory) > 1:
                tournament_size = max(2, int(np.ceil(num_of_improvements / 2)))
                tournament_selections = self.rng.choice(pride_territory, tournament_size, replace=False)
                tournament_winner = tournament_selections[0].x.copy()
                tournament_min_f = tournament_selections[0].f
                for candidate in tournament_selections[1:]:
                    if candidate.f < tournament_min_f:
                        tournament_min_f = candidate.f
                        tournament_winner = candidate.x.copy()
            else:
                tournament_winner = pride_territory[0].x.copy()
                tournament_min_f = pride_territory[0].f

            # Move female non-hunters
            for lion in population[index_counter_pride:index_counter_pride + curr_pride_size]:
                if lion.gender == "f" and lion.hunting_group == 0:
                    # Get vector r_one.
                    r_one = tournament_winner.copy()
                    r_one -= lion.x
                    # Get vector 2_two with Gram-Schmidt process.
                    if np.linalg.norm((r_one).T) == 0:
                        # If r_one vector is 0 then Gram-Schmidt process return wrong values
                        r_two = np.zeros(len(r_one.T))
                        rand_index = self.integers(0, len(r_one.T))
                        r_two[rand_index] = 1
                    else:
                        # Gram-Schmidt process to find orthogonal vector r_two.
                        random_vec = self.standard_normal(len(r_one))
                        r_two = random_vec - ((r_one.T).dot(random_vec)) / ((r_one.T).dot(r_one)) * r_one
                    # Calculate other variables and new lion's position
                    d = np.linalg.norm(r_one) / np.linalg.norm(task.upper[0] - task.lower[0])
                    rnd_num = self.random()
                    rnd_num_u = self.uniform(-1, 1)
                    angle = self.uniform(-np.pi / 6, np.pi / 6)
                    lion.current_x += 2 * d * rnd_num * r_one + rnd_num_u * np.tan(angle) * d * r_two
                    # Check if lion's current position is in limits.
                    lion.current_x = task.repair(lion.current_x)
                    lion.current_f = task.eval(lion.current_x)
                    # If lion's position has improved update best position and fitness
                    if lion.current_f < lion.f:
                        lion.x = np.copy(lion.current_x)
                        lion.f = lion.current_f
            index_counter_pride += curr_pride_size

        return population

    def roaming(self, population, pride_size, task):
        r"""Male lions move towards new position.

        Args:
            population (numpy.ndarray[Lion]): Lion population.
            pride_size (numpy.ndarray[int]): Pride and nomad sizes.
            task (Task): Optimization task.

        Returns:
            population (numpy.ndarray[Lion]): Lion population that finished with roaming.

        """
        num_of_prides = len(pride_size) - 1
        index_counter_pride = 0
        # Pride lions roam.
        for pride_i in range(num_of_prides):
            curr_pride_size = pride_size[pride_i]
            for lion in population[index_counter_pride:index_counter_pride + curr_pride_size]:
                if lion.gender == "m":
                    # Select all lions in pride.
                    pride_lions = []
                    for p_l in population[index_counter_pride:index_counter_pride + curr_pride_size]:
                        lion_copy = copy.deepcopy(p_l)
                        pride_lions = np.append(pride_lions, objects_to_array([lion_copy]))
                    # Select random lions, their amount is based on roaming factor.
                    num_of_selected_lions = round(len(pride_lions) * self.roaming_factor)
                    selected_lions = self.rng.choice(pride_lions, num_of_selected_lions, replace=False)
                    # Move towards territories of selected lions
                    for selected_lion in selected_lions:
                        d = np.linalg.norm(selected_lion.x - lion.x) / np.linalg.norm(task.upper[0] - task.lower[0])
                        x = self.uniform(0, 2 * d)
                        angle = self.uniform(-np.pi / 6, np.pi / 6)
                        lion.current_x += x * d * np.tan(angle)
                        # Check if lion's new position is in limits.
                        lion.current_x = task.repair(lion.current_x)
                        lion.current_f = task.eval(lion.current_x)
                        # Update best position/fitness if lion's best position is improved
                        if lion.current_f < lion.f:
                            lion.x = np.copy(lion.current_x)
                            lion.f = lion.current_f

            index_counter_pride += curr_pride_size

        # Nomad lions roam.
        nomad_size = pride_size[-1]
        for lion in population[len(population) - nomad_size:]:
            best_nomad_fitness = np.min([c_l.current_f for c_l in population[len(population) - nomad_size:]])
            roaming_probability = 0.1 + np.minimum(0.5, (lion.current_f - best_nomad_fitness) / best_nomad_fitness)
            # If roaming threshold is met, move lion to a random new position.
            if self.random() <= roaming_probability:
                lion.current_x = self.uniform(task.lower, task.upper, task.dimension)
                lion.current_f = task.eval(lion.current_x)
            # Update best position/fitness if lion's best position is improved
            if lion.current_f < lion.f:
                lion.x = np.copy(lion.current_x)
                lion.f = lion.current_f

        return population

    def mating(self, population, pride_size, gender_distribution, task):
        r"""Female lions mate with male lions to produce offspring.

        Args:
            population (numpy.ndarray[Lion]): Lion population.
            pride_size (numpy.ndarray[int]): Pride and nomad sizes.
            gender_distribution (numpy.ndarray[int]): Pride and nomad gender distribution.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray[Lion], numpy.ndarray[int]):
                1. Lion population that finished with mating.
                2. Pride and nomad excess gender quantities.

        """
        added_cubs = []
        excess_lion_gender_quantities = np.zeros((self.num_of_prides + 1, 2), dtype=int)
        num_of_prides = len(pride_size) - 1
        # Copy of all pride lions.
        pride_lions = []
        index_counter_pride = 0
        for pride_i in range(num_of_prides):
            curr_pride_size = pride_size[pride_i]
            for lion in population[index_counter_pride:index_counter_pride + curr_pride_size]:
                lion_copy = copy.deepcopy(lion)
                pride_lions = np.append(pride_lions, objects_to_array([lion_copy]))
            index_counter_pride += curr_pride_size

        # Copy of all nomad lions.
        nomad_lions = []
        nomad_size = pride_size[-1]
        for lion in population[len(population) - nomad_size]:
            lion_copy = copy.deepcopy(lion)
            nomad_lions = np.append(nomad_lions, objects_to_array([lion_copy]))

        # Prides mating.
        index_counter_pride = 0
        for pride_i in range(num_of_prides):
            curr_pride_size = pride_size[pride_i]
            num_of_males = gender_distribution[pride_i][1]
            # If there's at least 1 male, proceed
            if num_of_males != 0:
                # Array of males.
                males = []
                for lion in population[index_counter_pride:index_counter_pride + curr_pride_size]:
                    if lion.gender == "m":
                        lion_copy = copy.deepcopy(lion)
                        males = np.append(males, objects_to_array([lion_copy]))
                # Mate all females with a mating probability.
                for lion in population[index_counter_pride:index_counter_pride + curr_pride_size]:
                    if lion.gender == "f" and self.random() < self.mating_factor:
                        # Choose males that will mate.
                        num_of_mating_males = self.integers(1, num_of_males)
                        mating_males = self.rng.choice(males, num_of_mating_males, replace=False)

                        beta = self.normal(0.5, 0.1)

                        # Total position x of mating males. Needed for calculating average.
                        mating_males_x_sum = np.zeros(task.dimension)
                        for mating_male in mating_males:
                            mating_males_x_sum = np.add(mating_males_x_sum, mating_male.x)
                        # Calculate position x for offsprings.
                        offspring_one_position = beta * lion.x + ((1 - beta) * mating_males_x_sum / num_of_mating_males)
                        offspring_two_position = (1 - beta) * lion.x + (beta * mating_males_x_sum / num_of_mating_males)

                        # Mutation of the genes with mutation probability.
                        for i in range(task.dimension):
                            if self.random() < self.mutation_factor:
                                offspring_one_position[i] = self.uniform(task.lower[i], task.upper[i], 1)
                            if self.random() < self.mutation_factor:
                                offspring_two_position[i] = self.uniform(task.lower[i], task.upper[i], 1)
                        # Create offspring Lion objects
                        offspring_one = copy.deepcopy(lion)
                        offspring_two = copy.deepcopy(lion)
                        offspring_one.has_pride = True
                        offspring_two.has_pride = True
                        offspring_one.pride = pride_i
                        offspring_two.pride = pride_i
                        offspring_one.hunting_group = 0
                        offspring_two.hunting_group = 0
                        offspring_one.has_improved = True
                        offspring_two.has_improved = True
                        # Randomly assign genders to offsprings.
                        if self.random() < 0.5:
                            offspring_one.gender = "m"
                            offspring_two.gender = "f"
                        else:
                            offspring_one.gender = "f"
                            offspring_two.gender = "m"
                        offspring_one.x = offspring_one_position
                        offspring_two.x = offspring_two_position
                        # Check if offspring's position is in limits.
                        offspring_one.evaluate(task)
                        offspring_two.evaluate(task)
                        # Assign other offspring's values.
                        offspring_one.current_x = np.copy(offspring_one.x)
                        offspring_two.current_x = np.copy(offspring_two.x)
                        offspring_one.current_f = offspring_one.f
                        offspring_two.current_f = offspring_two.f
                        offspring_one.previous_iter_best_f = offspring_one.f + 1
                        offspring_two.previous_iter_best_f = offspring_two.f + 1
                        # Add offspring to array of added cubs.
                        added_cubs = np.append(added_cubs, objects_to_array([offspring_one]))
                        added_cubs = np.append(added_cubs, objects_to_array([offspring_two]))
                        excess_lion_gender_quantities[pride_i][0] += 1
                        excess_lion_gender_quantities[pride_i][1] += 1
                        pride_size[pride_i] += 2
                        gender_distribution[pride_i][0] += 1
                        gender_distribution[pride_i][1] += 1
            index_counter_pride += curr_pride_size

        # Nomads mating.
        nomad_size = pride_size[-1]
        num_of_males = gender_distribution[pride_i][1]
        # If there's at least one male nomad, proceed
        if num_of_males != 0:
            # Create array of males that can mate.
            males = []
            for lion in population[len(population) - nomad_size:]:
                if lion.gender == "m":
                    lion_copy = copy.deepcopy(lion)
                    males = np.append(males, objects_to_array([lion_copy]))
            # Mate all females with mating probability.
            for lion in population[len(population) - nomad_size:]:
                if lion.gender == "f" and self.random() < self.mating_factor:
                    # Choose one male that will mate.
                    mating_male = self.rng.choice(males)

                    beta = self.normal(0.5, 0.1)

                    # Calculate x for offsprings.
                    offspring_one_position = beta * lion.x + ((1 - beta) * mating_male.x)
                    offspring_two_position = (1 - beta) * lion.x + beta * mating_male.x

                    # Mutation of the genes with mutation probability.
                    for i in range(task.dimension):
                        if self.random() < self.mutation_factor:
                            offspring_one_position[i] = self.uniform(task.lower[i], task.upper[i])
                        if self.random() < self.mutation_factor:
                            offspring_two_position[i] = self.uniform(task.lower[i], task.upper[i])

                    # Create offspring Lion objects.
                    offspring_one = copy.deepcopy(lion)
                    offspring_two = copy.deepcopy(lion)
                    offspring_one.has_pride = False
                    offspring_two.has_pride = False
                    offspring_one.pride = -1
                    offspring_two.pride = -1
                    offspring_one.hunting_group = 0
                    offspring_two.hunting_group = 0
                    offspring_one.has_improved = True
                    offspring_two.has_improved = True
                    # Randomly assign genders to offsprings.
                    if self.random() < 0.5:
                        offspring_one.gender = "m"
                        offspring_two.gender = "f"
                    else:
                        offspring_one.gender = "f"
                        offspring_two.gender = "m"

                    offspring_one.x = offspring_one_position
                    offspring_two.x = offspring_two_position
                    # Check if offspring's position is in limits.
                    offspring_one.evaluate(task)
                    offspring_two.evaluate(task)
                    # Assign other offspring's values.
                    offspring_one.current_x = np.copy(offspring_one.x)
                    offspring_two.current_x = np.copy(offspring_two.x)
                    offspring_one.current_f = offspring_one.f
                    offspring_two.current_f = offspring_two.f
                    offspring_one.previous_iter_best_f = offspring_one.f + 1
                    offspring_two.previous_iter_best_f = offspring_two.f + 1
                    # Add offspring to array of added cubs
                    added_cubs = np.append(added_cubs, objects_to_array([offspring_one]))
                    added_cubs = np.append(added_cubs, objects_to_array([offspring_two]))
                    excess_lion_gender_quantities[-1][0] += 1
                    excess_lion_gender_quantities[-1][1] += 1
                    pride_size[-1] += 2
                    gender_distribution[-1][0] += 1
                    gender_distribution[-1][1] += 1

        # Add pride originals and cubs to same population.
        new_population = []
        original_index_counter_pride = 0
        cub_index_counter_pride = 0
        for pride_i in range(num_of_prides):
            # Append original pride lion.
            curr_original_pride_size = pride_size[pride_i] - excess_lion_gender_quantities[pride_i][0] - excess_lion_gender_quantities[pride_i][1]
            for lion in population[original_index_counter_pride:original_index_counter_pride + curr_original_pride_size]:
                lion_copy = copy.deepcopy(lion)
                new_population = np.append(new_population, objects_to_array([lion_copy]))
            # Append cub pride lions.
            curr_cub_pride_size = excess_lion_gender_quantities[pride_i][0] + excess_lion_gender_quantities[pride_i][1]
            for lion in added_cubs[cub_index_counter_pride:cub_index_counter_pride + curr_cub_pride_size]:
                lion_copy = copy.deepcopy(lion)
                new_population = np.append(new_population, objects_to_array([lion_copy]))

            original_index_counter_pride += curr_original_pride_size
            cub_index_counter_pride += curr_cub_pride_size
        # Add nomad originals and cubs to same population.
        originals_nomad_size = pride_size[-1] - excess_lion_gender_quantities[-1][0] - excess_lion_gender_quantities[-1][1]
        for lion in population[len(population) - originals_nomad_size:]:
            lion_copy = copy.deepcopy(lion)
            new_population = np.append(new_population, objects_to_array([lion_copy]))
        cubs_nomad_size = excess_lion_gender_quantities[-1][0] + excess_lion_gender_quantities[-1][1]
        for lion in added_cubs[len(added_cubs) - cubs_nomad_size:]:
            lion_copy = copy.deepcopy(lion)
            new_population = np.append(new_population, objects_to_array([lion_copy]))

        return new_population, excess_lion_gender_quantities

    def defense(self, population, pride_size, gender_distribution, excess_lion_gender_quantities, task):
        r"""Male lions attack other lions in pride.

        Args:
            population (numpy.ndarray[Lion]): Lion population.
            pride_size (numpy.ndarray[int]): Pride and nomad sizes.
            gender_distribution (numpy.ndarray[int]): Pride and nomad gender distribution.
            excess_lion_gender_quantities (numpy.ndarray[int]): Pride and nomad excess members.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray[Lion], numpy.ndarray[int]):
                1. Lion population that finished with defending.
                2. Pride and nomad excess gender quantities.

        """
        new_nomads = []
        original_pride_lions = []
        num_of_prides = len(pride_size) - 1
        # Pride lions defense
        index_counter_pride = 0
        for pride_i in range(num_of_prides):
            curr_pride_size = pride_size[pride_i]
            num_of_males_to_be_kicked = excess_lion_gender_quantities[pride_i][1]
            males = []
            # Go through pride
            for lion in population[index_counter_pride:index_counter_pride + curr_pride_size]:
                lion_copy = copy.deepcopy(lion)
                if lion.gender == "m":
                    males = np.append(males, objects_to_array([lion_copy]))
                elif lion.gender == "f":
                    original_pride_lions = np.append(original_pride_lions, objects_to_array([lion_copy]))
            # Find males with worst fitness that will be kicked, leave the rest
            males = sorted(males, key=lambda lion: lion.current_f, reverse=True)
            for lion in males:
                lion_copy = copy.deepcopy(lion)
                if num_of_males_to_be_kicked == 0:
                    original_pride_lions = np.append(original_pride_lions, objects_to_array([lion_copy]))
                else:
                    new_nomads = np.append(new_nomads, objects_to_array([lion_copy]))
                    num_of_males_to_be_kicked -= 1
            index_counter_pride += curr_pride_size

        # Create new population after kicking pride lions
        moved_population = []
        # Append original pride lions.
        for lion in original_pride_lions:
            lion_copy = copy.deepcopy(lion)
            moved_population = np.append(moved_population, objects_to_array([lion_copy]))
        # Append original nomads.
        original_nomads_size = pride_size[-1]
        for lion in population[len(population) - original_nomads_size:]:
            lion_copy = copy.deepcopy(lion)
            moved_population = np.append(moved_population, objects_to_array([lion_copy]))
        # Append new nomads.
        for lion in new_nomads:
            lion_copy = copy.deepcopy(lion)

            excess_lion_gender_quantities[lion_copy.pride][1] -= 1
            gender_distribution[lion_copy.pride][1] -= 1
            pride_size[lion_copy.pride] -= 1

            lion_copy.has_pride = False
            lion_copy.pride = -1

            moved_population = np.append(moved_population, objects_to_array([lion_copy]))

            excess_lion_gender_quantities[-1][1] += 1
            gender_distribution[-1][1] += 1
            pride_size[-1] += 1

        # Nomad lions defense.
        nomads_size = pride_size[-1]
        for nomad_lion in moved_population[len(moved_population) - nomads_size:]:
            nomad_lion_has_won = False
            # Create binary template - which prides a nomad lion will attack.
            pride_index_to_attack = np.zeros(num_of_prides, dtype=int)
            for i in range(num_of_prides):
                if self.random() < 0.5:
                    pride_index_to_attack[i] = 1
            # Nomad attacks prides based on binary template.
            index_counter_pride = 0
            for pride_i in range(num_of_prides):
                curr_pride_size = pride_size[pride_i]
                if pride_index_to_attack[pride_i] == 1:
                    # Attack all male lions.
                    for pride_lion in moved_population[index_counter_pride:index_counter_pride + curr_pride_size]:
                        if lion.gender == "m":
                            # Swap nomad and pride lion if nomad has better fitness.
                            if nomad_lion.current_f < pride_lion.current_f:
                                copy_nomad_lion = copy.deepcopy(nomad_lion)
                                copy_pride_lion = copy.deepcopy(pride_lion)

                                pride_lion = copy_nomad_lion
                                pride_lion.has_pride = True
                                pride_lion.pride = pride_i

                                nomad_lion = copy_pride_lion
                                nomad_lion.has_pride = False
                                nomad_lion.pride = -1
                                # If nomad lion won the attack, we continue with next nomad.
                                nomad_lion_has_won = True
                                break
                    if nomad_lion_has_won:
                        break
                index_counter_pride += curr_pride_size

        return moved_population, excess_lion_gender_quantities

    def migration(self, population, pride_size, gender_distribution, excess_lion_gender_quantities, task):
        r"""Female lions randomly become nomad.

        Args:
            population (numpy.ndarray[Lion]): Lion population.
            pride_size (numpy.ndarray[int]): Pride and nomad sizes.
            gender_distribution (numpy.ndarray[int]): Pride and nomad gender distribution.
            excess_lion_gender_quantities (numpy.ndarray[int]): Pride and nomad excess members.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray[Lion], numpy.ndarray[int]):
                1. Lion population that finished with migration.
                2. Pride and nomad excess gender quantities.

        """
        new_nomads = []
        original_pride_lions = []

        num_of_prides = len(pride_size) - 1
        # Pride females migration
        index_counter_pride = 0
        for pride_i in range(num_of_prides):
            curr_pride_size = pride_size[pride_i]
            num_of_females = gender_distribution[pride_i][0]
            num_of_excess_females = excess_lion_gender_quantities[pride_i][0]
            num_of_females_to_migrate = num_of_excess_females + round(((num_of_females - num_of_excess_females) * self.immigration_factor))

            females = []
            # Go through pride
            for lion in population[index_counter_pride:index_counter_pride + curr_pride_size]:
                lion_copy = copy.deepcopy(lion)
                if lion.gender == "m":
                    original_pride_lions = np.append(original_pride_lions, objects_to_array([lion_copy]))
                elif lion.gender == "f":
                    females = np.append(females, objects_to_array([lion_copy]))
            # Migrate random females, leave the rest
            females_indices_to_migrate = np.zeros(num_of_females, dtype=int)
            for i in range(num_of_females_to_migrate):
                females_indices_to_migrate[i] = 1
            self.rng.shuffle(females_indices_to_migrate)

            for i, lion in enumerate(females):
                lion_copy = copy.deepcopy(lion)
                if females_indices_to_migrate[i] == 1:
                    new_nomads = np.append(new_nomads, objects_to_array([lion_copy]))
                else:
                    original_pride_lions = np.append(original_pride_lions, objects_to_array([lion_copy]))
            index_counter_pride += curr_pride_size

        # Create new population after migrating pride lions
        moved_population = []
        # Append original pride lions
        for lion in original_pride_lions:
            lion_copy = copy.deepcopy(lion)
            moved_population = np.append(moved_population, objects_to_array([lion_copy]))
        # Append original nomads
        original_nomads_size = pride_size[-1]
        for lion in population[len(population) - original_nomads_size:]:
            lion_copy = copy.deepcopy(lion)
            moved_population = np.append(moved_population, objects_to_array([lion_copy]))
        # Append new nomads
        for lion in new_nomads:
            lion_copy = copy.deepcopy(lion)

            excess_lion_gender_quantities[lion_copy.pride][0] -= 1
            gender_distribution[lion_copy.pride][0] -= 1
            pride_size[lion_copy.pride] -= 1

            lion_copy.has_pride = False
            lion_copy.pride = -1

            moved_population = np.append(moved_population, objects_to_array([lion_copy]))

            excess_lion_gender_quantities[-1][0] += 1
            gender_distribution[-1][0] += 1
            pride_size[-1] += 1

        # Fill up empty female pride spaces with best nomad female lions
        prides_spots_to_be_filled = 0
        for i in range(num_of_prides):
            prides_spots_to_be_filled += np.abs(excess_lion_gender_quantities[i][0])

        nomad_females = []
        nomad_males = []
        original_nomads_size = pride_size[-1]
        for lion in moved_population[len(population) - original_nomads_size:]:
            lion_copy = copy.deepcopy(lion)
            if lion.gender == "f":
                nomad_females = np.append(nomad_females, objects_to_array([lion_copy]))
            elif lion.gender == "m":
                nomad_males = np.append(nomad_males, objects_to_array([lion_copy]))
        nomad_females = sorted(nomad_females, key=lambda lion: lion.current_f, reverse=False)
        nomad_females_to_move = []
        nomad_females_to_keep = []
        counter = prides_spots_to_be_filled
        for lion in nomad_females:
            lion_copy = copy.deepcopy(lion)
            if not counter == 0:
                nomad_females_to_move = np.append(nomad_females_to_move, objects_to_array([lion_copy]))
                counter -= 1
            else:
                nomad_females_to_keep = np.append(nomad_females_to_keep, objects_to_array([lion_copy]))
        self.rng.shuffle(nomad_females_to_move)

        # Append pride lions and moved female nomads
        final_population = []
        index_counter_pride = 0
        index_females_to_move = 0
        for pride_i in range(num_of_prides):
            curr_pride_size = pride_size[pride_i]
            # Append pride lions
            for lion in moved_population[index_counter_pride:index_counter_pride + curr_pride_size]:
                lion_copy = copy.deepcopy(lion)
                final_population = np.append(final_population, objects_to_array([lion_copy]))
            # Append female nomads
            curr_pride_spots_empty = np.abs(excess_lion_gender_quantities[pride_i][0])
            for lion in nomad_females_to_move[index_females_to_move:index_females_to_move + curr_pride_spots_empty]:
                lion_copy = copy.deepcopy(lion)

                excess_lion_gender_quantities[lion_copy.pride][0] -= 1
                gender_distribution[lion_copy.pride][0] -= 1
                pride_size[lion_copy.pride] -= 1

                lion_copy.has_pride = True
                lion_copy.pride = pride_i

                final_population = np.append(final_population, objects_to_array([lion_copy]))

                excess_lion_gender_quantities[pride_i][0] += 1
                gender_distribution[pride_i][0] += 1
                pride_size[pride_i] += 1
            # Increase starting indices for array search
            index_counter_pride += curr_pride_size
            index_females_to_move += curr_pride_spots_empty
        # Append the kept nomad females
        for lion in nomad_females_to_keep:
            lion_copy = copy.deepcopy(lion)
            final_population = np.append(final_population, objects_to_array([lion_copy]))
        # Append nomad males
        for lion in nomad_males:
            lion_copy = copy.deepcopy(lion)
            final_population = np.append(final_population, objects_to_array([lion_copy]))
        return final_population, excess_lion_gender_quantities

    def population_equilibrium(self, population, pride_size, gender_distribution, excess_lion_gender_quantities, task):
        r"""Remove extra nomad lions.

        Args:
            population (numpy.ndarray[Lion]): Lion population.
            pride_size (numpy.ndarray[int]): Pride and nomad sizes.
            gender_distribution (numpy.ndarray[int]): Pride and nomad gender distribution.
            excess_lion_gender_quantities (numpy.ndarray[int]): Pride and nomad excess members.
            task (Task): Optimization task.

        Returns:
            final_population (numpy.ndarray[Lion]): Lion population with removed extra nomads.

        """
        nomad_females = []
        nomad_males = []
        kept_nomads = []
        # Number of males and females that need to be removed.
        num_of_female_nomads_to_remove = excess_lion_gender_quantities[-1][0]
        num_of_male_nomads_to_remove = excess_lion_gender_quantities[-1][1]
        original_nomads_size = pride_size[-1]
        # Get lists of nomad males and females.
        for lion in population[len(population) - original_nomads_size:]:
            lion_copy = copy.deepcopy(lion)
            if lion.gender == "f":
                nomad_females = np.append(nomad_females, objects_to_array([lion_copy]))
            elif lion.gender == "m":
                nomad_males = np.append(nomad_males, objects_to_array([lion_copy]))
        # Sort lists descendingly.
        nomad_males = sorted(nomad_males, key=lambda lion: lion.current_f, reverse=True)
        nomad_females = sorted(nomad_females, key=lambda lion: lion.current_f, reverse=True)
        # Remove extra lions that have bad fitness, keep the rest.
        for lion in nomad_males[num_of_male_nomads_to_remove:]:
            lion_copy = copy.deepcopy(lion)
            kept_nomads = np.append(kept_nomads, objects_to_array([lion_copy]))
        for lion in nomad_females[num_of_female_nomads_to_remove:]:
            lion_copy = copy.deepcopy(lion)
            kept_nomads = np.append(kept_nomads, objects_to_array([lion_copy]))

        # Append pride lions to final population.
        final_population = []
        for lion in population[:len(population) - original_nomads_size]:
            lion_copy = copy.deepcopy(lion)
            final_population = np.append(final_population, objects_to_array([lion_copy]))
        # Append kept nomads to final population.
        for lion in kept_nomads:
            lion_copy = copy.deepcopy(lion)
            final_population = np.append(final_population, objects_to_array([lion_copy]))
        pride_size[-1] -= (num_of_female_nomads_to_remove + num_of_male_nomads_to_remove)
        gender_distribution[-1][0] -= num_of_female_nomads_to_remove
        gender_distribution[-1][1] -= num_of_male_nomads_to_remove

        return final_population

    def data_correction(self, population, pride_size, task):
        r"""Update lion's data if his position has improved since last iteration.

        Args:
            population (numpy.ndarray[Lion]): Lion population.
            pride_size (numpy.ndarray[int]): Pride and nomad sizes.
            task (Task): Optimization task.

        Returns:
            population (numpy.ndarray[Lion]): Lion population with corrected data.

        """
        for lion in population:
            if lion.f < lion.previous_iter_best_f:
                lion.has_improved = True
                lion.previous_iter_best_f = lion.f
            else:
                lion.has_improved = False
        return population

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        pride_size = params.pop('pride_size')
        gender_distribution = params.pop('gender_distribution')
        # Algorithm steps
        lions = self.hunting(population, pride_size, task)
        lions = self.move_to_safe_place(lions, pride_size, task)
        lions = self.roaming(lions, pride_size, task)
        lions, excess_lion_gender_quantities = self.mating(lions, pride_size, gender_distribution, task)
        lions, excess_lion_gender_quantities = self.defense(lions, pride_size, gender_distribution, excess_lion_gender_quantities, task)
        lions, excess_lion_gender_quantities = self.migration(lions, pride_size, gender_distribution, excess_lion_gender_quantities, task)
        lions = self.population_equilibrium(lions, pride_size, gender_distribution, excess_lion_gender_quantities, task)
        lions = self.data_correction(lions, pride_size, task)

        lions_fitness = np.asarray([lion.f for lion in lions])
        best_x, best_fitness = self.get_best(lions, lions_fitness, best_x, best_fitness)
        return lions, lions_fitness, best_x, best_fitness, {'pride_size': pride_size, 'gender_distribution': gender_distribution}
