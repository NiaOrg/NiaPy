# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm, Individual
from niapy.util import objects_to_array

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
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
        endurance (float): Camel endurance.
        supply (float): Camel supply.
        x_past (numpy.ndarray): Camel's past position.
        f_past (float): Camel's past function/fitness value.
        steps (int): Age of camel.

    See Also:
        * :class:`niapy.algorithms.Individual`

    """

    def __init__(self, endurance_init=None, supply_init=None, **kwargs):
        r"""Initialize the Camel.

        Args:
            endurance_init (Optional[float]): Starting endurance of Camel.
            supply_init (Optional[float]): Stating supply of Camel.

        See Also:
            * :func:`niapy.algorithms.Individual.__init__`

        """
        super().__init__(**kwargs)
        self.endurance = endurance_init
        self.endurance_past = endurance_init
        self.supply = supply_init
        self.supply_past = supply_init
        self.x_past = self.x
        self.f_past = self.f
        self.temperature = None
        self.steps = 0

    def next_temperature(self, min_temperature, max_temperature, rng):
        r"""Apply nextT function on Camel.

        Args:
            min_temperature (float): Minimum temperature.
            max_temperature (float): Maximum temperature.
            rng (numpy.random.Generator): Random number generator.

        """
        self.temperature = rng.uniform(min_temperature, max_temperature)

    def next_supply(self, burden_factor, max_iters):
        r"""Apply nextS on Camel.

        Args:
            burden_factor (float): Burden factor.
            max_iters (int): Number of Camel Algorithm iterations/generations.

        """
        self.supply = self.supply_past * (1 - burden_factor * self.steps / max_iters)

    def next_endurance(self, max_temperature, max_iters):
        r"""Apply function nextE on function on Camel.

        Args:
            max_iters (int): Number of Camel Algorithm iterations/generations
            max_temperature (float): Maximum temperature of environment

        """
        self.endurance = self.endurance_past * (1 - self.temperature / max_temperature) * (1 - self.steps / max_iters)

    def next_x(self, cb, endurance_init, supply_init, task, rng):
        r"""Apply function next_x on Camel.

        This method/function move this Camel to new position in search space.

        Args:
            cb (numpy.ndarray): Best Camel in population.
            endurance_init (float): Starting endurance of camel.
            supply_init (float): Starting supply of camel.
            task (Task): Optimization task.
            rng (numpy.random.Generator): Random number generator.

        """
        delta = rng.uniform(-1, 1)
        self.x = self.x_past + delta * (1 - (self.endurance / endurance_init)) * np.exp(
            1 - self.supply / supply_init) * (
                         cb - self.x_past)
        if not task.is_feasible(self.x):
            self.x = self.x_past
        else:
            self.f = task.eval(self.x)

    def next(self):
        r"""Save new position of Camel to old position."""
        self.x_past = self.x.copy()
        self.f_past = self.f
        self.endurance_past = self.endurance
        self.supply_past = self.supply
        self.steps += 1
        return self

    def refill(self, supply=None, endurance=None):
        r"""Apply this function to Camel.

        Args:
            supply (float): New value of Camel supply.
            endurance (float): New value of Camel endurance.

        """
        self.supply = supply
        self.endurance = endurance


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
        Ali, Ramzy. (2016). Novel Optimization Algorithm Inspired by Camel Traveling Behavior.
        Iraq J. Electrical and Electronic Engineering. 12. 167-177.

    Attributes:
        Name (List[str]): List of strings representing name of the algorithm.
        population_size (Optional[int]): Population size :math:`\in [1, \infty)`.
        burden_factor (Optional[float]): Burden factor :math:`\in [0, 1]`.
        death_rate (Optional[float]): Dying rate :math:`\in [0, 1]`.
        visibility (Optional[float]): View range of camel.
        supply_init (Optional[float]): Initial supply :math:`\in (0, \infty)`.
        endurance_init (Optional[float]): Initial endurance :math:`\in (0, \infty)`.
        min_temperature (Optional[float]): Minimum temperature, must be true :math:`$T_{min} < T_{max}`.
        max_temperature (Optional[float]): Maximum temperature, must be true :math:`T_{min} < T_{max}`.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['CamelAlgorithm', 'CA']

    @staticmethod
    def info():
        r"""Get information about algorithm.

        Returns:
            str: Algorithm information

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r'''Ali, Ramzy. (2016). Novel Optimization Algorithm Inspired by Camel Traveling Behavior.
        Iraq J. Electrical and Electronic Engineering. 12. 167-177.'''

    def __init__(self, population_size=50, burden_factor=0.25, death_rate=0.5, visibility=0.5, supply_init=10,
                 endurance_init=10, min_temperature=-10, max_temperature=10, *args, **kwargs):
        r"""Initialize CamelAlgorithm.

        Args:
            population_size (Optional[int]): Population size :math:`\in [1, \infty)`.
            burden_factor (Optional[float]): Burden factor :math:`\in [0, 1]`.
            death_rate (Optional[float]): Dying rate :math:`\in [0, 1]`.
            visibility (Optional[float]): View range of camel.
            supply_init (Optional[float]): Initial supply :math:`\in (0, \infty)`.
            endurance_init (Optional[float]): Initial endurance :math:`\in (0, \infty)`.
            min_temperature (Optional[float]): Minimum temperature, must be true :math:`$T_{min} < T_{max}`.
            max_temperature (Optional[float]): Maximum temperature, must be true :math:`T_{min} < T_{max}`.


        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, individual_type=kwargs.pop('individual_type', Camel),
                         initialization_function=kwargs.pop('initialization_function', self.init_pop), *args, **kwargs)
        self.burden_factor = burden_factor
        self.death_rate = death_rate
        self.visibility = visibility
        self.supply_init = supply_init
        self.endurance_init = endurance_init
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

    def set_parameters(self, population_size=50, burden_factor=0.25, death_rate=0.5, visibility=0.5, supply_init=10,
                       endurance_init=10, min_temperature=-10, max_temperature=10, **kwargs):
        r"""Set the arguments of an algorithm.

        Args:
            population_size (Optional[int]): Population size :math:`\in [1, \infty)`.
            burden_factor (Optional[float]): Burden factor :math:`\in [0, 1]`.
            death_rate (Optional[float]): Dying rate :math:`\in [0, 1]`.
            visibility (Optional[float]): View range of camel.
            supply_init (Optional[float]): Initial supply :math:`\in (0, \infty)`.
            endurance_init (Optional[float]): Initial endurance :math:`\in (0, \infty)`.
            min_temperature (Optional[float]): Minimum temperature, must be true :math:`$T_{min} < T_{max}`.
            max_temperature (Optional[float]): Maximum temperature, must be true :math:`T_{min} < T_{max}`.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, individual_type=Camel,
                               initialization_function=kwargs.pop('initialization_function', self.init_pop), **kwargs)
        self.burden_factor = burden_factor
        self.death_rate = death_rate
        self.visibility = visibility
        self.supply_init = supply_init
        self.endurance_init = endurance_init
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm Parameters.

        """
        d = super().get_parameters()
        d.update({
            'burden_factor': self.burden_factor,
            'death_rate': self.death_rate,
            'visibility': self.visibility,
            'supply_init': self.supply_init,
            'endurance_init': self.endurance_init,
            'min_temperature': self.min_temperature,
            'max_temperature': self.max_temperature
        })
        return d

    def init_pop(self, task, population_size, rng, individual_type, **_kwargs):
        r"""Initialize starting population.

        Args:
            task (Task): Optimization task.
            population_size (int): Number of camels in population.
            rng (numpy.random.Generator): Random number generator.
            individual_type (Type[Individual]): Individual type.

        Returns:
            Tuple[numpy.ndarray[Camel], numpy.ndarray[float]]:
                1. Initialize population of camels.
                2. Initialized populations function/fitness values.

        """
        caravan = objects_to_array(
            [individual_type(endurance_init=self.endurance_init, supply_init=self.supply_init, task=task, rng=rng,
                             e=True) for _ in range(population_size)])
        return caravan, np.asarray([c.f for c in caravan])

    def walk(self, camel, best_x, task):
        r"""Move the camel in search space.

        Args:
            camel (Camel): Camel that we want to move.
            best_x (numpy.ndarray): Global best coordinates.
            task (Task): Optimization task.

        Returns:
            Camel: Camel that moved in the search space.

        """
        camel.next_temperature(self.min_temperature, self.max_temperature, self.rng)
        camel.next_supply(self.burden_factor, task.max_iters)
        camel.next_endurance(self.max_temperature, task.max_iters)
        camel.next_x(best_x, self.endurance_init, self.supply_init, task, self.rng)
        return camel

    def oasis(self, c):
        r"""Apply oasis function to camel.

        Args:
            c (Camel): Camel to apply oasis on.

        Returns:
            Camel: Camel with applied oasis on.

        """
        if self.random() > (1 - self.visibility) and c.f < c.f_past:
            c.refill(self.supply_init, self.endurance_init)
        return c

    def life_cycle(self, camel, task):
        r"""Apply life cycle to Camel.

        Args:
            camel (Camel): Camel to apply life cycle.
            task (Task): Optimization task.

        Returns:
            Camel: Camel with life cycle applied to it.

        """
        if camel.f_past < self.death_rate * camel.f:
            return Camel(self.endurance_init, self.supply_init, rng=self.rng, task=task)
        else:
            return camel.next()

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Camel Algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray[Camel]): Current population of Camels.
            population_fitness (numpy.ndarray[float]): Current population fitness/function values.
            best_x (numpy.ndarray): Current best Camel.
            best_fitness (float): Current best Camel fitness/function value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, dict]:
                1. New population
                2. New population function/fitness value
                3. New global best solution
                4. New global best fitness/objective value
                5. Additional arguments

        """
        new_caravan = objects_to_array([self.walk(c, best_x, task) for c in population])
        new_caravan = objects_to_array([self.oasis(c) for c in new_caravan])
        new_caravan = objects_to_array([self.life_cycle(c, task) for c in new_caravan])
        new_caravan_fitness = np.asarray([c.f for c in new_caravan])
        best_x, best_fitness = self.get_best(new_caravan, new_caravan_fitness, best_x, best_fitness)
        return new_caravan, new_caravan_fitness, best_x, best_fitness, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
