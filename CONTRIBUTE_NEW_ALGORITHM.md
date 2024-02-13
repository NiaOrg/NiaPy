# Contributing new nature-inspired algorithms to NiaPy framework

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Adding a New Algorithm](#adding-a-new-algorithm)
  - [Algorithm Overview](#algorithm-overview)
  - [Code Structure](#code-structure)
    - [Main Algorithm File](#main-algorithm-file)
    - [Documentation](#documentation)
    - [Tests](#tests)
    - [Examples](#examples)
    - [Code Style and PEP 8 Compliance](#code-style-and-pep-8-compliance)
- [License](#license)
- [References](#references)

## Introduction

Thank you for taking the time to explore this document. The NiaPy framework has experienced over 5 years of growth, thanks to the helpfulness and willingness of our community contributors. Their significant contributions have played a vital role in the success of this project. While designing and implementing key functions for the main framework is crucial, adding new algorithms is an equally important step in keeping the framework dynamic and up-to-date.

Currently, the NiaPy framework hosts more than 30 algorithms, categorized into three families:

- **Basic Algorithms**: These are vanilla implementations of nature-inspired algorithms, often derived from research papers where authors propose new algorithms.

- **Modified Algorithms**: This category includes modified, hybrid, and adaptive variants of basic algorithms.

- **Other Algorithms**: This encompasses algorithms that fit well under the optimization umbrella but cannot be directly classified into the previously mentioned families (e.g., random search).

For a comprehensive overview of the implemented algorithms, please refer to the [following document](https://raw.githubusercontent.com/firefly-cpp/NiaPy/master/Algorithms.md).

## Prerequisites

Before diving into the contribution process, ensure you meet the following prerequisites:

- Basic knowledge of optimization algorithms.
- Familiarity with Python and Numpy.
- An understanding of NiaPy's structure and existing algorithms.

We'll delve into the details of the last prerequisite in this document.

## Adding a New Algorithm

### Algorithm Overview

Given the current influx of metaphor-based nature-inspired algorithms in the research area, it is essential to make wise choices when selecting which algorithm to implement and include in the NiaPy framework.

### Code Structure

#### Main Algorithm File

- Create a New File: Create a new file in `niapy/algorithms/{basic/modified/other}`. The file name should correspond to the acronym of the nature-inspired algorithm.

- Define Algorithm Class: Begin by defining a new class for your algorithm, inheriting from the appropriate base class (e.g., Algorithm). Name your class after the nature-inspired algorithm; avoid using acronyms.

```python
import logging
import numpy as np
from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['BatAlgorithm']

class BatAlgorithm(Algorithm):
    r"""Implementation of Bat algorithm.

    Reference paper:
        Yang, Xin-She. "A new metaheuristic bat-inspired algorithm." Nature inspired cooperative strategies for optimization (NICSO 2010). Springer, Berlin, Heidelberg, 2010. 65-74.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        loudness (float): Initial loudness.
        pulse_rate (float): Initial pulse rate.
        alpha (float): Parameter for controlling loudness decrease.
        gamma (float): Parameter for controlling pulse rate increase.
        min_frequency (float): Minimum frequency.
        max_frequency (float): Maximum frequency.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ["BatAlgorithm", "BA"]
```

- Initialization: Implement the `__init__` method to initialize necessary parameters and attributes specific to your algorithm. Ensure clarity and consistency in parameter naming.

```python
def __init__(self, population_size=40, loudness=1.0, pulse_rate=1.0, alpha=0.97, gamma=0.1, min_frequency=0.0,
                 max_frequency=2.0, *args, **kwargs):
        """Initialize BatAlgorithm.

        Args:
            population_size (Optional[int]): Population size.
            loudness (Optional[float]): Initial loudness.
            pulse_rate (Optional[float]): Initial pulse rate.
            alpha (Optional[float]): Parameter for controlling loudness decrease.
            gamma (Optional[float]): Parameter for controlling pulse rate increase.
            min_frequency (Optional[float]): Minimum frequency.
            max_frequency (Optional[float]): Maximum frequency.

        See Also:
            :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.alpha = alpha
        self.gamma = gamma
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
```

- Set Parameters Method: Override the `set_parameters` method to set parameters specific to your algorithm. Update the docstring to provide clear information about each parameter.

```python
def set_parameters(self, population_size=20, loudness=1.0, pulse_rate=1.0, alpha=0.97, gamma=0.1, min_frequency=0.0,
                       max_frequency=2.0, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            loudness (Optional[float]): Initial loudness.
            pulse_rate (Optional[float]): Initial pulse rate.
            alpha (Optional[float]): Parameter for controlling loudness decrease.
            gamma (Optional[float]): Parameter for controlling pulse rate increase.
            min_frequency (Optional[float]): Minimum frequency.
            max_frequency (Optional[float]): Maximum frequency.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.alpha = alpha
        self.gamma = gamma
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
```
- Get Parameters Method: Override the `get_parameters` method to return your algorithms parameters.

```python
def get_parameters(self):
    r"""Get parameters of the algorithm.

    Returns:
        Dict[str, Any]: Algorithm parameters.

    """
    parameters = super().get_parameters()
    parameters.update({
        'loudness': self.loudness,
        'pulse_rate': self.pulse_rate,
        'alpha': self.alpha,
        'gamma': self.gamma,
        'min_frequency': self.min_frequency,
        'max_frequency': self.max_frequency
    })
        return parameters
```

- Init Population Method: Adjust the `init_population` method to initialize the initial population for your algorithm. Add any additional parameters required by your algorithm.

```python
def init_population(self, task):
    r"""Initialize the starting population.

    Parameters:
        task (Task): Optimization task

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
            1. New population.
            2. New population fitness/function values.
            3. Additional arguments:
                * velocities (numpy.ndarray[float]): Velocities.
                * alpha (float): Previous iterations loudness.

    See Also:
        * :func:`niapy.algorithms.Algorithm.init_population`

    """
    population, fitness, d = super().init_population(task)
    velocities = np.zeros((self.population_size, task.dimension))
    d.update({'velocities': velocities, 'loudness': self.loudness})
    return population, fitness, d
```

- Run Iteration Method: Customize the `run_iteration` method, the core function of your algorithm. Implement the main logic, considering the optimization task, population, and iteration-specific parameters.

```python
def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
    r"""Core function of Bat Algorithm.

    Parameters:
        task (Task): Optimization task.
        population (numpy.ndarray): Current population
        population_fitness (numpy.ndarray[float]): Current population fitness/function values
        best_x (numpy.ndarray): Current best individual
        best_fitness (float): Current best individual function/fitness value
        params (Dict[str, Any]): Additional algorithm arguments

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
            1. New population
            2. New population fitness/function values
            3. New global best solution
            4. New global best fitness/objective value
            5. Additional arguments:
                * velocities (numpy.ndarray): Velocities.
                * alpha (float): Previous iterations loudness.

    """
    velocities = params.pop('velocities')
    loudness = params.pop('loudness') * self.alpha

    pulse_rate = self.pulse_rate * (1 - np.exp(-self.gamma * task.iters))

    for i in range(self.population_size):
        frequency = self.min_frequency + (self.max_frequency - self.min_frequency) * self.random()
        velocities[i] += (population[i] - best_x) * frequency
        if self.random() < pulse_rate:
            solution = task.repair(best_x + 0.1 * self.standard_normal(task.dimension) * loudness)
        else:
            solution = task.repair(population[i] + velocities[i], rng=self.rng)
        new_fitness = task.eval(solution)
        if (new_fitness <= population_fitness[i]) and (self.random() > loudness):
            population[i], population_fitness[i] = solution, new_fitness
        if new_fitness <= best_fitness:
            best_x, best_fitness = solution.copy(), new_fitness
    return population, population_fitness, best_x, best_fitness, {'velocities': velocities, 'loudness': loudness}
```

#### Documentation

- Provide Detailed Information: Document your algorithm, including its reference paper, attributes, and usage instructions. We use the [google docstring format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for documenting our codebase.

#### Tests

- Write Test Cases: Write at least two test cases to ensure the correctness of your algorithm's implementation (see [tests/test_ba.py](https://github.com/NiaOrg/NiaPy/blob/master/tests/test_ba.py)).

#### Examples

- Provide Simple Example: Include a simple example for running your algorithm (see [examples/run_ba.py](https://github.com/NiaOrg/NiaPy/blob/master/examples/run_ba.py)).

#### Code Style and PEP 8 Compliance

- Follow the PEP 8 style guide for Python.
- Maintain consistency in formatting and naming conventions.

## License

- MIT License: Each code contribution in this repository is licensed under the MIT license.

## References
