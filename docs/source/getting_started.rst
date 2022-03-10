Getting Started
===============

It's time to write your first NiaPy example. Firstly, if you haven't already, install NiaPy package on your system
using following command:

.. code:: bash

    pip install niapy

or:

.. code:: bash

    conda install -c niaorg niapy

When package is successfully installed you are ready to write you first example.

Basic example
-------------
In this example, let's say, we want to try out Gray Wolf Optimizer algorithm against the Pintér problem.
Firstly, we have to create new file, with name, for example *basic_example.py*. Then we have to import chosen
algorithm from NiaPy, so we can use it. Afterwards we initialize ParticleSwarmAlgorithm class instance and run the algorithm.
Given bellow is complete source code of basic example.

.. code:: python

    from niapy.algorithms.basic import ParticleSwarmAlgorithm
    from niapy.task import Task

    # we will run 10 repetitions of Weighed, velocity clamped PSO on the Pinter problem
    for i in range(10):
        task = Task(problem='pinter', dimension=10, max_evals=10000)
        algorithm = ParticleSwarmAlgorithm(population_size=100, w=0.9, c1=0.5, c2=0.3, min_velocity=-1, max_velocity=1)
        best_x, best_fit = algorithm.run(task)
        print(best_fit)



Given example can be run with ``python basic_example.py`` command and should give you similar output as
following:

.. code:: bash

    0.008773534890863646
    0.036616190934621755
    186.75116812592546
    0.024186452828927896
    263.5697469837348
    45.420706924365916
    0.6946753611091367
    7.756100204780568
    5.839673314425907
    0.06732518679742806



Customize problem bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~
By default, the Pintér problem has the bound set to -10 and 10. We can override those predefined
values very easily. We will modify our basic example to run PSO against Pintér problem
function with custom problem bounds set to -5 and 5. Given bellow is the complete source code of customized
basic example.

.. code:: python

    from niapy.algorithms.basic import ParticleSwarmAlgorithm
    from niapy.task import Task
    from niapy.problems import Pinter

    # initialize Pinter problem with custom bound
    pinter = Pinter(dimension=20, lower=-5, upper=5)

    # we will run 10 repetitions of PSO against Pinter problem function
    for i in range(10):
        task = Task(problem=pinter, max_iters=100)
        algorithm = ParticleSwarmAlgorithm(population_size=100, w=0.9, c1=0.5, c2=0.3, min_velocity=-1, max_velocity=1)

        # running algorithm returns best found coordinates and fitness
        best_x, best_fit = algorithm.run(task)

        # printing best minimum
        print(best_fit)

Given example can be run with ``python basic_example.py`` command and should give you similar output as
following:

.. code:: bash

    352.42267398695526
    15.962765124936741
    356.51781541486224
    195.64616754731315
    99.92445777071993
    142.36934412674793
    1.9566799783197366
    350.4330002633882
    183.93200436114898
    208.5557966507149

Advanced example
----------------
In this example we will show you how to implement a custom problem class and use it with any of
implemented algorithms. First let's create new file named advanced_example.py. As in the previous examples
we wil import algorithm we want to use from niapy module.

For our custom optimization function, we have to create new class. Let's name it *MyProblem*. In the initialization
method of *MyProblem* class we have to set the *dimension*, *lower* and *upper* bounds of the problem. Afterwards we have to
override the abstract method _evaluate which takes a parameter *x*, the solution to be evaluated, and returns the function value.
Now we should have something similar as is shown in code snippet bellow.

.. code:: python

    from niapy.task import Task
    from niapy.problems import Problem
    from niapy.algorithms.basic import ParticleSwarmAlgorithm
    import numpy as np

    # our custom Problem class
    class MyProblem(Problem):
        def __init__(self, dimension, lower=-10, upper=10, *args, **kwargs):
            super().__init__(dimension, lower, upper, *args, **kwargs)

        def _evaluate(self, x):
            return np.sum(x ** 2)


Now, all we have to do is to initialize our algorithm as in previous examples and pass as problem parameter,
instance of our *MyProblem* class.

.. code:: python

    my_problem = MyProblem(dimension=20)
    for i in range(10):
        task = Task(problem=my_problem, max_iters=100)
        algorithm = ParticleSwarmAlgorithm(population_size=100, w=0.9, c1=0.5, c2=0.3, min_velocity=-1, max_velocity=1)

        # running algorithm returns best found minimum
        best_x, best_fit = algorithm.run(task)

        # printing best minimum
        print(best_fit)

Now we can run our advanced example with following command python advanced_example.py. The results should be
similar to those bellow.

.. code:: bash

    0.0009232355257327939
    0.0012993317932349976
    0.0026231249714186128
    0.001404157010165644
    0.0012822904697534436
    0.002202199078241452
    0.00216496834770605
    0.0010092926171364153
    0.0007432303831633373
    0.0006545778971016809

Advanced example with custom population initialization
------------------------------------------------------
In this examples we will showcase how to define our own population initialization function for previous advanced example.
We extend previous example by adding another function, lets name it my_init which would receive the task, population size,
a random number generator and optional parameters. Such population initialization function is presented bellow.

.. code:: python

    import numpy as np


    # custom population initialization function
    def my_init(task, population_size, rng, **kwargs):
        pop = 0.2 + rng.random((population_size, task.dimension)) * task.range
        fitness = np.apply_along_axis(task.eval, 1, pop)
        return pop, fitness


The complete example would look something like this.

.. code:: python

    import numpy as np
    from niapy.task import Task
    from niapy.problems import Problem
    from niapy.algorithms.basic import ParticleSwarmAlgorithm

    # our custom Problem class
    class MyProblem(Problem):
        def __init__(self, dimension, lower=-10, upper=10, *args, **kwargs):
            super().__init__(dimension, lower, upper, *args, **kwargs)

        def _evaluate(self, x):
            return np.sum(x ** 2)

    # custom population initialization function
    def my_init(task, population_size, rng, **kwargs):
        pop = 0.2 + rng.random((population_size, task.dimension)) * task.range
        fpop = np.apply_along_axis(task.eval, 1, pop)
        return pop, fpop

    # we will run 10 repetitions of PSO against our custom MyProblem problem function
    my_problem = MyProblem(dimension=20)
    for i in range(10):
        task = Task(problem=my_problem, max_iters=100)
        algorithm = ParticleSwarmAlgorithm(population_size=100, w=0.9, c1=0.5, c2=0.3, min_velocity=-1, max_velocity=1, initialization_function=my_init)

        # running algorithm returns best found minimum
        best_x, best_fit = algorithm.run(task)

        # printing best minimum
        print(best_fit)

And results when running the above example should be similar to those bellow.

.. code:: bash

    0.0370956467450487
    0.0036632556827966758
    0.0017599467532291731
    0.0006688678943170477
    0.0010923591711792472
    0.001714310421328247
    0.002196032177635475
    0.0011230918470056704
    0.0007371056198024898
    0.013706530361724643

Runner example
--------------
For easier comparison between many different algorithms and problems, we developed a useful feature called
*Runner*. Runner can take an array of algorithms and an array of problems to compare and run all combinations
for you. We also provide an extra feature, which lets you easily exports those results in many different formats
(Pandas DataFrame, Excel, JSON).

Below is given a usage example of our *Runner*, which will run various algorithms and problems
functions. Results will be exported as JSON.

.. code:: python

    from niapy import Runner
    from niapy.algorithms.basic import (
        GreyWolfOptimizer,
        ParticleSwarmAlgorithm
    )
    from niapy.problems import (
        Problem,
        Ackley,
        Griewank,
        Sphere,
        HappyCat
    )

    class MyProblem(Problem):
        def __init__(self, dimension, lower=-10, upper=10, *args, **kwargs):
            super().__init__(dimension, lower, upper, *args, **kwargs)

        def _evaluate(self, x):
            return np.sum(x ** 2)

    runner = Runner(
        dimension=40,
        max_evals=100,
        runs=2,
        algorithms=[
            GreyWolfOptimizer(),
            "FlowerPollinationAlgorithm",
            ParticleSwarmAlgorithm(),
            "HybridBatAlgorithm",
            "SimulatedAnnealing",
            "CuckooSearch"],
        problems=[
            Ackley(40),
            Griewank(40),
            Sphere(40),
            HappyCat(40),
            "rastrigin",
            MyProblem(dimension=40)
        ]
    )

    runner.run(export='json', verbose=True)


Output of running above example should look like something as following.

.. code:: bash

    INFO:niapy.runner.Runner:Running GreyWolfOptimizer...
    INFO:niapy.runner.Runner:Running GreyWolfOptimizer algorithm on Ackley problem...
    INFO:niapy.runner.Runner:Running GreyWolfOptimizer algorithm on Griewank problem...
    INFO:niapy.runner.Runner:Running GreyWolfOptimizer algorithm on Sphere problem...
    INFO:niapy.runner.Runner:Running GreyWolfOptimizer algorithm on HappyCat problem...
    INFO:niapy.runner.Runner:Running GreyWolfOptimizer algorithm on rastrigin problem...
    INFO:niapy.runner.Runner:Running GreyWolfOptimizer algorithm on MyProblem problem...
    INFO:niapy.runner.Runner:---------------------------------------------------
    INFO:niapy.runner.Runner:Running FlowerPollinationAlgorithm...
    INFO:niapy.runner.Runner:Running FlowerPollinationAlgorithm algorithm on Ackley problem...
    INFO:niapy.runner.Runner:Running FlowerPollinationAlgorithm algorithm on Griewank problem...
    INFO:niapy.runner.Runner:Running FlowerPollinationAlgorithm algorithm on Sphere problem...
    INFO:niapy.runner.Runner:Running FlowerPollinationAlgorithm algorithm on HappyCat problem...
    INFO:niapy.runner.Runner:Running FlowerPollinationAlgorithm algorithm on rastrigin problem...
    INFO:niapy.runner.Runner:Running FlowerPollinationAlgorithm algorithm on MyProblem problem...
    INFO:niapy.runner.Runner:---------------------------------------------------
    INFO:niapy.runner.Runner:Running ParticleSwarmAlgorithm...
    INFO:niapy.runner.Runner:Running ParticleSwarmAlgorithm algorithm on Ackley problem...
    INFO:niapy.runner.Runner:Running ParticleSwarmAlgorithm algorithm on Griewank problem...
    INFO:niapy.runner.Runner:Running ParticleSwarmAlgorithm algorithm on Sphere problem...
    INFO:niapy.runner.Runner:Running ParticleSwarmAlgorithm algorithm on HappyCat problem...
    INFO:niapy.runner.Runner:Running ParticleSwarmAlgorithm algorithm on rastrigin problem...
    INFO:niapy.runner.Runner:Running ParticleSwarmAlgorithm algorithm on MyProblem problem...
    INFO:niapy.runner.Runner:---------------------------------------------------
    INFO:niapy.runner.Runner:Running HybridBatAlgorithm...
    INFO:niapy.runner.Runner:Running HybridBatAlgorithm algorithm on Ackley problem...
    INFO:niapy.runner.Runner:Running HybridBatAlgorithm algorithm on Griewank problem...
    INFO:niapy.runner.Runner:Running HybridBatAlgorithm algorithm on Sphere problem...
    INFO:niapy.runner.Runner:Running HybridBatAlgorithm algorithm on HappyCat problem...
    INFO:niapy.runner.Runner:Running HybridBatAlgorithm algorithm on rastrigin problem...
    INFO:niapy.runner.Runner:Running HybridBatAlgorithm algorithm on MyProblem problem...
    INFO:niapy.runner.Runner:---------------------------------------------------
    INFO:niapy.runner.Runner:Running SimulatedAnnealing...
    INFO:niapy.runner.Runner:Running SimulatedAnnealing algorithm on Ackley problem...
    INFO:niapy.runner.Runner:Running SimulatedAnnealing algorithm on Griewank problem...
    INFO:niapy.runner.Runner:Running SimulatedAnnealing algorithm on Sphere problem...
    INFO:niapy.runner.Runner:Running SimulatedAnnealing algorithm on HappyCat problem...
    INFO:niapy.runner.Runner:Running SimulatedAnnealing algorithm on rastrigin problem...
    INFO:niapy.runner.Runner:Running SimulatedAnnealing algorithm on MyProblem problem...
    INFO:niapy.runner.Runner:---------------------------------------------------
    INFO:niapy.runner.Runner:Running CuckooSearch...
    INFO:niapy.runner.Runner:Running CuckooSearch algorithm on Ackley problem...
    INFO:niapy.runner.Runner:Running CuckooSearch algorithm on Griewank problem...
    INFO:niapy.runner.Runner:Running CuckooSearch algorithm on Sphere problem...
    INFO:niapy.runner.Runner:Running CuckooSearch algorithm on HappyCat problem...
    INFO:niapy.runner.Runner:Running CuckooSearch algorithm on rastrigin problem...
    INFO:niapy.runner.Runner:Running CuckooSearch algorithm on MyProblem problem...
    INFO:niapy.runner.Runner:---------------------------------------------------
    INFO:niapy.runner.Runner:Export to JSON completed!

Results will be also exported in a JSON file (in export folder).
