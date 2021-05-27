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
In this example, let's say, we want to try out Gray Wolf Optimizer algorithm against Pintér benchmark function.
Firstly, we have to create new file, with name, for example *basic_example.py*. Then we have to import chosen
algorithm from NiaPy, so we can use it. Afterwards we initialize GreyWolfOptimizer class instance and run the algorithm.
Given bellow is complete source code of basic example.

.. code:: python

    from niapy.algorithms.basic import GreyWolfOptimizer
    from niapy.task import StoppingTask

    # we will run 10 repetitions of Grey Wolf Optimizer against Pinter benchmark function
    for i in range(10):
        task = StoppingTask(benchmark='pinter', dimension=10, max_evals=1000)
        algorithm = GreyWolfOptimizer(population_size=20)
        best = algorithm.run(task)
        print(best[-1])


Given example can be run with ``python basic_example.py`` command and should give you similar output as
following:

.. code:: bash

    0.27046073106003377
    50.89301186976975
    1.089147452727528
    1.18418058254198
    102.46876441081712
    0.11237241605812048
    1.8869331711450696
    0.04861881403346098
    2.5748611081742325
    135.6754069530421


Customize benchmark bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~
By default, Pintér benchmark has the bound set to -10 and 10. We can simply override those predefined
values very easily. We will modify our basic example to run Grey Wolf Optimizer against Pintér benchmark
function with custom benchmark bounds set to -5 and 5. Given bellow is complete source code of customized
basic example.

.. code:: python

    from niapy.algorithms.basic import GreyWolfOptimizer
    from niapy.task import StoppingTask, OptimizationType
    from niapy.benchmarks import Pinter

    # initialize Pinter benchmark with custom bound
    pinter = Pinter(dimension=20, lower=-5, upper=5)

    # we will run 10 repetitions of Grey Wolf Optimizer against Pinter benchmark function
    for i in range(10):
        # first parameter takes dimension of problem
        # second parameter takes the number of function evaluations
        # third parameter is benchmark optimization type
        # forth parameter is benchmark function
        task = StoppingTask(benchmark=pinter, max_iters=100, optimization_type=OptimizationType.MINIMIZATION)

        # parameter is population size
        algo = GreyWolfOptimizer(population_size=20)

        # running algorithm returns best found minimum
        best = algo.run(task)

        # printing best minimum
        print(best[-1])

Given example can be run with ``python basic_example.py`` command and should give you similar output as
following:

.. code:: bash

    3.6505427897004535e-05
    3.8199245597156976e-05
    0.0001411622032519498
    3.756895566558108e-06
    4.424570228729335e-05
    6.114113555664476e-06
    1.3978581995165064e-05
    5.5851861300797835e-06
    7.909208902574658e-06
    2.4419767659672064e-05

Advanced example
----------------
In this example we will show you how to implement your own benchmark function and use it with any of
implemented algorithms. First let's create new file named advanced_example.py. As in the previous examples
we wil import algorithm we want to use from NiaPy module.

For our custom benchmark function, we have to create new class. Let's name it *MyBenchmark*. In the initialization
method of *MyBenchmark* class we have to set *Lower* and *Upper* bounds of the function. Afterwards we have to
implement a function which returns evaluation function which takes two parameters *D* (as dimension of problem)
and *sol* (as solution of problem). Now we should have something similar as is shown in code snippet bellow.

.. code:: python

    from niapy.task import StoppingTask
    from niapy.benchmarks import Benchmark
    from niapy.algorithms.basic import GreyWolfOptimizer
    import numpy as np

    # our custom benchmark class
    class MyBenchmark(Benchmark):
        def __init__(self, dimension, lower=-10, upper=10, *args, **kwargs):
            super().__init__(dimension, lower, upper, *args, **kwargs)

        def _evaluate(self, x):
            return np.sum(x ** 2)


Now, all we have to do is to initialize our algorithm as in previous examples and pass as benchmark parameter,
instance of our *MyBenchmark* class.

.. code:: python

    my_Benchmark = MyBenchmark(dimension=20)
    for i in range(10):
        task = StoppingTask(benchmark=my_benchmark, max_iters=100)

        # parameter is population size
        algo = GreyWolfOptimizer(population_size=20)

        # running algorithm returns best found minimum
        best = algo.run(task)

        # printing best minimum
        print(best[-1])

Now we can run our advanced example with following command python advanced_example.py. The results should be
similar to those bellow.

.. code:: bash

    7.606465129178389e-09
    5.288697102580944e-08
    6.875762169124336e-09
    1.386574251424837e-08
    2.174923591233085e-08
    2.578545710051624e-09
    1.1400628541972142e-08
    2.99387377733644e-08
    7.029492316948289e-09
    7.426212520156997e-09

Advanced example with custom population initialization
------------------------------------------------------
In this examples we will showcase how to define our own population initialization function for previous advanced example.
We extend previous example by adding another function, lets name it my_init which would receive the task, population size,
a random number generator and optional parameters. Such population initialization function is presented bellow.

.. code:: python

    import numpy as np


    # custom population initialization function
    def my_init(task, population_size, rng, **kwargs):
        pop = 0.2 + rng.random(population_size, task.dimension) * task.range
        fpop = np.apply_along_axis(task.eval, 1, pop)
        return pop, fpop


The complete example would look something like this.

.. code:: python

    import numpy as np
    from niapy.task import StoppingTask
    from niapy.benchmarks import Benchmark
    from niapy.algorithms.basic import GreyWolfOptimizer

    # our custom benchmark class
    class MyBenchmark(Benchmark):
        def __init__(self, dimension, lower=-10, upper=10, *args, **kwargs):
            super().__init__(dimension, lower, upper, *args, **kwargs)

        def _evaluate(self, x):
            return np.sum(x ** 2)

    # custom population initialization function
    def my_init(task, population_size, rng, **kwargs):
        pop = 0.2 + rng.random(population_size, task.dimension) * task.range
        fpop = np.apply_along_axis(task.eval, 1, pop)
        return pop, fpop

    # we will run 10 repetitions of Grey Wolf Optimizer against our custom MyBenchmark benchmark function
    my_benchmark = MyBenchmark(dimension=20)
    for i in range(10):
        task = StoppingTask(benchmark=my_benchmark, max_iters=100)

        # parameter is population size
        algo = GreyWolfOptimizer(population_size=20, initialization_function=my_init)

        # running algorithm returns best found minimum
        best = algo.run(task)

        # printing best minimum
        print(best[-1])

And results when running the above example should be similar to those bellow.

.. code:: bash

    4.708930032276375e-08
    3.074627144384774e-08
    3.4164735698703244e-08
    4.9961114415227386e-08
    7.804954011212186e-09
    8.54822031684741e-08
    1.8625917477836128e-08
    1.0765481838194546e-08
    4.535387196032371e-08
    1.3303233444716197e-07

Runner example
--------------
For easier comparison between many different algorithms and benchmarks, we developed a useful feature called
*Runner*. Runner can take an array of algorithms and an array of benchmarks to compare and run all combinations
for you. We also provide an extra feature, which lets you easily exports those results in many different formats
(Pandas DataFrame, Excel, JSON).

Below is given a usage example of our *Runner*, which will run various algorithms and benchmark
functions. Results will be exported as JSON.

.. code:: python

    from niapy import Runner
    from niapy.algorithms.basic import (
        GreyWolfOptimizer,
        ParticleSwarmAlgorithm
    )
    from niapy.benchmarks import (
        Benchmark,
        Ackley,
        Griewank,
        Sphere,
        HappyCat
    )

    class MyBenchmark(Benchmark):
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
        benchmarks=[
            Ackley(40),
            Griewank(40),
            Sphere(40),
            HappyCat(40),
            "rastrigin",
            MyBenchmark(40)
        ]
    )

    runner.run(export='json', verbose=True)


Output of running above example should look like something as following.

.. code:: bash

    INFO:niapy.runner.Runner:Running GreyWolfOptimizer...
    INFO:niapy.runner.Runner:Running GreyWolfOptimizer algorithm on Ackley benchmark...
    INFO:niapy.runner.Runner:Running GreyWolfOptimizer algorithm on Griewank benchmark...
    INFO:niapy.runner.Runner:Running GreyWolfOptimizer algorithm on Sphere benchmark...
    INFO:niapy.runner.Runner:Running GreyWolfOptimizer algorithm on HappyCat benchmark...
    INFO:niapy.runner.Runner:Running GreyWolfOptimizer algorithm on rastrigin benchmark...
    INFO:niapy.runner.Runner:Running GreyWolfOptimizer algorithm on MyBenchmark benchmark...
    INFO:niapy.runner.Runner:---------------------------------------------------
    INFO:niapy.runner.Runner:Running FlowerPollinationAlgorithm...
    INFO:niapy.runner.Runner:Running FlowerPollinationAlgorithm algorithm on Ackley benchmark...
    INFO:niapy.runner.Runner:Running FlowerPollinationAlgorithm algorithm on Griewank benchmark...
    INFO:niapy.runner.Runner:Running FlowerPollinationAlgorithm algorithm on Sphere benchmark...
    INFO:niapy.runner.Runner:Running FlowerPollinationAlgorithm algorithm on HappyCat benchmark...
    INFO:niapy.runner.Runner:Running FlowerPollinationAlgorithm algorithm on rastrigin benchmark...
    INFO:niapy.runner.Runner:Running FlowerPollinationAlgorithm algorithm on MyBenchmark benchmark...
    INFO:niapy.runner.Runner:---------------------------------------------------
    INFO:niapy.runner.Runner:Running ParticleSwarmAlgorithm...
    INFO:niapy.runner.Runner:Running ParticleSwarmAlgorithm algorithm on Ackley benchmark...
    INFO:niapy.runner.Runner:Running ParticleSwarmAlgorithm algorithm on Griewank benchmark...
    INFO:niapy.runner.Runner:Running ParticleSwarmAlgorithm algorithm on Sphere benchmark...
    INFO:niapy.runner.Runner:Running ParticleSwarmAlgorithm algorithm on HappyCat benchmark...
    INFO:niapy.runner.Runner:Running ParticleSwarmAlgorithm algorithm on rastrigin benchmark...
    INFO:niapy.runner.Runner:Running ParticleSwarmAlgorithm algorithm on MyBenchmark benchmark...
    INFO:niapy.runner.Runner:---------------------------------------------------
    INFO:niapy.runner.Runner:Running HybridBatAlgorithm...
    INFO:niapy.runner.Runner:Running HybridBatAlgorithm algorithm on Ackley benchmark...
    INFO:niapy.runner.Runner:Running HybridBatAlgorithm algorithm on Griewank benchmark...
    INFO:niapy.runner.Runner:Running HybridBatAlgorithm algorithm on Sphere benchmark...
    INFO:niapy.runner.Runner:Running HybridBatAlgorithm algorithm on HappyCat benchmark...
    INFO:niapy.runner.Runner:Running HybridBatAlgorithm algorithm on rastrigin benchmark...
    INFO:niapy.runner.Runner:Running HybridBatAlgorithm algorithm on MyBenchmark benchmark...
    INFO:niapy.runner.Runner:---------------------------------------------------
    INFO:niapy.runner.Runner:Running SimulatedAnnealing...
    INFO:niapy.runner.Runner:Running SimulatedAnnealing algorithm on Ackley benchmark...
    INFO:niapy.runner.Runner:Running SimulatedAnnealing algorithm on Griewank benchmark...
    INFO:niapy.runner.Runner:Running SimulatedAnnealing algorithm on Sphere benchmark...
    INFO:niapy.runner.Runner:Running SimulatedAnnealing algorithm on HappyCat benchmark...
    INFO:niapy.runner.Runner:Running SimulatedAnnealing algorithm on rastrigin benchmark...
    INFO:niapy.runner.Runner:Running SimulatedAnnealing algorithm on MyBenchmark benchmark...
    INFO:niapy.runner.Runner:---------------------------------------------------
    INFO:niapy.runner.Runner:Running CuckooSearch...
    INFO:niapy.runner.Runner:Running CuckooSearch algorithm on Ackley benchmark...
    INFO:niapy.runner.Runner:Running CuckooSearch algorithm on Griewank benchmark...
    INFO:niapy.runner.Runner:Running CuckooSearch algorithm on Sphere benchmark...
    INFO:niapy.runner.Runner:Running CuckooSearch algorithm on HappyCat benchmark...
    INFO:niapy.runner.Runner:Running CuckooSearch algorithm on rastrigin benchmark...
    INFO:niapy.runner.Runner:Running CuckooSearch algorithm on MyBenchmark benchmark...
    INFO:niapy.runner.Runner:---------------------------------------------------
    INFO:niapy.runner.Runner:Export to JSON completed!

Results will be also exported in a JSON file (in export folder).
