Getting Started
===============

It's time to write your first NiaPy example. Firstly, if you haven't already, install NiaPy package on your system
using following command:

.. code:: bash

    pip install NiaPy

When package is successfully installed you are ready to write you first example.

Basic example
-------------
In this example, let's say, we want to try out Gray Wolf Optimizer algorithm against Pintér benchmark function.
Firstly, we have to create new file, with name, for example *basic_example.py*. Then we have to import chosen 
algorithm from NiaPy, so we can use it. Afterwards we initialize GreyWolfOptimizer class instance and run the algorithm.
Given bellow is complete source code of basic example.

.. code:: python

    from NiaPy.algorithms.basic import GreyWolfOptimizer

    # we will run 10 repetitions of Grey Wolf Optimizer against Pinter benchmark function
    for i in range(10):
        # first parameter takes dimension of problem
        # second parameter is population size
        # third parameter takes the number of function evaluations
        # fourth parameter is benchmark function 
        algorithm = GreyWolfOptimizer(10, 20 , 10000, 'pinter')
        
        # running algorithm returns best found minimum
        best = algorithm.run()

        # printing best minimum
        print(best)


Given example can be run with ``python basic_example.py`` command and should give you similar output as
following:

.. code:: bash

    5.00762243998e-61
    2.67621982742e-57
    1.07156289063e-65
    8.43622715953e-61
    1.20903733381e-57
    6.32743651354e-62
    8.5819291808e-59
    8.10197009706e-59
    2.91642600474e-66
    5.73888425977e-54


Customize benchmark bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~
By default, Pintér benchmark has the bound set to -10 and 10. We can simply override those predefined
values very easily. We will modify our basic example to run Grey Wolf Optimizer against Pintér benchmark
function with custom benchmark bounds set to -5 and 5. Given bellow is complete source code of customized 
basic example.

.. code:: python

    from NiaPy.algorithms.basic import GreyWolfOptimizer
    from NiaPy.benchmarks import Pinter

    # initialize Pinter benchamrk with custom bound
    pinterCustom = Pinter(-5, 5)

    # we will run 10 repetitions of Grey Wolf Optimizer against Pinter benchmark function
    for i in range(10):
        # first parameter takes dimension of problem
        # second parameter is population size
        # third parameter takes the number of function evaluations
        # fourth parameter is benchmark function 
        algorithm = GreyWolfOptimizer(10, 20 , 10000, pinterCustom)
        
        # running algorithm returns best found minimum
        best = algorithm.run()

        # printing best minimum
        print(best)

Given example can be run with ``python basic_example.py`` command and should give you similar output as
following:

.. code:: bash

    7.43266143347e-64
    1.45053917474e-58
    1.01835349035e-55
    6.50410738064e-59
    2.18186445002e-61
    3.20274657669e-63
    3.23728585089e-62
    1.78481271215e-63
    7.81043837076e-66
    7.30943390302e-64

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

    from NiaPy.algorithms.basic import GreyWolfOptimizer

    # our custom benchmark classs
    class MyBenchmark(object):
        def __init__(self):
            # define lower bound of benchmark function
            self.Lower = -11
            # define upper bound of benchmark function
            self.Upper = 11

        # function which returns evaluate function
        def function(self):
            def evaluate(D, sol):
                val = 0.0
                for i in range(D):
                    val = val + sol[i] * sol[i]
                return val
            return evaluate


Now, all we have to do is to initialize our algorithm as in previous examples and pass as benchmark parameter,
instance of our *MyBenchmark* class.

.. code:: python

    for i in range(10):

        algorithm = GreyWolfOptimizer(10, 20, 10000, MyBenchmark())
        best = algorithm.run()

        print(best)

Now we can run our advanced example with following command python advanced_example.py. The results should be
similar to those bellow.

.. code:: bash

    1.99601075063e-63
    1.03831459307e-65
    6.76105610278e-63
    2.39738295065e-64
    1.11826744557e-46
    1.95914350691e-65
    6.33575259075e-58
    9.84100808621e-68
    2.62423542073e-66
    4.20503964752e-64

Runner example
--------------
For easier comparison between many different algorithms and benchmarks, we developed a useful feature called
*Runner*. Runner can take an array of algorithms and an array of benchmarks to compare and run all combinations
for you. We also provide an extra feature, which lets you easily exports those results in many different formats 
(LaTeX, Excell, JSON).

Below is given a usage example of our *Runner*, which will run three given algorithms and four given benchmark
functions. Results will be exported as JSON.

.. code:: python

    import NiaPy

    class MyBenchmark(object):
        def __init__(self):
            self.Lower = -5.12
            self.Upper = 5.12

        def function(self):
            def evaluate(D, sol):
                val = 0.0
                for i in range(D):
                    val = val + sol[i] * sol[i]
                return val
            return evaluate


    algorithms = ['DifferentialEvolutionAlgorithm',
                'ArtificialBeeColonyAlgorithm',
                'GreyWolfOptimizer']
    benchmarks = ['ackley', 'whitley', 'alpine2', MyBenchmark()]

    NiaPy.Runner(10, 40, 10000, 3, algorithms, benchmarks).run(export='json', verbose=True)


Output of running above example should look like something as following.

.. code:: bash

    Running DifferentialEvolutionAlgorithm...
    Running DifferentialEvolutionAlgorithm algorithm on ackley benchmark...
    Running DifferentialEvolutionAlgorithm algorithm on whitley benchmark...
    Running DifferentialEvolutionAlgorithm algorithm on alpine2 benchmark...
    Running DifferentialEvolutionAlgorithm algorithm on MyBenchmark benchmark...
    ---------------------------------------------------
    Running ArtificialBeeColonyAlgorithm...
    Running ArtificialBeeColonyAlgorithm algorithm on ackley benchmark...
    Running ArtificialBeeColonyAlgorithm algorithm on whitley benchmark...
    Running ArtificialBeeColonyAlgorithm algorithm on alpine2 benchmark...
    Running ArtificialBeeColonyAlgorithm algorithm on MyBenchmark benchmark...
    ---------------------------------------------------
    Running GreyWolfOptimizer...
    Running GreyWolfOptimizer algorithm on ackley benchmark...
    Running GreyWolfOptimizer algorithm on whitley benchmark...
    Running GreyWolfOptimizer algorithm on alpine2 benchmark...
    Running GreyWolfOptimizer algorithm on MyBenchmark benchmark...
    ---------------------------------------------------
    Export to JSON completed!

Results exported as JSON should look like this.

.. code:: json

    {
        "GreyWolfOptimizer": {
            "MyBenchmark": [
            6.766062076017854e-46,
            2.6426533581097554e-43,
            8.658015542865062e-44
            ],
            "ackley": [
            4.440892098500626e-16,
            4.440892098500626e-16,
            4.440892098500626e-16
            ],
            "whitley": [
            41.15672884009374,
            45.405829107898754,
            45.285854036223746
            ],
            "alpine2": [
            -334.17253174936184,
            -26.600888674701295,
            -214.48104063289853
            ]
        },
        "ArtificialBeeColonyAlgorithm": {
            "MyBenchmark": [
            1.381020772809769e-09,
            4.082544319484199e-09,
            2.5174669579239143e-11
            ],
            "ackley": [
            0.0001596817850928467,
            0.0017004800794961916,
            0.00018082865898749745
            ],
            "whitley": [
            20.622549664235308,
            14.085647205633876,
            1.838650658412531
            ],
            "alpine2": [
            -23686.224202267975,
            -23678.92101630358,
            -14320.040364388877
            ]
        },
        "DifferentialEvolutionAlgorithm": {
            "MyBenchmark": [
            1.692521623510217e-10,
            1.7135875905552047e-10,
            1.2860888219094234e-10
            ],
            "ackley": [
            0.00012939348497598147,
            0.00010798205896778157,
            0.00011202026154366607
            ],
            "whitley": [
            59.35951990376928,
            58.805393587160424,
            63.532977687055386
            ],
            "alpine2": [
            -23698.80535644514,
            -19925.409402805282,
            -23500.48062034027
            ]
        }
    }
