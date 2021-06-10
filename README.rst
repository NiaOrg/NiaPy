.. image:: https://raw.githubusercontent.com/NiaOrg/NiaPy/master/.github/imgs/NiaPyLogo.png
    :align: center

--------------

|Check codestyle and test build| |PyPI Version| |PyPI - Python Version|
|PyPI - Status| |PyPI - Downloads| |GitHub Release Date|
|Anaconda-Server Badge| |Documentation Status| |GitHub license|

|Scrutinizer Code Quality| |Coverage Status| |GitHub commit activity|
|Updates| |Average time to resolve an issue| |Percentage of issues still
open| |GitHub contributors|

|DOI| |image1|

Nature-inspired algorithms are a very popular tool for solving
optimization problems. Numerous variants of nature-inspired algorithms
have been developed (`paper 1 <https://arxiv.org/abs/1307.4186>`__,
`paper 2 <https://www.mdpi.com/2076-3417/8/9/1521>`__) since the
beginning of their era. To prove their versatility, those were tested in
various domains on various applications, especially when they are
hybridized, modified or adapted. However, implementation of
nature-inspired algorithms is sometimes a difficult, complex and tedious
task. In order to break this wall, NiaPy is intended for simple and
quick use, without spending time for implementing algorithms from
scratch.

-  **Free software:** MIT license
-  **Documentation:** https://niapy.readthedocs.io/en/stable/
-  **Python versions:** 3.6.x, 3.7.x, 3.8.x, 3.9.x
-  **Dependencies:** `click
   here <CONTRIBUTING.md#development-dependencies>`__

Mission
=======

Our mission is to build a collection of nature-inspired algorithms and
create a simple interface for managing the optimization process. NiaPy
offers:

-  numerous optimization problem implementations,
-  use of various nature-inspired algorithms without struggle and effort
   with a simple interface,
-  easy comparison between nature-inspired algorithms, and
-  export of results in various formats such as Pandas DataFrame, JSON
   or even Excel (only when using Python >= 3.6).

Installation
============

Install NiaPy with pip:

Latest version (2.0.0rc17)
--------------------------

.. code:: sh

   $ pip install niapy==2.0.0rc17

To install NiaPy with conda, use:

.. code:: sh

   $ conda install -c niaorg niapy=2.0.0rc17

Latest stable version
---------------------

.. code:: sh

   $ pip install niapy

To install NiaPy with conda, use:

.. code:: sh

   $ conda install -c niaorg niapy

To install NiaPy on Fedora, use:

.. code:: sh

   $ dnf install python3-niapy

Install from source
-------------------

In case you want to install directly from the source code, use:

.. code:: sh

   $ git clone https://github.com/NiaOrg/NiaPy.git
   $ cd NiaPy
   $ python setup.py install

Usage
=====

After installation, you can import NiaPy as any other Python module:

.. code:: sh

   $ python
   >>> import niapy
   >>> niapy.__version__

Let’s go through a basic and advanced example.

Basic Example
-------------

Let’s say, we want to try out Gray Wolf Optimizer algorithm against the
Pintér problem. Firstly, we have to create a new file, called *basic_example.py*.
Then we have to import chosen algorithm from NiaPy, so we can use it. Afterwards we initialize
GreyWolfOptimizer class instance and run the algorithm. Given bellow is the complete source code of basic example.

.. code:: python

    from niapy.algorithms.basic import GreyWolfOptimizer
    from niapy.task import Task

    # we will run 10 repetitions of Grey Wolf Optimizer against the Pinter problem
    for i in range(10):
        task = Task(problem='pinter', dimension=10, max_evals=1000)
        algorithm = GreyWolfOptimizer(population_size=20)
        best = algorithm.run(task)
        print(best[-1])

Given example can be run with *python basic_example.py* command and
should give you similar output as following:

.. code:: sh

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

Advanced Example
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
    from niapy.algorithms.basic import GreyWolfOptimizer

    # our custom Problem class
    class MyProblem(Problem):
        def __init__(self, dimension, lower=-10, upper=10, *args, **kwargs):
            super().__init__(dimension, lower, upper, *args, **kwargs)

        def _evaluate(self, x):
            return np.sum(x ** 2)

Now, all we have to do is to initialize our algorithm as in previous examples and pass an instance of our MyProblem class as the problem argument.

.. code:: python

    my_problem = MyProblem(dimension=20)
    for i in range(10):
        task = Task(problem=my_problem, max_iters=100)
        algo = GreyWolfOptimizer(population_size=20)

        # running algorithm returns best found minimum
        best = algo.run(task)

        # printing best minimum
        print(best[-1])

Now we can run our advanced example with following command: *python
advanced_example.py*. The results should be similar to those bellow.

.. code:: sh

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

For more usage examples please look at `examples </examples>`__ folder.

More advanced examples can also be found in the `NiaPy-examples
repository <https://github.com/NiaOrg/NiaPy-examples>`__.

Cite us
=======

Are you using NiaPy in your project or research? Please cite us!

Plain format
------------

::

         Vrbančič, G., Brezočnik, L., Mlakar, U., Fister, D., & Fister Jr., I. (2018).
         NiaPy: Python microframework for building nature-inspired algorithms.
         Journal of Open Source Software, 3(23), 613\. <https://doi.org/10.21105/joss.00613>

Bibtex format
-------------

::

       @article{NiaPyJOSS2018,
           author  = {Vrban{\v{c}}i{\v{c}}, Grega and Brezo{\v{c}}nik, Lucija
                     and Mlakar, Uro{\v{s}} and Fister, Du{\v{s}}an and {Fister Jr.}, Iztok},
           title   = {{NiaPy: Python microframework for building nature-inspired algorithms}},
           journal = {{Journal of Open Source Software}},
           year    = {2018},
           volume  = {3},
           issue   = {23},
           issn    = {2475-9066},
           doi     = {10.21105/joss.00613},
           url     = {https://doi.org/10.21105/joss.00613}
       }

RIS format
----------

::

       TY  - JOUR
       T1  - NiaPy: Python microframework for building nature-inspired algorithms
       AU  - Vrbančič, Grega
       AU  - Brezočnik, Lucija
       AU  - Mlakar, Uroš
       AU  - Fister, Dušan
       AU  - Fister Jr., Iztok
       PY  - 2018
       JF  - Journal of Open Source Software
       VL  - 3
       IS  - 23
       DO  - 10.21105/joss.00613
       UR  - http://joss.theoj.org/papers/10.21105/joss.00613


Contributing
------------

|Open Source Helpers|

We encourage you to contribute to NiaPy! Please check out the
`Contributing to NiaPy guide <CONTRIBUTING.md>`__ for guidelines about
how to proceed.

Everyone interacting in NiaPy’s codebases, issue trackers, chat rooms
and mailing lists is expected to follow the NiaPy `code of
conduct <CODE_OF_CONDUCT.md>`__.

Licence
-------

This package is distributed under the MIT License. This license can be
found online at http://www.opensource.org/licenses/MIT.

Disclaimer
----------

This framework is provided as-is, and there are no guarantees that it
fits your purposes or that it is bug-free. Use it at your own risk!

.. |Check codestyle and test build| image:: https://github.com/NiaOrg/NiaPy/workflows/Check%20and%20Test/badge.svg
.. |PyPI Version| image:: https://img.shields.io/pypi/v/NiaPy.svg
   :target: https://pypi.python.org/pypi/NiaPy
.. |PyPI - Python Version| image:: https://img.shields.io/pypi/pyversions/NiaPy.svg
.. |PyPI - Status| image:: https://img.shields.io/pypi/status/NiaPy.svg
.. |PyPI - Downloads| image:: https://img.shields.io/pypi/dm/NiaPy.svg
.. |GitHub Release Date| image:: https://img.shields.io/github/release-date/NiaOrg/NiaPy.svg
.. |Anaconda-Server Badge| image:: https://anaconda.org/niaorg/niapy/badges/installer/conda.svg
   :target: https://conda.anaconda.org/niaorg
.. |Documentation Status| image:: https://readthedocs.org/projects/niapy/badge/?version=latest
   :target: http://niapy.readthedocs.io/en/latest/?badge=latest
.. |GitHub license| image:: https://img.shields.io/github/license/NiaOrg/NiaPy.svg
   :target: https://github.com/NiaOrg/NiaPy/blob/master/LICENSE
.. |Scrutinizer Code Quality| image:: https://scrutinizer-ci.com/g/NiaOrg/NiaPy/badges/quality-score.png?b=master
   :target: https://scrutinizer-ci.com/g/NiaOrg/NiaPy/?branch=master
.. |Coverage Status| image:: https://img.shields.io/coveralls/NiaOrg/NiaPy/master.svg
   :target: https://coveralls.io/r/NiaOrg/NiaPy
.. |GitHub commit activity| image:: https://img.shields.io/github/commit-activity/w/NiaOrg/NiaPy.svg
.. |Updates| image:: https://pyup.io/repos/github/NiaOrg/NiaPy/shield.svg
   :target: https://pyup.io/repos/github/NiaOrg/NiaPy/
.. |Average time to resolve an issue| image:: http://isitmaintained.com/badge/resolution/NiaOrg/NiaPy.svg
   :target: http://isitmaintained.com/project/NiaOrg/NiaPy
.. |Percentage of issues still open| image:: http://isitmaintained.com/badge/open/NiaOrg/NiaPy.svg
   :target: http://isitmaintained.com/project/NiaOrg/NiaPy
.. |GitHub contributors| image:: https://img.shields.io/github/contributors/NiaOrg/NiaPy.svg
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1205048.svg
   :target: https://doi.org/10.5281/zenodo.1205048
.. |image1| image:: http://joss.theoj.org/papers/10.21105/joss.00613/status.svg
   :target: https://doi.org/10.21105/joss.00613
.. |Open Source Helpers| image:: https://www.codetriage.com/niaorg/niapy/badges/users.svg
   :target: https://www.codetriage.com/niaorg/niapy
