|PyPI Version|
|PyPI - Python Version|
|PyPI - Status|
|PyPI - Downloads|
|GitHub Release Date|
|Anaconda-Server Badge|
|Documentation Status|
|GitHub license|

|Scrutinizer Code Quality|
|Coverage Status|
|GitHub commit activity|
|Updates|
|Average time to resolve an issue|
|Percentage of issues still open|
|GitHub contributors|

|DOI zenodo|
|DOI JOSS|



About
=====

Nature-inspired algorithms are a very popular tool for solving
optimization problems. Numerous variants of `nature-inspired algorithms
have been developed <https://arxiv.org/abs/1307.4186>`__ since the
beginning of their era. To prove their versatility, those were tested in
various domains on various applications, especially when they are
hybridized, modified or adapted. However, implementation of
nature-inspired algorithms is sometimes a difficult, complex and tedious
task. In order to break this wall, NiaPy is intended for simple and
quick use, without spending time for implementing algorithms from
scratch.


.. image:: http://c1.staticflickr.com/5/4757/26625486258_41ea6d95e0.jpg
    :align: center

Mission
-------

| Our mission is to build a collection of nature-inspired algorithms and
  create a simple interface for managing the optimization process.
| NiaPy will offer:

-  numerous benchmark functions implementations,
-  use of various nature-inspired algorithms without struggle and effort
   with a simple interface,
-  easy comparison between nature-inspired algorithms and
-  export of results in various formats (LaTeX, JSON, Excel).

Overview
========

Python micro framework for building nature-inspired algorithms. Official documentation is available `here <https://niapy.readthedocs.io/en/stable/>`_.

The micro framework features following algorithms:

-  basic:
    -  Artificial bee colony algorithm
    -  Bat algorithm
    -  Camel algorithm
    -  Cuckoo search
    -  Differential evolution algorithm
    -  Evolution Strategy
    -  Firefly algorithm
    -  Fireworks algorithm
    -  Flower pollination algorithm
    -  Forest optimization algorithm
    -  Genetic algorithm
    -  Glowworm swarm optimization
    -  Grey wolf optimizer
    -  Monarch butterfly optimization
    -  Moth flame optimizer
    -  Harmony Search algorithm
    -  Krill herd algorithm
    -  Monkey king evolution
    -  Multiple trajectory search
    -  Particle swarm optimization
    -  Sine cosine algorithm
-  modified:
    -  Hybrid bat algorithm
    -  Self-adaptive differential evolution algorithm
    -  Dynamic population size self-adaptive differential evolution algorithm
-  other:
    -  Anarchic society optimization algorithm
    -  Hill climbing algorithm
    -  Multiple trajectory search
    -  Nelder mead method or downhill simplex method or amoeba method
    -  Simulated annealing algorithm

The following benchmark functions are included in NiaPy:

-  Ackley
-  Alpine
    -  Alpine1
    -  Alpine2
-  Bent Cigar
-  Chung Reynolds
-  Csendes
-  Discus
-  Dixon-Price
-  Elliptic
-  Griewank
-  Happy cat
-  HGBat
-  Katsuura
-  Levy
-  Michalewicz
-  Perm
-  Pintér
-  Powell
-  Qing
-  Quintic
-  Rastrigin
-  Ridge
-  Rosenbrock
-  Salomon
-  Schumer Steiglitz
-  Schwefel
    -  Schwefel 2.21
    -  Schwefel 2.22
-  Sphere
    -  Sphere2 -> Sphere with different powers
    -  Sphere3 -> Rotated hyper-ellipsoid
-  Step
    -  Step2
    -  Step3
-  Stepint
-  Styblinski-Tang
-  Sum Squares
-  Trid
-  Weierstrass
-  Whitley
-  Zakharov

Setup
=====

Requirements
------------

-  Python 3.6.x or 3.7.x (backward compatibility with 2.7.x)
-  Pip

Dependencies
~~~~~~~~~~~~

-  numpy >= 1.16.2
-  scipy >= 1.2.1
-  enum34 >= 1.1.6 (if using python version < 3.4)
-  xlsxwriter >= 1.1.6
-  matplotlib >= 2.2.4

List of development dependencies and requirements can be found in the `installation section of NiaPy documentation <http://niapy.readthedocs.io/en/stable/installation.html>`_.

Installation
------------

Install NiaPy with pip:

.. code:: sh

    $ pip install NiaPy

Install NiaPy with conda:

.. code:: sh

    $ conda install -c niaorg niapy

Or directly from the source code:

.. code:: sh

    $ git clone https://github.com/NiaOrg/NiaPy.git
    $ cd NiaPy
    $ python setup.py install

Usage
=====

After installation, the package can imported:

.. code:: sh

    $ python
    >>> import NiaPy
    >>> NiaPy.__version__

For more usage examples please look at **examples** folder.

Contributing
------------

|Open Source Helpers|

We encourage you to contribute to NiaPy! Please check out the
`Contributing to NiaPy guide <CONTRIBUTING.md>`__ for guidelines about
how to proceed.

Everyone interacting in NiaPy's codebases, issue trackers, chat rooms
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


.. |Scrutinizer Code Quality| image:: https://img.shields.io/scrutinizer/g/NiaOrg/NiaPy.svg
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


.. |DOI zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1205048.svg
   :target: https://doi.org/10.5281/zenodo.1205048
.. |DOI JOSS| image:: http://joss.theoj.org/papers/10.21105/joss.00613/status.svg
   :target: https://doi.org/10.21105/joss.00613

.. |Open Source Helpers| image:: https://www.codetriage.com/niaorg/niapy/badges/users.svg
   :target: https://www.codetriage.com/niaorg/niapy