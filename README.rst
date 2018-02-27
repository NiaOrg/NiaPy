|Unix Build Status|
|Windows Build status|
|Coverage Status|
|Scrutinizer Code Quality|
|PyPI Version|
|Documentation Status|
|GitHub license|

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

Python micro framework for building nature-inspired algorithms.

The micro framework features following algorithms:

-  basic:
-  Artificial bee colony algorithm (`see
   example <examples/run_abc.py>`__)
-  Bat algorithm (`see example <examples/run_ba.py>`__)
-  Cuckoo Search algorithm (`see example <examples/run_cs.py>`__)
-  Differential evolution algorithm (`see
   example <examples/run_de.py>`__)
-  Firefly algorithm (`see example <examples/run_fa.py>`__)
-  Flower pollination algorithm (`see example <examples/run_fpa.py>`__)
-  Genetic algorithm (`see example <examples/run_ga.py>`__)
-  Grey wolf optimizer (`see example <examples/run_gwo.py>`__)
-  Particle swarm optimization (`see example <examples/run_pso.py>`__)
-  modified:
-  Hybrid bat algorithm (`see example <examples/run_hba.py>`__)
-  Self-adaptive differential evolution algorithm (`see
   example <examples/run_jde.py>`__)

The following benchmark functions are included in NiaPy:

-  Ackley
-  Alpine
-  Alpine1
-  Alpine2
-  Chung Reynolds
-  Csendes
-  Griewank
-  Happy cat
-  Pintér
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
-  Step
-  Step2
-  Step3
-  Stepint
-  Styblinski-Tang
-  Sum Squares
-  Whitley

Setup
=====

Requirements
------------

-  Python 3.6+ (backward compatibility with 2.7.14)
-  Pip

Installation
------------

Install NiaPy with pip (will be available soon):

.. code:: sh

    $ pip install NiaPy

or directly from the source code:

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



Revision History
================

0.1.3a2
-------
-  fixes PyPI project description style

0.1.3a1
-------
-  fixes image issue in PyPI project description

0.1.2a4
-------
-  fixes problem with build scripts

0.1.2a3
-------
-  fixes PyPI project description
-  alpha3 version


.. |Unix Build Status| image:: https://img.shields.io/travis/NiaOrg/NiaPy/master.svg
   :target: https://travis-ci.org/NiaOrg/NiaPy
.. |Windows Build status| image:: https://ci.appveyor.com/api/projects/status/l5c0rp04mp04mbtq?svg=true
   :target: https://ci.appveyor.com/project/GregaVrbancic/niapy
.. |Coverage Status| image:: https://img.shields.io/coveralls/NiaOrg/NiaPy/master.svg
   :target: https://coveralls.io/r/NiaOrg/NiaPy
.. |Scrutinizer Code Quality| image:: https://img.shields.io/scrutinizer/g/NiaOrg/NiaPy.svg
   :target: https://scrutinizer-ci.com/g/NiaOrg/NiaPy/?branch=master
.. |PyPI Version| image:: https://img.shields.io/pypi/v/NiaPy.svg
   :target: https://pypi.python.org/pypi/NiaPy
.. |Documentation Status| image:: https://readthedocs.org/projects/niapy/badge/?version=latest
   :target: http://niapy.readthedocs.io/en/latest/?badge=latest
.. |Average time to resolve an issue| image:: http://isitmaintained.com/badge/resolution/NiaOrg/NiaPy.svg
   :target: http://isitmaintained.com/project/NiaOrg/NiaPy
.. |Percentage of issues still open| image:: http://isitmaintained.com/badge/open/NiaOrg/NiaPy.svg
   :target: http://isitmaintained.com/project/NiaOrg/NiaPy
.. |GitHub license| image:: https://img.shields.io/github/license/NiaOrg/NiaPy.svg
   :target: https://github.com/NiaOrg/NiaPy/blob/master/LICENSE
.. |Open Source Helpers| image:: https://www.codetriage.com/niaorg/niapy/badges/users.svg
   :target: https://www.codetriage.com/niaorg/niapy

