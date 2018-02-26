[![Unix Build Status](https://img.shields.io/travis/NiaOrg/NiaPy/master.svg)](https://travis-ci.org/NiaOrg/NiaPy)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/l5c0rp04mp04mbtq?svg=true)](https://ci.appveyor.com/project/GregaVrbancic/niapy)
[![Coverage Status](https://img.shields.io/coveralls/NiaOrg/NiaPy/master.svg)](https://coveralls.io/r/NiaOrg/NiaPy) [![Scrutinizer Code Quality](https://img.shields.io/scrutinizer/g/NiaOrg/NiaPy.svg)](https://scrutinizer-ci.com/g/NiaOrg/NiaPy/?branch=master)
[![PyPI Version](https://img.shields.io/pypi/v/NiaPy.svg)](https://pypi.python.org/pypi/NiaPy)
[![Documentation Status](https://readthedocs.org/projects/niapy/badge/?version=latest)](http://niapy.readthedocs.io/en/latest/?badge=latest)
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/NiaOrg/NiaPy.svg)](http://isitmaintained.com/project/NiaOrg/NiaPy "Average time to resolve an issue")
[![Percentage of issues still open](http://isitmaintained.com/badge/open/NiaOrg/NiaPy.svg)](http://isitmaintained.com/project/NiaOrg/NiaPy "Percentage of issues still open")
[![GitHub license](https://img.shields.io/github/license/NiaOrg/NiaPy.svg)](https://github.com/NiaOrg/NiaPy/blob/master/LICENSE)

## About
Nature-inspired algorithms are a very popular tool for solving optimization problems. Numerous variants of [nature-inspired algorithms have been developed](https://arxiv.org/abs/1307.4186) since the beginning of their era. To prove their versatility, those were tested in various domains on various applications, especially when they are hybridized, modified or adapted. However, implementation of nature-inspired algorithms is sometimes a difficult, complex and tedious task. In order to break this wall, NiaPy is intended for simple and quick use, without spending time for implementing algorithms from scratch.

<p align="center"><img src=".github/imgs/NiaPyLogo.png" alt="NiaPy" title="NiaPy"/></p>

## Mission
Our mission is to build a collection of nature-inspired algorithms and create a simple interface for managing the optimization process.
NiaPy will offer:


- numerous benchmark functions implementations,
- use of various nature-inspired algorithms without struggle and effort with a simple interface,
- easy comparison between nature-inspired algorithms and
- export of results in various formats (LaTeX, JSON, Excel).

# Overview

Python micro framework for building nature-inspired algorithms.

The micro framework features following algorithms:

- basic:
  - Artificial bee colony algorithm ([see example](examples/run_abc.py))
  - Bat algorithm ([see example](examples/run_ba.py))
  - Cuckoo Search algorithm ([see example](examples/run_cs.py))
  - Differential evolution algorithm ([see example](examples/run_de.py))
  - Firefly algorithm ([see example](examples/run_fa.py))
  - Flower pollination algorithm ([see example](examples/run_fpa.py))
  - Genetic algorithm ([see example](examples/run_ga.py))
  - Grey wolf optimizer ([see example](examples/run_gwo.py))
  - Particle swarm optimization ([see example](examples/run_pso.py))
- modified:
  - Hybrid bat algorithm ([see example](examples/run_hba.py))
  - Self-adaptive differential evolution algorithm ([see example](examples/run_jde.py))

The following benchmark functions are included in NiaPy:

- Ackley
- Alpine
  - Alpine1
  - Alpine2
- Chung Reynolds
- Csendes
- Griewank
- Happy cat
- Pintér
- Qing
- Quintic
- Rastrigin
- Ridge
- Rosenbrock
- Salomon
- Schumer Steiglitz
- Schwefel
  - Schwefel 2.21
  - Schwefel 2.22
- Sphere
- Step
  - Step2
  - Step3
- Stepint
- Styblinski-Tang
- Sum Squares
- Whitley


# Setup

## Requirements

* Python 3.6+ (backward compatibility with 2.7.14)
* Pip

## Installation

Install NiaPy with pip (will be available soon):

```sh
$ pip install NiaPy
```

or directly from the source code:

```sh
$ git clone https://github.com/NiaOrg/NiaPy.git
$ cd NiaPy
$ python setup.py install
```

# Usage

After installation, the package can imported:

```sh
$ python
>>> import NiaPy
>>> NiaPy.__version__
```

For more usage examples please look at **examples** folder.

## Contributing

[![Open Source Helpers](https://www.codetriage.com/niaorg/niapy/badges/users.svg)](https://www.codetriage.com/niaorg/niapy)

We encourage you to contribute to NiaPy! Please check out the [Contributing to NiaPy guide](CONTRIBUTING.md) for guidelines about how to proceed.

Everyone interacting in NiaPy's codebases, issue trackers, chat rooms and mailing lists is expected to follow the NiaPy [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

[<img alt="GregaVrbancic" src="https://avatars0.githubusercontent.com/u/1894788?v=4&s=117" width="117">](https://github.com/GregaVrbancic) |[<img alt="firefly-cpp" src="https://avatars2.githubusercontent.com/u/1633361?v=4&s=117" width="117">](https://github.com/firefly-cpp) |[<img alt="lucijabrezocnik" src="https://avatars2.githubusercontent.com/u/36370699?v=4&s=117" width="117">](https://github.com/lucijabrezocnik) |[<img alt="mlaky88" src="https://avatars1.githubusercontent.com/u/23091578?v=4&s=117" width="117">](https://github.com/mlaky88) |[<img alt="rhododendrom" src="https://avatars1.githubusercontent.com/u/3198785?v=4&s=117" width="117">](https://github.com/rhododendrom) |
:---: |:---: |:---: |:---: |:---: |
[GregaVrbancic](https://github.com/GregaVrbancic) |[firefly-cpp](https://github.com/firefly-cpp) |[lucijabrezocnik](https://github.com/lucijabrezocnik) |[mlaky88](https://github.com/mlaky88) |[rhododendrom](https://github.com/rhododendrom) |
## Licence
This package is distributed under the MIT License. This license can be found online at http://www.opensource.org/licenses/MIT.

## Disclaimer
This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it at your own risk!
