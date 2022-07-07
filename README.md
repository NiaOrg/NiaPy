<p align="center"><img src=".github/imgs/NiaPyLogo.png" alt="NiaPy" title="NiaPy"/></p>

---



![Check codestyle and test build](https://github.com/NiaOrg/NiaPy/workflows/Check%20and%20Test/badge.svg)
[![PyPI Version](https://img.shields.io/pypi/v/NiaPy.svg)](https://pypi.python.org/pypi/NiaPy)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/NiaPy.svg)
![PyPI - Status](https://img.shields.io/pypi/status/NiaPy.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/NiaPy.svg)
[![Downloads](https://pepy.tech/badge/niapy)](https://pepy.tech/project/niapy)
![GitHub Release Date](https://img.shields.io/github/release-date/NiaOrg/NiaPy.svg)
[![Anaconda-Server Badge](https://anaconda.org/niaorg/niapy/badges/installer/conda.svg)](https://conda.anaconda.org/niaorg)
[![Fedora package](https://img.shields.io/fedora/v/python3-niapy?color=blue&label=Fedora%20Linux&logo=fedora)](https://src.fedoraproject.org/rpms/python-niapy)
[![AUR package](https://img.shields.io/aur/version/python-niapy?color=blue&label=Arch%20Linux&logo=arch-linux)](https://aur.archlinux.org/packages/python-niapy)
[![Documentation Status](https://readthedocs.org/projects/niapy/badge/?version=latest)](http://niapy.readthedocs.io/en/latest/?badge=latest)
[![GitHub license](https://img.shields.io/github/license/NiaOrg/NiaPy.svg)](https://github.com/NiaOrg/NiaPy/blob/master/LICENSE)

[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/NiaOrg/NiaPy/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/NiaOrg/NiaPy/?branch=master)
[![Coverage Status](https://img.shields.io/coveralls/NiaOrg/NiaPy/master.svg)](https://coveralls.io/r/NiaOrg/NiaPy)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/NiaOrg/NiaPy.svg)
[![Updates](https://pyup.io/repos/github/NiaOrg/NiaPy/shield.svg)](https://pyup.io/repos/github/NiaOrg/NiaPy/)
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/NiaOrg/NiaPy.svg)](http://isitmaintained.com/project/NiaOrg/NiaPy "Average time to resolve an issue")
[![Percentage of issues still open](http://isitmaintained.com/badge/open/NiaOrg/NiaPy.svg)](http://isitmaintained.com/project/NiaOrg/NiaPy "Percentage of issues still open")
![GitHub contributors](https://img.shields.io/github/contributors/NiaOrg/NiaPy.svg)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1205048.svg)](https://doi.org/10.5281/zenodo.1205048)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00613/status.svg)](https://doi.org/10.21105/joss.00613)


Nature-inspired algorithms are a very popular tool for solving optimization problems. Numerous variants of nature-inspired algorithms have been developed ([paper 1](https://arxiv.org/abs/1307.4186), [paper 2](https://www.mdpi.com/2076-3417/8/9/1521)) since the beginning of their era. To prove their versatility, those were tested in various domains on various applications, especially when they are hybridized, modified or adapted. However, implementation of nature-inspired algorithms is sometimes a difficult, complex and tedious task. In order to break this wall, NiaPy is intended for simple and quick use, without spending time for implementing algorithms from scratch.

* **Free software:** MIT license
* **Documentation:** https://niapy.readthedocs.io/en/stable/
* **Python versions:** 3.6.x, 3.7.x, 3.8.x, 3.9.x
* **Dependencies:** [click here](CONTRIBUTING.md#development-dependencies)

# Mission

Our mission is to build a collection of nature-inspired algorithms and create a simple interface for managing the optimization process. NiaPy offers:

- numerous optimization problem implementations,
- use of various nature-inspired algorithms without struggle and effort with a simple interface,
- easy comparison between nature-inspired algorithms, and
- export of results in various formats such as Pandas DataFrame, JSON or even Excel.


# Installation

Install NiaPy with pip:

```sh
$ pip install niapy
```

To install NiaPy with conda, use:

```sh
$ conda install -c niaorg niapy
```

To install NiaPy on Fedora, use:

```sh
$ dnf install python3-niapy
```

To install NiaPy on Arch Linux, please use an [AUR helper](https://wiki.archlinux.org/title/AUR_helpers):

```sh
$ yay -Syyu python-niapy
```

To install NiaPy on Alpine Linux, please enable Community repository and use:

```sh
$ apk add py3-niapy
```

To install NiaPy on NixOS, please use:

```sh
$ nix-env -iA nixos.python310Packages.niapy
```

## Install from source

In case you want to install directly from the source code, use:

```sh
$ git clone https://github.com/NiaOrg/NiaPy.git
$ cd NiaPy
$ python setup.py install
```

# Algorithms

[Click here](Algorithms.md) for the list of implemented algorithms.

# Problems

[Click here](Problems.md) for the list of implemented test problems.

# Usage

After installation, you can import NiaPy as any other Python module:

```sh
$ python
>>> import niapy
>>> niapy.__version__
```


Let's go through a basic and advanced example.

## Basic Example
Let’s say, we want to try out PSO against the Pintér problem function. Firstly, we have to create new file, with name, for example *basic_example.py*. Then we have to import chosen algorithm from NiaPy, so we can use it. Afterwards we initialize ParticleSwarmAlgorithm class instance and run the algorithm. Given bellow is the complete source code of basic example.

```python
from niapy.algorithms.basic import ParticleSwarmAlgorithm
from niapy.task import Task

# we will run 10 repetitions of Weighted, velocity clamped PSO on the Pinter problem
for i in range(10):
    task = Task(problem='pinter', dimension=10, max_evals=10000)
    algorithm = ParticleSwarmAlgorithm(population_size=100, w=0.9, c1=0.5, c2=0.3, min_velocity=-1, max_velocity=1)
    best_x, best_fit = algorithm.run(task)
    print(best_fit)
```

Given example can be run with *python basic_example.py* command and should give you similar output as following:

```sh
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
```

## Advanced Example

In this example we will show you how to implement a custom problem class and use it with any of
implemented algorithms. First let's create new file named advanced_example.py. As in the previous examples
we wil import algorithm we want to use from niapy module.

For our custom optimization function, we have to create new class. Let's name it *MyProblem*. In the initialization
method of *MyProblem* class we have to set the *dimension*, *lower* and *upper* bounds of the problem. Afterwards we have to
override the abstract method _evaluate which takes a parameter *x*, the solution to be evaluated, and returns the function value.
Now we should have something similar as is shown in code snippet bellow.

```python
import numpy as np
from niapy.task import Task
from niapy.problems import Problem
from niapy.algorithms.basic import ParticleSwarmAlgorithm


# our custom problem class
class MyProblem(Problem):
    def __init__(self, dimension, lower=-10, upper=10, *args, **kwargs):
        super().__init__(dimension, lower, upper, *args, **kwargs)

    def _evaluate(self, x):
        return np.sum(x ** 2)
```

Now, all we have to do is to initialize our algorithm as in previous examples and pass an instance of our MyProblem class as the problem argument.

```python
my_problem = MyProblem(dimension=20)
for i in range(10):
    task = Task(problem=my_problem, max_iters=100)
    algo = ParticleSwarmAlgorithm(population_size=100, w=0.9, c1=0.5, c2=0.3, min_velocity=-1, max_velocity=1)

    # running algorithm returns best found minimum
    best_x, best_fit = algo.run(task)
    # printing best minimum
    print(best_fit)
```

Now we can run our advanced example with following command: *python advanced_example.py*. The results should be similar to those bellow.

```sh
0.002455614050761476
0.000557652972392164
0.0029791325679865413
0.0009443595274525336
0.001012658824492069
0.0006837236892816072
0.0026789725774685495
0.005017746993004601
0.0011654473402322196
0.0019074442166293853
```

For more usage examples please look at [examples](/examples) folder.

More advanced examples can also be found in the [NiaPy-examples repository](https://github.com/NiaOrg/NiaPy-examples).



# Cite us

Are you using NiaPy in your project or research? Please cite us!

## Plain format

```
      Vrbančič, G., Brezočnik, L., Mlakar, U., Fister, D., & Fister Jr., I. (2018).
      NiaPy: Python microframework for building nature-inspired algorithms.
      Journal of Open Source Software, 3(23), 613\. <https://doi.org/10.21105/joss.00613>
```

## Bibtex format

```
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
```

## RIS format

```
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
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/GregaVrbancic"><img src="https://avatars0.githubusercontent.com/u/1894788?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Grega Vrbančič</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=GregaVrbancic" title="Code">💻</a> <a href="https://github.com/NiaOrg/NiaPy/commits?author=GregaVrbancic" title="Documentation">📖</a> <a href="https://github.com/NiaOrg/NiaPy/issues?q=author%3AGregaVrbancic" title="Bug reports">🐛</a> <a href="#example-GregaVrbancic" title="Examples">💡</a> <a href="#maintenance-GregaVrbancic" title="Maintenance">🚧</a> <a href="#platform-GregaVrbancic" title="Packaging/porting to new platform">📦</a> <a href="#projectManagement-GregaVrbancic" title="Project Management">📆</a> <a href="https://github.com/NiaOrg/NiaPy/pulls?q=is%3Apr+reviewed-by%3AGregaVrbancic" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="https://github.com/firefly-cpp"><img src="https://avatars2.githubusercontent.com/u/1633361?v=4?s=100" width="100px;" alt=""/><br /><sub><b>firefly-cpp</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=firefly-cpp" title="Code">💻</a> <a href="https://github.com/NiaOrg/NiaPy/commits?author=firefly-cpp" title="Documentation">📖</a> <a href="https://github.com/NiaOrg/NiaPy/issues?q=author%3Afirefly-cpp" title="Bug reports">🐛</a> <a href="#example-firefly-cpp" title="Examples">💡</a> <a href="https://github.com/NiaOrg/NiaPy/pulls?q=is%3Apr+reviewed-by%3Afirefly-cpp" title="Reviewed Pull Requests">👀</a> <a href="#question-firefly-cpp" title="Answering Questions">💬</a> <a href="https://github.com/NiaOrg/NiaPy/commits?author=firefly-cpp" title="Tests">⚠️</a> <a href="#platform-firefly-cpp" title="Packaging/porting to new platform">📦</a></td>
    <td align="center"><a href="https://github.com/lucijabrezocnik"><img src="https://avatars2.githubusercontent.com/u/36370699?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Lucija Brezočnik</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=lucijabrezocnik" title="Code">💻</a> <a href="https://github.com/NiaOrg/NiaPy/commits?author=lucijabrezocnik" title="Documentation">📖</a> <a href="https://github.com/NiaOrg/NiaPy/issues?q=author%3Alucijabrezocnik" title="Bug reports">🐛</a> <a href="#example-lucijabrezocnik" title="Examples">💡</a></td>
    <td align="center"><a href="https://github.com/mlaky88"><img src="https://avatars1.githubusercontent.com/u/23091578?v=4?s=100" width="100px;" alt=""/><br /><sub><b>mlaky88</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=mlaky88" title="Code">💻</a> <a href="https://github.com/NiaOrg/NiaPy/commits?author=mlaky88" title="Documentation">📖</a> <a href="#example-mlaky88" title="Examples">💡</a></td>
    <td align="center"><a href="https://github.com/rhododendrom"><img src="https://avatars1.githubusercontent.com/u/3198785?v=4?s=100" width="100px;" alt=""/><br /><sub><b>rhododendrom</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=rhododendrom" title="Code">💻</a> <a href="https://github.com/NiaOrg/NiaPy/commits?author=rhododendrom" title="Documentation">📖</a> <a href="#example-rhododendrom" title="Examples">💡</a> <a href="https://github.com/NiaOrg/NiaPy/issues?q=author%3Arhododendrom" title="Bug reports">🐛</a> <a href="https://github.com/NiaOrg/NiaPy/pulls?q=is%3Apr+reviewed-by%3Arhododendrom" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="https://github.com/kb2623"><img src="https://avatars3.githubusercontent.com/u/7480221?s=460&v=4?s=100" width="100px;" alt=""/><br /><sub><b>Klemen</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=kb2623" title="Code">💻</a> <a href="https://github.com/NiaOrg/NiaPy/commits?author=kb2623" title="Documentation">📖</a> <a href="#example-kb2623" title="Examples">💡</a> <a href="https://github.com/NiaOrg/NiaPy/issues?q=author%3Akb2623" title="Bug reports">🐛</a> <a href="https://github.com/NiaOrg/NiaPy/pulls?q=is%3Apr+reviewed-by%3Akb2623" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="https://github.com/flyzoor"><img src="https://avatars2.githubusercontent.com/u/38717032?s=40&v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jan Popič</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=flyzoor" title="Code">💻</a> <a href="https://github.com/NiaOrg/NiaPy/commits?author=flyzoor" title="Documentation">📖</a> <a href="#example-flyzoor" title="Examples">💡</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/lukapecnik"><img src="https://avatars1.githubusercontent.com/u/23029992?s=460&v=4?s=100" width="100px;" alt=""/><br /><sub><b>Luka Pečnik</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=lukapecnik" title="Code">💻</a> <a href="https://github.com/NiaOrg/NiaPy/commits?author=lukapecnik" title="Documentation">📖</a> <a href="#example-lukapecnik" title="Examples">💡</a> <a href="https://github.com/NiaOrg/NiaPy/issues?q=author%3Alukapecnik" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/bankojan"><img src="https://avatars3.githubusercontent.com/u/44372016?s=460&v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jan Banko</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=bankojan" title="Code">💻</a> <a href="https://github.com/NiaOrg/NiaPy/commits?author=bankojan" title="Documentation">📖</a> <a href="#example-bankojan" title="Examples">💡</a></td>
    <td align="center"><a href="https://github.com/RokPot"><img src="https://avatars0.githubusercontent.com/u/23029990?s=460&v=4?s=100" width="100px;" alt=""/><br /><sub><b>RokPot</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=RokPot" title="Code">💻</a> <a href="https://github.com/NiaOrg/NiaPy/commits?author=RokPot" title="Documentation">📖</a> <a href="#example-RokPot" title="Examples">💡</a></td>
    <td align="center"><a href="https://github.com/mihael-mika"><img src="https://avatars2.githubusercontent.com/u/22932805?s=460&v=4?s=100" width="100px;" alt=""/><br /><sub><b>mihaelmika</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=mihael-mika" title="Code">💻</a> <a href="https://github.com/NiaOrg/NiaPy/commits?author=mihael-mika" title="Documentation">📖</a> <a href="#example-mihael-mika" title="Examples">💡</a></td>
    <td align="center"><a href="https://github.com/jacebrowning"><img src="https://avatars1.githubusercontent.com/u/939501?s=460&v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jace Browning</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=jacebrowning" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/musawakiliML"><img src="https://avatars1.githubusercontent.com/u/19978292?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Musa Adamu Wakili</b></sub></a><br /><a href="#question-musawakiliML" title="Answering Questions">💬</a></td>
    <td align="center"><a href="http://www.uni-kassel.de/eecs/en/faculties/e2n/staff/florian-schaefer.html"><img src="https://avatars2.githubusercontent.com/u/23655422?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Florian Schaefer</b></sub></a><br /><a href="#ideas-FlorianShepherd" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
  <tr>
    <td align="center"><a href="http://www.jhmenke.de"><img src="https://avatars0.githubusercontent.com/u/25080218?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jan-Hendrik Menke</b></sub></a><br /><a href="#question-jhmenke" title="Answering Questions">💬</a></td>
    <td align="center"><a href="https://github.com/brett18618"><img src="https://avatars2.githubusercontent.com/u/44141573?v=4?s=100" width="100px;" alt=""/><br /><sub><b>brett18618</b></sub></a><br /><a href="#question-brett18618" title="Answering Questions">💬</a></td>
    <td align="center"><a href="http://timzatko.eu"><img src="https://avatars2.githubusercontent.com/u/11925394?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Timotej Zaťko</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/issues?q=author%3Atimzatko" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/sisco0"><img src="https://avatars0.githubusercontent.com/u/25695302?v=4?s=100" width="100px;" alt=""/><br /><sub><b>sisco0</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=sisco0" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/zStupan"><img src="https://avatars.githubusercontent.com/u/48752988?v=4?s=100" width="100px;" alt=""/><br /><sub><b>zStupan</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=zStupan" title="Code">💻</a> <a href="https://github.com/NiaOrg/NiaPy/issues?q=author%3AzStupan" title="Bug reports">🐛</a> <a href="https://github.com/NiaOrg/NiaPy/commits?author=zStupan" title="Documentation">📖</a> <a href="#example-zStupan" title="Examples">💡</a> <a href="https://github.com/NiaOrg/NiaPy/commits?author=zStupan" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/hrnciar"><img src="https://avatars.githubusercontent.com/u/13086088?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Tomáš Hrnčiar</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=hrnciar" title="Code">💻</a></td>
    <td align="center"><a href="https://bandism.net/"><img src="https://avatars.githubusercontent.com/u/22633385?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ikko Ashimine</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=eltociear" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/andrazperson"><img src="https://avatars.githubusercontent.com/u/32928199?v=4?s=100" width="100px;" alt=""/><br /><sub><b>andrazperson</b></sub></a><br /><a href="https://github.com/NiaOrg/NiaPy/commits?author=andrazperson" title="Code">💻</a></td>
    <td align="center"><a href="http://carlosal1015.github.io"><img src="https://avatars.githubusercontent.com/u/21283014?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Oromion</b></sub></a><br /><a href="#platform-carlosal1015" title="Packaging/porting to new platform">📦</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!

## Contributing

[![Open Source Helpers](https://www.codetriage.com/niaorg/niapy/badges/users.svg)](https://www.codetriage.com/niaorg/niapy)

We encourage you to contribute to NiaPy! Please check out the [Contributing to NiaPy guide](CONTRIBUTING.md) for guidelines about how to proceed.

Everyone interacting in NiaPy's codebases, issue trackers, chat rooms and mailing lists is expected to follow the NiaPy [code of conduct](CODE_OF_CONDUCT.md).

## Licence

This package is distributed under the MIT License. This license can be found online at <http://www.opensource.org/licenses/MIT>.

## Disclaimer

This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it at your own risk!
