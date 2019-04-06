Installation
============

Setup development environment
-----------------------------

Requirements
~~~~~~~~~~~~

- Python: `download <https://www.python.org/downloads/>`_ (at least version 2.7.14, prefferable 3.6.x) 
- Pip: `installation docs <https://pip.pypa.io/en/stable/installing/>`_
- Make
    - Windows: `download <http://mingw.org/download/installer>`_ [:doc:`/guides/mingw_installation`] 
    - Mac: `download <http://developer.apple.com/xcode>`_
    - Linux: `download <http://www.gnu.org/software/make>`_
- pipenv: `docs <http://docs.pipenv.org>`_ (run ``pip install pipenv`` command)
- Pandoc: `installation docs <http://johnmacfarlane.net/pandoc/installing.html>`_ * optional
- Graphviz: `download <http://www.graphviz.org/Download.php>`_ * optional

To confirm these system dependencies are configured correctly:

.. code-block:: bash

    make doctor


Installation of development dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List of NiaPy's dependencies:

==========  =======  ======== 
Package     Version  Platform
==========  =======  ======== 
click       Any      All 
numpy       1.14.0   All 
scipy       1.0.0    All 
xlsxwriter  1.0.2    All 
matplotlib  *        All
==========  =======  ========

List of development dependencies:

=============================  =======  ======== 
Package                        Version  Platform
=============================  =======  ========
pylint                         Any      Any
pycodestyle                    Any      Any
pydocstyle                     Any      Any
pytest                         ~=3.3    Any
pytest-describe                Any      Any 
pytest-expecter                Any      Any
pytest-random                  Any      Any
pytest-cov                     Any      Any
freezegun                      Any      Any
coverage-space                 Any      Any
docutils                       Any      Any
pygments                       Any      Any
wheel                          Any      Any
pyinstaller                    Any      Any
twine                          Any      Any
sniffer                        Any      Any
macfsevents                    Any      darwin
enum34                         Any      Any
singledispatch                 Any      Any
backports.functools-lru-cache  Any      Any
configparser                   Any      Any
sphinx                         Any      Any
sphinx-rtd-theme               Any      Any
funcsigs                       Any      Any
futures                        Any      Any
autopep8                       Any      Any
sphinx-autobuild               Any      Any     
=============================  =======  ========

Install project dependencies into a virtual environment:

.. code-block:: bash

    make install

To enter created virtual environment with all installed development dependencies run: 

.. code-block:: bash

    pipenv shell