Installation
============

Setup development environment
-----------------------------

Requirements
~~~~~~~~~~~~

- Python: `download <https://www.python.org/downloads/>`__ (3.6 or greater)
- Pip: `installation docs <https://pip.pypa.io/en/stable/installing/>`__
- Make
    - Windows: `download <http://mingw.org/download/installer>`__ [:doc:`/guides/mingw_installation`]
    - Mac: `download <http://developer.apple.com/xcode>`__
    - Linux: `download <http://www.gnu.org/software/make>`__
- pipenv: `docs <http://docs.pipenv.org>`__ (run ``pip install pipenv`` command)
- Pandoc: `installation docs <http://johnmacfarlane.net/pandoc/installing.html>`__ * optional
- Graphviz: `download <http://www.graphviz.org/Download.php>`__ * optional

To confirm these system dependencies are configured correctly:

.. code-block:: bash

    make doctor


Installation of development dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List of NiaPy's dependencies:

==========  ========  ===================
Package     Version   Platform
==========  ========  ===================
numpy       >=1.16.2   All
scipy       >=1.1.1    All
pandas      >=0.24.2   All
matplotlib  >=2.2.4    All
openpyxl    ==3.0.3    All
xlwt        ==1.3.0    All
enum34      >=1.1.6    All: python < 3.4
future      >=0.18.2   All: python < 3
==========  ========  ===================

Install project dependencies into a virtual environment:

.. code-block:: bash

    make install

Run tests with:

.. code-block:: bash

    make test

To enter created virtual environment with all installed development dependencies run:

.. code-block:: bash

    pipenv shell
