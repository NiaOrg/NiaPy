Installation
============

Setup development environment
-----------------------------

Requirements
~~~~~~~~~~~~

- Python: `download <https://www.python.org/downloads/>`__ (3.10 or greater)
- Poetry: `docs <https://python-poetry.org/docs/>`__


Installation of development dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install project dependencies into a virtual environment:

.. code-block:: bash

    poetry install

Install documentation dependencies with:

.. code-block:: bash

    poetry install --with docs

Run tests with:

.. code-block:: bash

    poetry run pytest

To enter created virtual environment with all installed development dependencies run:

.. code-block:: bash

    poetry shell
