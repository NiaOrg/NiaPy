Installation
============

Setup development environment
-----------------------------

Requirements
~~~~~~~~~~~~

- Python: `download <https://www.python.org/downloads/>`__ (3.9 or greater)
- Poetry: `docs <https://python-poetry.org/docs/>`__


Installation of development dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NiaPy dependencies:**

+-----------+-----------+----------+
| Package   | Version   | Platform |
+===========+===========+==========+
| numpy     | >=1.26.1  | All      |
+-----------+-----------+----------+
| matplotlib| >=3.8.0   | All      |
+-----------+-----------+----------+
| pandas    | >=2.1.1   | All      |
+-----------+-----------+----------+
| openpyxl  | >=3.1.2   | All      |
+-----------+-----------+----------+

Install project dependencies into a virtual environment:

.. code-block:: bash

    poetry install

Run tests with:

.. code-block:: bash

    poetry run pytest

To enter created virtual environment with all installed development dependencies run:

.. code-block:: bash

    poetry shell
