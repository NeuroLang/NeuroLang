Installing NeuroLang
====================

NeuroLang requires **Python ≥ 3.12**. The recommended way to install it is
with `uv <https://github.com/astral-sh/uv>`_ — a fast Python package manager —
or with standard ``pip``.

.. important::

   The latest release published on PyPI (``0.1a19``, October 2023) predates
   the SQUALL controlled natural language, the ``neurolang-query`` command
   line tool and the engine registry used throughout these tutorials.
   Until a new release is published, install NeuroLang **from GitHub**
   as shown below.

.. tab-set::

   .. tab-item:: Windows

      **1. Install uv**

      Open PowerShell and run:

      .. code-block:: powershell

         powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

      Or use pip directly if you already have Python installed:

      .. code-block:: text

         pip install uv

      **2. Install NeuroLang**

      .. code-block:: text

         uv pip install "neurolang @ git+https://github.com/NeuroLang/NeuroLang.git"

      **3. Verify**

      Open a new terminal and run:

      .. code-block:: text

         python -c "import neurolang; print('NeuroLang installed OK')"

   .. tab-item:: macOS

      **1. Install uv**

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

      **2. Install NeuroLang**

      .. code-block:: bash

         uv pip install "neurolang @ git+https://github.com/NeuroLang/NeuroLang.git"

      **3. Verify**

      .. code-block:: bash

         python -c "import neurolang; print('NeuroLang installed OK')"

   .. tab-item:: Linux

      **1. Install uv**

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

      **2. Install NeuroLang**

      .. code-block:: bash

         uv pip install "neurolang @ git+https://github.com/NeuroLang/NeuroLang.git"

      **3. Verify**

      .. code-block:: bash

         python -c "import neurolang; print('NeuroLang installed OK')"

   .. tab-item:: From source

      Clone the repository and install in editable mode with all dev and doc
      dependencies:

      .. code-block:: bash

         git clone https://github.com/NeuroLang/NeuroLang.git
         cd NeuroLang
         uv pip install -e ".[dev,doc]"

      To update your local copy:

      .. code-block:: bash

         git pull
         uv pip install -e ".[dev,doc]"

      Verify:

      .. code-block:: bash

         python -c "import neurolang; print('NeuroLang installed OK')"


Using pip without uv
---------------------

If you prefer plain pip:

.. code-block:: bash

   pip install "neurolang @ git+https://github.com/NeuroLang/NeuroLang.git"


Dependencies
------------

NeuroLang requires the following libraries (installed automatically):

* `NumPy <https://numpy.org>`_, `SciPy <https://scipy.org>`_ and
  `pandas <https://pandas.pydata.org>`_
* `nibabel <https://nipy.org/nibabel/>`_ and
  `nilearn <https://nilearn.github.io>`_ for neuroimaging data
* `PySDD <https://github.com/wannesm/PySDD>`_ for probabilistic inference
  through knowledge compilation
* `Lark <https://github.com/lark-parser/lark>`_ for the Datalog and SQUALL
  parsers
* `rdflib <https://rdflib.readthedocs.io>`_ for ontologies
* `siibra <https://siibra-python.readthedocs.io>`_ for atlases
* `scikit-learn <https://scikit-learn.org>`_,
  `matplotlib <https://matplotlib.org>`_ and
  `seaborn <https://seaborn.pydata.org>`_
* `PyYAML <https://pyyaml.org>`_ and ``typing_inspect``

Optional extras
---------------

.. list-table::
   :header-rows: 1

   * - Extra
     - Installs
     - Use it to
   * - ``ras-polars``
     - ``polars``, ``pyarrow``
     - use the Polars relational-algebra backend (see :doc:`performance`)
   * - ``dev``
     - ``pytest``, ``pytest-cov``, ``pytest-benchmark``, ``problog``
     - run the test suite
   * - ``doc``
     - ``sphinx`` and extensions
     - build this documentation

For instance, from a clone of the repository:

.. code-block:: bash

   uv pip install -e ".[ras-polars]"


Alternative backend
-------------------

By default NeuroLang stores relations in pandas data frames. One
alternative backend is available:

``polars``
   Backed by `Polars <https://pola.rs>`_ lazy frames. Install the
   ``ras-polars`` extra. It is usually faster on relations holding numbers
   and strings; see :doc:`performance` for details.

The former ``dask`` (dask-sql) backend has been removed; selecting it raises
a ``ValueError`` that lists the available backends.

To select a backend, either edit the ``[RAS] backend`` entry of
``neurolang/config/config.ini``, or call

.. code-block:: python

   from neurolang.config import config
   config.set_query_backend("polars")  # "pandas" or "polars"

**at the top of your script, before importing other NeuroLang modules.**
