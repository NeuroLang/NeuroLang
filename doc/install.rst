Installing NeuroLang
====================

NeuroLang requires **Python ≥ 3.12**. The recommended way to install it is
with `uv <https://github.com/astral-sh/uv>`_ — a fast Python package manager —
or with standard ``pip``.

NeuroLang is four packages, each built on the ones before it:

``nl-expr``
   the expression engine: typed expressions, pattern-matching walkers and
   first-order logic;
``nl-core``
   relational algebra, Datalog and probabilistic Datalog;
``neurolang``
   the Datalog and SQUALL languages, the frontends
   (:class:`~neurolang.NeurolangDL`, :class:`~neurolang.NeurolangPDL`) and
   brain regions;
``neurolang-apps``
   the ``neurolang-query`` command line, its interactive mode, the query
   assistant and the engines of predefined data.

``neurolang[apps]`` installs them all; ``neurolang`` alone installs the
library without the applications.

.. note::

   ``pip install neurolang`` installs the latest release published on PyPI
   (``0.1a21``, October 2026), from before the split: one package, with the
   applications. The commands below install the current ``master`` branch
   from GitHub instead, with the changes listed as *Unreleased* in the
   :doc:`changelog`.

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

         uv pip install "neurolang[apps] @ git+https://github.com/NeuroLang/NeuroLang.git#subdirectory=packages/neurolang"

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

         uv pip install "neurolang[apps] @ git+https://github.com/NeuroLang/NeuroLang.git#subdirectory=packages/neurolang"

      **3. Verify**

      .. code-block:: bash

         python -c "import neurolang; print('NeuroLang installed OK')"

   .. tab-item:: Linux

      **1. Install uv**

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

      **2. Install NeuroLang**

      .. code-block:: bash

         uv pip install "neurolang[apps] @ git+https://github.com/NeuroLang/NeuroLang.git#subdirectory=packages/neurolang"

      **3. Verify**

      .. code-block:: bash

         python -c "import neurolang; print('NeuroLang installed OK')"

   .. tab-item:: From source

      Clone the repository and install its four packages in editable mode,
      in the environment ``.venv`` of the clone, with all extras and the
      development and documentation tools:

      .. code-block:: bash

         git clone https://github.com/NeuroLang/NeuroLang.git
         cd NeuroLang
         uv sync --all-packages --all-extras --no-extra embeddings

      To update your local copy:

      .. code-block:: bash

         git pull
         uv sync --all-packages --all-extras --no-extra embeddings

      Run Python in that environment with ``uv run`` or by activating it
      (``source .venv/bin/activate``).

      Verify:

      .. code-block:: bash

         python -c "import neurolang; print('NeuroLang installed OK')"


Using pip without uv
---------------------

If you prefer plain pip, which does not know that the packages come from
the same repository, name the four of them:

.. code-block:: bash

   pip install \
     "nl-expr @ git+https://github.com/NeuroLang/NeuroLang.git#subdirectory=packages/nl_expr" \
     "nl-core @ git+https://github.com/NeuroLang/NeuroLang.git#subdirectory=packages/nl_core" \
     "neurolang @ git+https://github.com/NeuroLang/NeuroLang.git#subdirectory=packages/neurolang" \
     "neurolang-apps @ git+https://github.com/NeuroLang/NeuroLang.git#subdirectory=packages/neurolang_apps"


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
* ``typing_inspect``
* for the applications, `scikit-learn <https://scikit-learn.org>`_,
  `matplotlib <https://matplotlib.org>`_ and `PyYAML <https://pyyaml.org>`_

Optional extras
---------------

.. list-table::
   :header-rows: 1

   * - Extra
     - Installs
     - Use it to
   * - ``apps``
     - ``neurolang-apps``
     - use the ``neurolang-query`` command line and the engines
   * - ``tui``
     - ``prompt-toolkit``, ``rich``
     - use the interactive mode of ``neurolang-query``
   * - ``ras-polars``
     - ``polars``, ``pyarrow``
     - use the Polars relational-algebra backend (see :doc:`performance`)
   * - ``dev``
     - ``pytest``, ``problog``
     - run the test suite

For instance, from a clone of the repository:

.. code-block:: bash

   uv sync --all-packages --extra ras-polars


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
``packages/nl_expr/nl_expr/config/config.ini``, or call

.. code-block:: python

   from neurolang.config import config
   config.set_query_backend("polars")  # "pandas" or "polars"

**at the top of your script, before importing other NeuroLang modules.**
