.. _contributing:

Contributing to NeuroLang
=========================

Thank you for considering a contribution! This page explains how to set up
a development environment and submit changes.


Prerequisites
-------------

* **Python ≥ 3.12**
* **git**
* **uv** (fast Python package manager) — install with:

  .. code-block:: bash

     curl -LsSf https://astral.sh/uv/install.sh | sh

  Or on Windows (PowerShell):

  .. code-block:: powershell

     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"


Development Install
-------------------

.. code-block:: bash

   git clone https://github.com/NeuroLang/NeuroLang.git
   cd NeuroLang
   uv sync --all-packages --all-extras --no-extra embeddings

This installs NeuroLang's four packages in *editable* mode, in the
environment ``.venv`` of the clone, together with all development
(``pytest``, ``pytest-cov``) and documentation (Sphinx) dependencies. To
install them in another environment, ``uv pip install -e ".[apps,dev,doc]"``
does the same from the root of the clone, whose ``pyproject.toml`` is a
project, never published, depending on the four.


The Packages
------------

The repository is a `uv workspace
<https://docs.astral.sh/uv/concepts/projects/workspaces/>`_ of four
packages, each of which imports only the ones before it:

=============================  ==============================================
``packages/nl_expr``           the expression engine: expressions, walkers,
                               types, logic, configuration
``packages/nl_core``           relational algebra and its sets, Datalog,
                               probabilistic Datalog
``packages/neurolang``         the languages and frontends, brain regions
``packages/neurolang_apps``    the command line, its interactive mode, the
                               query assistant, the engines
=============================  ==============================================

``.importlinter`` states these layers, which CI checks with
``uv run lint-imports``. A module put in the wrong package, or importing
one above it, fails that check. The modules keep their former names under
``neurolang``: ``neurolang.datalog`` is ``nl_core.datalog``, and
``neurolang.utils.cli`` is ``neurolang_apps.utils.cli``. Code inside the
packages imports the new names.

The tests of a package are in it, beside the modules they test, and use
only that package and the ones before it. CI runs the tests of the
packages a pull request changes and of those built on them.


Running the Tests
-----------------

.. code-block:: bash

   uv run pytest packages/

or those of one package, e.g. ``uv run pytest packages/nl_core``. To run
with coverage:

.. code-block:: bash

   uv run pytest --cov=nl_expr --cov=nl_core --cov=neurolang \
       --cov=neurolang_apps packages/

The test suite should pass fully before you submit a pull request.


Building the Documentation
--------------------------

.. code-block:: bash

   make -C doc html

This will build the HTML docs into ``doc/_build/html/``. Open
``doc/_build/html/index.html`` in your browser to preview.

.. note::

   The documentation build executes only the gallery examples whose file
   name matches ``plot_load_destrieux`` (``filename_pattern`` in
   :file:`doc/conf.py`); the other examples are rendered without being run.
   Running them requires the neuroimaging packages (nibabel, nilearn) and
   downloads data.  To skip running the gallery entirely during a quick doc
   build, disable ``plot_gallery``:

   .. code-block:: bash

      make -C doc html SPHINXOPTS="-D plot_gallery=0"


Submitting a Pull Request
--------------------------

1. Fork the repository on GitHub.
2. Create a feature branch: ``git checkout -b my-feature``
3. Make your changes.
4. Ensure all tests pass: ``uv run pytest packages/``
5. Add a changelog entry under *Unreleased* in ``doc/changelog.rst``.
6. Update docstrings for any changed public API.
7. Push your branch and open a pull request against ``master``.

**PR checklist:**

* [ ] Tests pass (``uv run pytest packages/``)
* [ ] New code has docstrings (NumPy format)
* [ ] Changelog entry added to ``doc/changelog.rst``
* [ ] No new Sphinx build warnings (``make -C doc html``)


Code Style
----------

NeuroLang follows `PEP 8 <https://peps.python.org/pep-0008/>`_ with
NumPy-style docstrings. Run ``flake8`` before committing:

.. code-block:: bash

   uvx flake8 packages/


Adding a New Dataset Engine
----------------------------

The ``neurolang-query`` CLI uses a declarative engine registry to manage
dataset backends.  To add a new engine:

1. Write a YAML entry in ``packages/neurolang_apps/neurolang_apps/utils/engines/engines.yaml``:

   .. code-block:: yaml

      my_dataset:
        description: "Short description of the dataset"
        requires_mni_mask: false
        python_init: "neurolang_apps.utils.engines.my_dataset.init"
        datalog_init: |
          derived_pred(X) :- base_pred(X)
        relations:
          base_pred: "my_dataset/base_data.csv"
        predicates:
          my_predicate:
            arity: 2
            columns: [name, value]
            description: "What this predicate represents"

   The optional fields ``datalog_init`` (inline Datalog rules) and
   ``relations`` (CSV/TSV files loaded as predicates) run after the
   Python init.  See :ref:`concepts` for details.

2. If your engine needs custom Python logic (data downloads, non-trivial
   processing), create the init module at
   ``packages/neurolang_apps/neurolang_apps/utils/engines/my_dataset/init.py`` that defines:

   .. code-block:: python

      from pathlib import Path
      import nibabel as nib
      from neurolang.frontend import NeurolangPDL

      def init_engine(
          nl: NeurolangPDL,
          mask: nib.Nifti1Image | None,
          data_dir: Path
      ) -> None:
          """Register symbols and load data into *nl*."""
          # Register predicates with nl.add_tuple_set, etc.

   If your engine needs the common neuroimaging symbols (``agg_count``,
   ``agg_create_region``, ``agg_create_region_overlay``,
   ``principal_direction``, ``region_union``), set
   ``use_base_symbols: true`` in the YAML entry instead of calling
   ``init_base_engine()`` from Python::

       my_dataset:
         description: "..."
         requires_mni_mask: true
         use_base_symbols: true
         builtins: [exp, log, startswith]
         ...

   This replaces the need for a Python init script when your engine
   only needs the standard neuroimaging symbols plus declarative YAML
   fields (``atlases``, ``relations``, ``datalog_init``, etc.).  See the
   ``destrieux`` engine entry for a complete example of a YAML-only engine.

   For engines that require custom data processing (like NeuroSynth's
   peak coordinate conversion), you still write a Python init module.
   In that case, import and call ``init_base_engine`` if you need the
   base symbols::

       from neurolang.utils.engines.base import init_base_engine

       def init_engine(nl, mask, data_dir):
           init_base_engine(nl, mask)
           # ... custom logic ...

   If using ``use_base_symbols: true`` in the YAML, do **not** also call
   ``init_base_engine`` from Python — the registry handles it for you.

3. Add unit tests in ``packages/neurolang_apps/neurolang_apps/utils/tests/test_cli.py``, specifically
   in the ``TestEngineRegistry`` class to verify that your engine is
   discoverable and its config loads correctly.

4. Run ``uv run pytest packages/neurolang_apps/neurolang_apps/utils/tests/test_cli.py`` to confirm everything
   passes.


Reporting Bugs
--------------

Open an issue on the
`GitHub issue tracker <https://github.com/NeuroLang/NeuroLang/issues>`_.
Include:

* NeuroLang version (``python -c "import neurolang; print(neurolang.__version__)"`` if available)
* Python version
* Operating system
* Minimal reproducible example
