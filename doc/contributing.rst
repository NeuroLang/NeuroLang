.. _contributing:

Contributing to NeuroLang
=========================

Thank you for considering a contribution! This page explains how to set up
a development environment and submit changes.


Prerequisites
-------------

* **Python ≥ 3.8**
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
   uv pip install -e ".[dev,doc]"

This installs NeuroLang in *editable* mode together with all development
(``pytest``, ``pytest-cov``) and documentation (Sphinx) dependencies.


Running the Tests
-----------------

.. code-block:: bash

   pytest neurolang/

To run with coverage:

.. code-block:: bash

   pytest --cov=neurolang neurolang/

The test suite should pass fully before you submit a pull request.


Building the Documentation
--------------------------

.. code-block:: bash

   make -C doc html

This will build the HTML docs into ``doc/_build/html/``. Open
``doc/_build/html/index.html`` in your browser to preview.

.. note::

   Building the full example gallery requires all neuroimaging packages
   (nibabel, nilearn, neurosynth) and can take several minutes. To skip
   the gallery during a quick doc build, set:

   .. code-block:: bash

      SPHINXOPTS="-D sphinx_gallery_conf.run_stale_examples=False" make -C doc html


Submitting a Pull Request
--------------------------

1. Fork the repository on GitHub.
2. Create a feature branch: ``git checkout -b my-feature``
3. Make your changes.
4. Ensure all tests pass: ``pytest neurolang/``
5. Add a changelog entry under *Unreleased* in ``doc/changelog.rst``.
6. Update docstrings for any changed public API.
7. Push your branch and open a pull request against ``master``.

**PR checklist:**

* [ ] Tests pass (``pytest neurolang/``)
* [ ] New code has docstrings (NumPy format)
* [ ] Changelog entry added to ``doc/changelog.rst``
* [ ] No new Sphinx build warnings (``make -C doc html``)


Code Style
----------

NeuroLang follows `PEP 8 <https://peps.python.org/pep-0008/>`_ with
NumPy-style docstrings. Run ``flake8`` before committing:

.. code-block:: bash

   pip install flake8
   flake8 neurolang/


Adding a New Dataset Engine
----------------------------

The ``neurolang-query`` CLI uses a declarative engine registry to manage
dataset backends.  To add a new engine:

1. Write a YAML entry in ``neurolang/utils/engines/engines.yaml``:

   .. code-block:: yaml

      my_dataset:
        description: "Short description of the dataset"
        requires_mni_mask: false
        python_init: "neurolang.utils.engines.my_dataset.init"
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
    ``neurolang/utils/engines/my_dataset/init.py`` that exports:

   .. code-block:: python

      from pathlib import Path
      import nibabel as nib
      from neurolang.frontend import NeurolangPDL

      def init_engine(
          nl: NeurolangPDL,
          mask: nib.Nifti1Image | None,
          data_dir: Path
      ) -> None:
          \"\"\"Register symbols and load data into *nl*.\"\"\"
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

3. Add unit tests in ``neurolang/utils/tests/test_cli.py``, specifically
   in the ``TestEngineRegistry`` class to verify that your engine is
   discoverable and its config loads correctly.

4. Run ``pytest neurolang/utils/tests/test_cli.py`` to confirm everything
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
