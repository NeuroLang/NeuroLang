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


Reporting Bugs
--------------

Open an issue on the
`GitHub issue tracker <https://github.com/NeuroLang/NeuroLang/issues>`_.
Include:

* NeuroLang version (``python -c "import neurolang; print(neurolang.__version__)"`` if available)
* Python version
* Operating system
* Minimal reproducible example
