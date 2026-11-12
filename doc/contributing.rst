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


Running the REPL in the Browser Locally
---------------------------------------

The `REPL in the browser <_static/repl/index.html>`_ runs NeuroLang on
Pyodide, from a static page.  To run that of your checkout on your machine:

.. code-block:: bash

   uv run --extra apps neurolang-repl-serve

The first run builds the page's bundle (NeuroLang's wheels and the data of
the ``neurosynth`` engine, a few minutes) in
:file:`~/.cache/neurolang/repl-site`; later runs serve it at once,
``--rebuild`` builds it again from the checkout, and ``--clear`` removes the
bundle built before and then builds it from nothing (the text embeddings
below are kept).  The page opens at
``http://localhost:8766/_static/repl/index.html``; the browser loads Pyodide
from its CDN.  ``--host 0.0.0.0 --server-name NAME`` serves the other
machines of the network, which reach it by NAME (see below),
``--port`` changes the port, ``--no-browser`` does not open the page, and
``--help`` lists the options.  The command is that of the workspace's root
project (:file:`tools/serve_repl`).

CI builds the REPL in the browser for ``master`` only, deployed with the
documentation: its bundle takes an hour on a runner's CPU when the
embeddings of the studies are not kept.  A pull request's CI uploads its
documentation, the artifact ``repl-pr-NUMBER`` (kept 30 days), without a
REPL that can start; to try the REPL of a branch, build it locally with
``neurolang-repl-serve``, as above.

The same page can run NeuroLang in the Python of this machine rather than
in the browser::

   uv run --extra apps --extra tui --extra llm neurolang-repl-serve --local --data-dir neurolang_data

builds no bundle: the page's worker (``local_worker.js``) sends what it is
asked to the server, which builds the engine from ``--data-dir`` (by default
``neurolang_data``, as ``neurolang-query``), with all of its analyses
(``lift``, ``ratios``...) unless ``--light``, and runs the page's REPL
(``neurolang_apps.utils.tui.local_backend``).  Queries take the time and the
memory they take in ``neurolang-query``; *Stop* interrupts the query on the
server, and the AI assistant is the one configured on this machine, unless a
key is given in the page.  The server answers this machine only, unless
``--host``: whoever it answers runs queries, and saves files, with its
Python.

Without a password (below), the server answers only the requests sent to
``localhost``, ``127.0.0.1``, ``::1``, the ``--host`` it is given (unless
``0.0.0.0`` or ``::``), and the names of ``--server-name NAME`` (repeat it
for several): a page of another site whose name it made lead to your
machine (DNS rebinding) would otherwise run queries in its Python.  The
other machines of the network need ``--server-name`` with the name, or the
address, by which they reach it, e.g. ``--host 0.0.0.0 --server-name
repl.lab.org``; the server's answer to another name says so.  With a
password, any name is answered.  With a password or without, a POST is
answered only if it is the page's: in JSON (``Content-Type:
application/json``, which a form of another site cannot send), and from
this site if the browser gives its ``Origin``; otherwise any site you visit
could run queries here.  A connection that says nothing for a minute (its
TLS handshake, its request) is closed, and the body of a request to
``/api/`` is a JSON object of 1 MB at most.

To serve it over HTTPS, give a certificate and its private key, PEM files
(the key may be in the certificate's file)::

   uv run --extra apps --extra tui --extra llm neurolang-repl-serve --local --host 0.0.0.0 --server-name repl.lab.org --cert cert.pem --key key.pem

A key with a passphrase is opened with the first line of
``--key-password-file FILE`` (readable by you only; the line as it is, but
for its line ending), else the passphrase is
asked in the terminal; a service needs the file.  If they cannot serve, the
server says why: the certificate's file has no key, the key is not that of
the certificate, or the passphrase is not that one.  A PEM file is text,
its first line ``-----BEGIN CERTIFICATE-----`` or ``-----BEGIN PRIVATE
KEY-----`` (``ENCRYPTED PRIVATE KEY``: with a passphrase); ``openssl x509
-in cert.pem -noout -subject`` and ``openssl pkey -in key.pem -noout`` read
them, and ``openssl x509 -inform DER -in cert.der -out cert.pem`` converts a
binary (DER) one.

To have it ask a password, put one on the first line of a file (readable by
you only: ``chmod 600``; the line as it is, spaces included, but for its
line ending) and give ``--password-file FILE``: the browser asks
for it (any user name will do) before the site's first file, and the
server answers no request without it.  Over the network it needs
``--cert``, as the password goes with each request.  With ``--local``,
``--env-file ENV`` gives those with the password the key of the AI
assistant: ENV's ``NAME=VALUE`` lines are put in the server's environment,
where the provider of :file:`~/.config/neurolang/llm.yaml` reads its key (the
variable its ``api_key_env`` names, e.g. ``ALBERT_API_KEY=...``).  The key
stays on the server; a key given in the page replaces it.  With a password,
the visitors share one session of the REPL, its rules and a key one of them
gives, and a restart restarts it for all.  The site's main page, ``/``, leads
to the REPL.

For several users, each with sessions of their own, give a users file
instead of a password, ``--users users.yaml``, readable by you only::

   ann:
     password: scrypt$16384$8$1$...     # neurolang-repl-serve --hash-password
     env:                               # optional: the key of her assistant
       ALBERT_API_KEY: ...
   bob:
     password: scrypt$16384$8$1$...     # no env: the --env-file key, or his own

The browser asks a name and its password.  With ``--local``, each tab of a
user has a session of its own, in a process of its own: a builder process
builds each engine once and forks a session's process from it, so that the
sessions share the memory of the engine's data until they write it, and
each has its own rules, AI assistant and queries, which run at the same
time; a stop or a restart is that of the tab's session only.  A user's
``env`` is put in the environment of their sessions only: the key of their
assistant, over that of ``--env-file``, and a key their page gives over
both.  At most ``--max-sessions`` (8) run at once, and a session unused for
``--idle-minutes`` (120) ends (its page says to restart it).  The users can
read what the server's account can, and their sessions share the data
directory, where the files they save by a relative name may collide; give
it to people you trust.  The feedback server, where the sessions' rated
answers may go, takes their writes at once (SQLite in WAL mode).

To keep it serving, as a service of your user started now and at each login,
add ``--daemon install`` to the options it serves with; ``--daemon
reinstall`` replaces it with the new options (once their bundle is built
and the service's Python can serve, so that a failure leaves the service
as it was), and ``--daemon remove``, with no other option, stops and
removes it; if systemd or launchd fails to start it, it is removed and the
error told::

   uv run --extra apps --extra tui --extra llm neurolang-repl-serve --daemon install --local --host 0.0.0.0 --cert cert.pem --key key.pem --users users.yaml --env-file llm.env

On Linux it is a unit of systemd's user manager,
:file:`~/.config/systemd/user/neurolang-repl.service` (``journalctl --user
-u neurolang-repl`` shows its log; ``loginctl enable-linger`` keeps it
running after you log out); on macOS a launchd agent,
:file:`~/Library/LaunchAgents/org.neurolang.repl.plist`, logging to
:file:`~/Library/Logs/neurolang-repl.log`.  ``install`` builds the bundle
first, where you see it; the service runs the Python of the environment
that installs it (with ``uv run``, the checkout's :file:`.venv/bin/python`;
``--python PATH`` chooses another), with that environment's ``PATH`` and the
paths made absolute, but not the other variables of your shell (API keys).  It uses that environment as it is: a
later ``uv run`` or ``uv sync`` without ``--extra apps`` (and the other
extras it was installed with) removes their packages from it, and the
service then fails to start until they are back.  A port below
1024 needs a service of the system, which this does not install.

``query_similarity`` needs the text embeddings in the bundle: the SPECTER2
query model and the embeddings of the studies.  They are kept in
:file:`~/.cache/neurolang/repl-embeddings` (``--embeddings``) and made
there, the first time, by a build with torch: ``uv run --extra apps --extra
embeddings neurolang-repl-serve --rebuild``, which downloads SPECTER2 and the
abstracts of the studies from PubMed and embeds them (minutes on a GPU,
longer on a CPU).  Without them the page works, but ``query_similarity``
says that it was built without them.


Submitting a Pull Request
--------------------------

1. Fork the repository on GitHub.
2. Create a feature branch: ``git checkout -b my-feature``
3. Make your changes.
4. Ensure all tests pass: ``uv run pytest packages/``
5. Add a changelog entry under *Unreleased* in ``doc/changelog.rst``.
6. Update docstrings for any changed public API.
7. Push your branch and open a pull request against ``master``.

CI runs on every pull request, also one whose base is the branch of another
pull request (a stacked pull request): the tests are those of the packages
it changes with respect to its base.  Its documentation is the artifact
``repl-pr-NUMBER``; the REPL in the browser of a branch is built locally
(see `Running the REPL in the Browser Locally`_).

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
   Python init.  See :ref:`concepts` for details.  Example queries
   (``examples``, ``assistant_examples``) and the query assistant's
   ``assistant_guidance`` go in ``engines/my_dataset/examples.yaml`` and
   ``engines/my_dataset/guidance.yaml``, merged into the entry (see
   :ref:`neurolang-engine-registry`).

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
