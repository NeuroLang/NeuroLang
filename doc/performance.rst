.. _performance:

Performance and Configuration
=============================

This page describes NeuroLang's configuration file, how to choose the
relational algebra backend, and a few practical tips to keep queries fast.


The configuration file
----------------------

NeuroLang reads its settings from a ``config.ini`` file when
:mod:`neurolang.config` is first imported.  The first file found in the
following locations is used:

#. ``<sys.prefix>/config/config.ini`` — e.g. inside your virtual
   environment, which lets you keep per-environment settings;
#. the default file shipped with the package,
   :file:`neurolang/config/config.ini`.

The settings are available at runtime through the ``config`` object::

    from neurolang.config import config
    config["RAS"]["backend"]          # 'pandas'
    config.get_chase_max_iterations() # 10000

The main sections are:

``[RAS]`` — relational algebra sets
    ``backend``
        Which implementation stores and manipulates relations: ``pandas``
        (default), ``dask`` or ``polars``.  The ``polars`` backend needs the
        optional dependencies ``pip install "neurolang[ras-polars]"``; the
        ``dask`` backend needs the ``dask`` and ``dask-sql`` packages.
    ``synchronous``, ``tableIds``
        Only used by the ``dask`` backend: whether to use the synchronous
        dask scheduler, and whether tables get human-readable or UUID names.

``[CHASE]`` — the Datalog fixpoint (chase) engine
    ``max_iterations``
        Maximum number of chase iterations (default ``10000``).  When it is
        exceeded, NeuroLang raises an error instead of looping forever; this
        protects against cyclic or very deep recursion.  Raise it if a
        legitimate deeply recursive program hits the limit.
    ``sort_join_predicates``
        When ``True`` (default), the body atoms of each rule are joined
        smallest relation first, a simple join-ordering heuristic.  It is
        switched off automatically when selecting the ``polars`` backend,
        whose own query optimiser reorders joins and for which computing
        relation sizes would force evaluation of lazy frames.

``[PROBABILISTIC_SOLVER]``
    ``check_unate``
        When ``True`` (default), the lifted probabilistic solver verifies that
        the query satisfies the conditions it needs (a single type of
        quantifier, and unateness) and declines it otherwise, so that the more
        general, slower, knowledge-compilation solver is used.  Turning it
        off skips these checks; only do so if you know your queries are
        safe for lifted inference.

``[DEFAULT]`` and ``[ONTOLOGIES]`` hold debugging, pretty-printing,
spatial-relation and ontology namespace options.


Switching the backend at runtime
--------------------------------

The backend can be changed from Python without editing ``config.ini``::

    from neurolang.config import config
    config.set_query_backend("polars")   # or "pandas", "dask"

    from neurolang.frontend import NeurolangPDL
    nl = NeurolangPDL()

Several modules decide which relation classes to use when they are
imported; ``set_query_backend`` reloads them, but objects created before the
switch keep the previous backend.  Therefore **call it before importing the
rest of NeuroLang and before creating any engine**.  The setting is
process-wide: it applies to every engine in the Python process.  The SQUALL
directive ``#set_backend('polars').`` (see :doc:`tutorial_squall`) calls the
same function and has the same process-wide effect.

When does ``polars`` help?  It is most effective on large relations whose
columns hold numbers or strings, such as peak coordinates, study
identifiers, or TF-IDF tables, which it processes in native, multi-threaded
code.  Columns holding arbitrary Python objects (for instance brain regions
such as :class:`~neurolang.regions.ExplicitVBR`) are stored with polars'
``Object`` type, and operations on them fall back to slower Python code; for
programs dominated by such relations ``pandas`` is usually as fast or faster.


Ask only for what you need: ``query`` vs ``solve_all``
------------------------------------------------------

``nl.query(...)`` (and the ``ans(...)`` rule of a Datalog program, or the
``obtain`` clause in SQUALL) rewrites the program with the **magic sets**
transformation before solving it.  Only the rules and tuples relevant to the
query are computed, and constants in the query (for instance a region name)
are propagated into the rules so that intermediate relations are filtered
early.  ``nl.solve_all()``, on the other hand, computes every relation
defined by the program.  On large data sets, prefer a query; see the
magic-sets sections of :doc:`tutorial_datalog` and :doc:`tutorial_squall`
for an example.  The rewritten program can be inspected with
``neurolang-query --show-rewritten``.


Practical tips
--------------

* **Pass DataFrames to** ``add_tuple_set``.  A :class:`pandas.DataFrame` is
  stored almost as is, while a list of Python tuples has to be converted
  first.  For large relations, build a DataFrame (or a NumPy array) once and
  pass it directly.
* **Reuse engine instances.**  Creating an engine and loading data (atlases,
  NeuroSynth tables, ontologies) is often more expensive than running a
  query.  Load the data once and run several queries on the same
  ``NeurolangDL`` / ``NeurolangPDL`` instance; use ``with nl.scope as e:``
  for temporary rules so they do not accumulate between queries.
* **Keep probabilistic queries liftable.**  The probabilistic solver first
  tries exact lifted inference and only falls back to knowledge compilation
  when the query is not supported; the latter can be much slower.  Defining
  intermediate deterministic rules before the probabilistic step, as in the
  Bayes factor examples of the tutorials, helps.
* **Threshold early.**  Filtering large relations (e.g. TF-IDF values above a
  threshold) when loading them, or in the first rule that uses them, reduces
  the size of every subsequent join.
