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
   :file:`packages/nl_expr/nl_expr/config/config.ini`.

The settings are available at runtime through the ``config`` object::

    from neurolang.config import config
    config["RAS"]["backend"]          # 'pandas'
    config.get_chase_max_iterations() # 10000

The main sections are:

``[RAS]`` — relational algebra sets
    ``backend``
        Which implementation stores and manipulates relations: ``pandas``
        (default) or ``polars``.  The ``polars`` backend needs the
        optional dependencies ``pip install "neurolang[ras-polars]"``.
        Any other value, including the removed ``dask`` backend, raises a
        ``ValueError``.

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
        the query satisfies the conditions its rules need (a single type of
        quantifier, and unateness) and declines it otherwise, so that the more
        general, slower, knowledge-compilation solver is used.  Turning it
        off disables these soundness checks: the lifted rules are then
        applied to queries for which they are not proved, and can return
        wrong probabilities instead of falling back to weighted model
        counting.  For instance, the separator variable rule takes the
        existential projection whenever the query has an existential
        quantifier anywhere (``separator_variable_plan``), which is wrong
        for a universally quantified separator variable.  Only do so if you
        know your queries are safe for lifted inference.

``[DEFAULT]`` and ``[ONTOLOGIES]`` hold debugging, pretty-printing,
spatial-relation and ontology namespace options.


Switching the backend at runtime
--------------------------------

The backend can be changed from Python without editing ``config.ini``::

    from neurolang.config import config
    config.set_query_backend("polars")   # or "pandas"

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


.. _several-queries:

Several queries at once
-----------------------

The queries ``ans[name](...) :- ...`` of a program are solved together.
Solving them one after the other computes what they share again for each of
them. Solved together, each query is rewritten with magic sets for its own
bindings, as it would be alone, and the rewritten rules of all of them are
joined into one program, which is stratified and run once:

* a predicate that two queries use is computed once. Its rules are the same
  in both rewritings when the queries bind the same arguments, and are kept
  once; and a predicate that they use without binding the same arguments, or
  without binding any, has a rule set for each way of binding it;
* the *magic* predicate that restricts a predicate holds the bindings of all
  the queries that use it (``magic_path^bf_0(1)`` and ``magic_path^bf_0(3)``
  for ``path(1, y)`` and ``path(3, y)``), as the seed of the recursion for
  both: the transitive closure from 1 and from 3 is computed in one
  fixpoint, over the nodes either reaches. This is sound, as magic sets only
  restrict the facts a query needs, and each query reads a shared predicate
  only through its own bindings, so that a query has the answer it has
  alone; the tests check it, for aggregates over a predicate that another
  query binds, and for probabilistic queries;
* probabilistic queries of the program are stratified together: the
  deterministic stratum is solved once, and the probabilistic queries over
  it after it.

The rewritten program of all the queries is shown by ``--show-rewritten``
and ``--dry-run``, and the plans by ``--show-ra`` are those of each query.

On a random graph of 3 000 nodes and 9 000 edges, with queries ``ans[q_i](y,
w) :- near(i, y), reach(y, w)`` over a recursive ``reach``, the times of 1, 2,
4 and 8 queries are 0.7 s, 0.8 s, 2.5 s and 4.9 s one after the other, and
0.4 s, 0.6 s, 1.6 s and 2.7 s together (one run, with the pandas backend).

This is the *multi-query optimisation* of databases, which finds the common
subexpressions of a batch of queries and computes them once [Sellis1988]_,
[Roy2000]_, applied to Datalog, where the sharing is that of the predicates
(and of their bindings) that the queries have in common rather than that of
subexpressions of their plans. Magic sets [Balbin1991]_ restrict a query to
its bindings, as the demand of the query; the demand of several queries
over a predicate is the union of theirs. The cost-based choice among the
subexpressions to share [Roy2000]_, [KathuriaSudarshan]_ and the sharing of
queries that bind a predicate with different adornments, which
subsumptive tabling [Tekle2011]_ does by answering a call from the answers
of a more general one, are not done: queries that bind a predicate differently have each their own rewriting
of it.

.. [Sellis1988] T. K. Sellis, *Multiple-query optimization*, ACM
   Transactions on Database Systems 13(1), 1988.
.. [Roy2000] P. Roy, S. Seshadri, S. Sudarshan, S. Bhobe, *Efficient and
   extensible algorithms for multi query optimization*, SIGMOD 2000
   (`arXiv:cs/9910021 <https://arxiv.org/abs/cs/9910021>`_).
.. [KathuriaSudarshan] A. Kathuria, S. Sudarshan, *Efficient and provable
   multi-query optimization*
   (`arXiv:1512.02568 <https://arxiv.org/abs/1512.02568>`_).
.. [Balbin1991] I. Balbin, G. S. Port, K. Ramamohanarao, K. Meenakshi,
   *Efficient bottom-up computation of queries on stratified databases*,
   Journal of Logic Programming 11(3-4), 1991.
.. [Tekle2011] K. T. Tekle, Y. A. Liu, *More efficient Datalog queries:
   subsumptive tabling beats magic sets*, SIGMOD 2011
   (`doi:10.1145/1989323.1989393
   <https://dl.acm.org/doi/10.1145/1989323.1989393>`_).


.. _euclidean-join-settings:

Memory and time of Euclidean distance bounds
--------------------------------------------

A rule bounding the distance between two points, such as
``close(i, j, k, s) :- voxel(i, j, k), peak(a, b, c, s), (EUCLIDEAN(i, j, k,
a, b, c) < 5)``, is rewritten to join a relation of the pairs of points
within the distance, found once by
:func:`~neurolang.utils.euclidean_join.euclidean_join`. The pairs of a few
hundred thousand peaks and a brain mask are tens to hundreds of millions,
so the memory the join holds is the memory of a query. There are two ways
to find them:

* with integer coordinates, or floats on a grid (multiples of 1, 0.5, 0.25,
  0.1, 0.05 or 0.01, as millimetre coordinates are), a *stencil* looks up in
  a grid of the larger set the points at each integer offset within the
  radius of each point of the smaller set. A look-up costs a few
  nanoseconds and a pair no more memory than its two indices;
* with other coordinates, *k-d trees* are searched, in chunks of the smaller
  set. A pair they return costs about 200 ns and 100 bytes of temporaries.

The stencil is used whenever the larger set fills at least one cell in 40
of the grid of its coordinates, and up to a number of look-ups otherwise.
On 300 000 peaks and the 318 000 voxels of a 2 mm mask, a radius of 6 voxels
(120 million pairs) takes 18 s and 6.6 GB, a radius of 3 takes 1.6 s and
1.0 GB; about half the memory and time of the k-d trees. The result alone
(four 64-bit columns) is 3.7 GB at radius 6: that is the floor.

The thresholds, the sizes of the chunks and the other tunable values are
in ``neurolang/utils/euclidean_join.yaml``, with the meaning of each. The
file defines *profiles*, which change some of them:

``default``
    balances time and memory;

``memory``
    uses the stencil up to a billion look-ups (the default one up to 20
    million, or when the larger set is dense), counts the pairs before
    storing them so that they are held once, searches the k-d trees in
    chunks of 250 000 pairs and uses small blocks. It saves
    about 10 % of the memory of ``default`` with the stencil (6.0 GB instead
    of 6.6 GB at radius 6) and 30 % with the k-d trees (4.5 GB instead of
    6.5 GB, in the same time, for floats off a grid at radius 5); it is the profile of the interactive REPL, in a
    terminal and in a browser (``TuiSettings.join_profile``).

Choose a profile with the environment variable
``NEUROLANG_EUCLIDEAN_JOIN_PROFILE``, in Python with
:func:`neurolang.utils.euclidean_join_settings.use_profile` or, for a
block of code, :func:`~neurolang.utils.euclidean_join_settings.profile`,
and read another file of the same layout with
``NEUROLANG_EUCLIDEAN_JOIN_CONFIG``. The file is read with OmegaConf, as
the engines' configuration is, and an unknown setting or a value of the
wrong type is an error. An engine changes some settings with its
``euclidean_join`` key (see :doc:`engine_configuration`); they are applied
when the engine is built and, over the ``memory`` profile, in the
interactive REPL.

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
