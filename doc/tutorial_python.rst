.. _tutorial-python:

NeuroLang in Python
===================

The rest of this documentation types queries into the REPL (:doc:`repl`).
The same engine is a Python object, for scripts and notebooks, for data
that is not in an engine, and for functions of your own. This page shows how,
in this order: an engine of the registry, Datalog text, your own relations
and probabilistic choices, SQUALL text, the syntax embedded in Python, and
what the REPL's dot-commands do in Python.

The examples with a ``>>>`` prompt run on small relations they build, and
are tested; they import:

    >>> from neurolang.frontend import NeurolangDL, NeurolangPDL

.. contents:: On this page
   :local:
   :depth: 1


An engine of the registry
-------------------------

``neurolang-query --engine neurosynth`` builds its engine with
:func:`~neurolang_apps.utils.engine_registry.build_engine`, which a script
calls too (it needs the ``apps`` extra, and downloads the data the first
time):

.. code-block:: python

   from pathlib import Path
   from neurolang_apps.utils.engine_registry import build_engine

   nl = build_engine("neurosynth", Path("neurolang_data"), resolution=2)
   result = nl.execute_datalog_program("ans(s) :- mentions(s, 'pain')")
   result.as_pandas_dataframe()          # the 516 studies that mention pain

``nl`` is a :class:`~neurolang.frontend.NeurolangPDL`: every query of the
tutorials runs on it as a program text, as below.


Datalog text
------------

``execute_datalog_program`` runs a program, the statements a REPL session
would take. A program with a query, an ``ans`` rule, returns its answer;
without one, the rules are added to the engine and ``solve_all()`` computes
every relation:

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [("alice",), ("bob",), ("carol",)], name="person"
    ... )
    >>> _ = nl.add_tuple_set([("alice",), ("carol",)], name="plays")
    >>> nl.execute_datalog_program("active(x) :- person(x), plays(x).")
    >>> solution = nl.solve_all()
    >>> sorted(solution["active"].as_pandas_dataframe().iloc[:, 0])
    ['alice', 'carol']

    >>> result = nl.execute_datalog_program(
    ...     """
    ...     player(x) :- person(x), plays(x).
    ...     non_player(x) :- person(x), ~plays(x).
    ...     ans(x) :- non_player(x).
    ...     """
    ... )
    >>> result.as_pandas_dataframe().iloc[:, 0].tolist()
    ['bob']

An answer is a set of tuples; ``.as_pandas_dataframe()`` turns it into a
data frame. A program with several named queries, ``ans[name](...)``,
returns a dictionary from their names to their answers:

    >>> results = nl.execute_datalog_program(
    ...     """
    ...     ans[players](x) :- plays(x).
    ...     ans[others](x) :- person(x), ~plays(x).
    ...     """
    ... )
    >>> sorted(results)
    ['others', 'players']

Comparisons, aggregations and every construct of :doc:`tutorial_datalog`
are written as in the REPL:

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [("a", 0), ("a", 1), ("b", 2), ("c", 3)], name="item_count"
    ... )
    >>> result = nl.execute_datalog_program(
    ...     """
    ...     max_count(x, m) :- AGGREGATE[x](item_count(x, c) @ max(c)) = m.
    ...     ans(x, m) :- max_count(x, m), (m >= 1).
    ...     """
    ... )
    >>> sorted(result.as_pandas_dataframe().itertuples(index=False, name=None))
    [('a', 1), ('b', 2), ('c', 3)]

The rules of a program stay in the engine, as in a REPL session; within
``with nl.scope:`` they are forgotten at the end of the block. ``show_ra=True``
prints the relational algebra plans of a program's queries instead of running
them (see :ref:`appendix-b-datalog`).


Your own relations and probabilistic choices
--------------------------------------------

``add_tuple_set`` adds a relation from an iterable of tuples, a NumPy array
or a pandas data frame. ``add_uniform_probabilistic_choice_over_set`` adds a
probabilistic choice where each tuple is the chosen one with the same
probability, as the engine's ``selected_study``, over which ``PROB`` computes
probabilities:

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [("s1", "A"), ("s2", "A"), ("s3", "B")], name="activates"
    ... )
    >>> _ = nl.add_uniform_probabilistic_choice_over_set(
    ...     [("s1",), ("s2",), ("s3",)], name="selected_study"
    ... )
    >>> result = nl.execute_datalog_program(
    ...     "ans(r, p) :- PROB[selected_study(s), activates(s, r)] = p."
    ... )
    >>> sorted(result.as_pandas_dataframe().itertuples(index=False, name=None))  # doctest: +ELLIPSIS
    [('A', 0.666...), ('B', 0.333...)]

``add_resampling`` declares replicates of such a choice, for the confidence
intervals of :doc:`tutorial_datalog` (10.7), as an engine's configuration
does:

    >>> _ = nl.add_resampling(
    ...     "boot_studies", choice="selected_study", n=200, scheme="bayesian"
    ... )
    >>> result = nl.execute_datalog_program(
    ...     """
    ...     p_a(p) :- PROB[selected_study(s), activates(s, 'A')] = p.
    ...     rep(b, p) :- RESAMPLE[boot_studies](p_a(p)) = b.
    ...     ans(percentile(p, 2.5), percentile(p, 97.5)) :- rep(b, p).
    ...     """
    ... )
    >>> low, high = next(result.as_pandas_dataframe().itertuples(index=False))
    >>> 0 < low < 2 / 3 < high < 1
    True

``add_probabilistic_facts_from_tuples`` and
``add_probabilistic_choice_from_tuples`` add probabilistic facts and choices
with probabilities of their own, the first column holding the probability.


.. _python-ontology:

An ontology
...........

``load_ontology`` reads an OWL or RDF ontology into relations, as the
``ontologies`` of an engine (:doc:`tutorial_cognitive_atlas`); the queries of
that page then run on any frontend:

.. code-block:: python

    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.load_ontology(
        "cogat.v2.owl", individuals=True, names="prefixed"
    )
    nl.execute_datalog_program("""
        cogat_subclass_of(x, y) :- neurolang:subClassOf(x, y)
        cogat_subclass_of(x, z) :-
            cogat_subclass_of(x, y), neurolang:subClassOf(y, z)
    """)
    below = nl.execute_datalog_program(
        "ans(l) :- rdfs:label(c, 'Working memory'),"
        " cogat_subclass_of(x, c), rdfs:label(x, l)"
    )


.. _part-9-appendix-a:

SQUALL from Python
------------------

All of the SQUALL syntax of :doc:`tutorial_squall` can be executed from
Python, as the text of ``execute_squall_program``.  The general workflow is:

1. Create a ``NeurolangPDL`` engine.
2. Register EDB facts with ``add_tuple_set``.
3. Execute a SQUALL program string with ``execute_squall_program``.
4. Inspect results via the direct return from ``obtain`` queries or with
   ``solve_all()``.

Registering facts and running a simple rule
...........................................

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [("alice",), ("bob",), ("carol",)], name="person"
    ... )
    >>> _ = nl.add_tuple_set(
    ...     [("alice",), ("carol",)], name="plays"
    ... )
    >>> nl.execute_squall_program(
    ...     "define as Active every person that plays."
    ... )
    >>> sorted(
    ...     nl.solve_all()["active"].as_pandas_dataframe().iloc[:, 0].tolist()
    ... )
    ['alice', 'carol']


Querying with ``obtain`` (direct return)
........................................

    >>> result = nl.execute_squall_program(
    ...     "obtain every person that plays."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice', 'carol']


Multiple rules and an obtain clause in one program
..................................................

    >>> result = nl.execute_squall_program(
    ...     "define as Player every person that plays. "
    ...     "define as Runner every person that runs. "
    ...     "obtain every Player."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice', 'carol']


Transitive verbs and binary predicates
......................................

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> _ = nl.add_tuple_set([("jazz",)], name="genre")
    >>> _ = nl.add_tuple_set([("alice", "jazz")], name="sings")
    >>> nl.execute_squall_program(
    ...     "define as Performer every person that sings a Genre."
    ... )
    >>> sorted(nl.solve_all()["performer"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']


Quantifiers: every, a, no
.........................

**Every**

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",)], name="plays")
    >>> nl.execute_squall_program("define as Active every person that plays.")
    >>> sorted(nl.solve_all()["active"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']

**A / an / some**

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("a",), ("b",), ("c",)], name="item")
    >>> _ = nl.add_tuple_set([("a", 1), ("b", 2)], name="item_count")
    >>> result = nl.execute_squall_program(
    ...     "obtain every item ?i that has an item_count ?c."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['a', 'b']

**No**

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("a",), ("b",), ("c",)], name="item")
    >>> _ = nl.add_tuple_set([("a", 1), ("b", 2)], name="item_count")
    >>> result = nl.execute_squall_program(
    ...     "obtain every item ?i that has no item_count ?c."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['c']


Relative clauses and negation
.............................

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",)], name="plays")
    >>> nl.execute_squall_program(
    ...     "define as NotPlaying every person that does not plays."
    ... )
    >>> sorted(nl.solve_all()["notplaying"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['bob']


Tuple (multi-dimensional) subjects
..................................

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [("v1", 0, 0, 1), ("v2", 1, 2, 3)], name="voxel"
    ... )
    >>> nl.execute_squall_program(
    ...     "define as active every voxel (?v; ?x; ?y; ?z)."
    ... )
    >>> solution = nl.solve_all()
    >>> sorted(solution["active"].as_pandas_dataframe().apply(tuple, axis=1).tolist())
    [('v1', 0, 0, 1), ('v2', 1, 2, 3)]

Anonymous wildcard ``_``::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [(10, 20, 30, "s1"), (11, 21, 31, "s2")], name="peak_reported"
    ... )
    >>> nl.execute_squall_program(
    ...     "define as Activation every Peak_reported (?i; ?j; ?k; _)."
    ... )
    >>> solution = nl.solve_all()
    >>> sorted(solution["activation"].as_pandas_dataframe().apply(tuple, axis=1).tolist())
    [(10, 20, 30), (11, 21, 31)]


Compound quantifiers and anaphora
.................................

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("A",), ("B",)], name="region")
    >>> _ = nl.add_tuple_set([("x",), ("y",)], name="term")
    >>> _ = nl.add_tuple_set([("s1",), ("s2",), ("s3",)], name="selected_study")
    >>> _ = nl.add_tuple_set([("s1", "A"), ("s2", "A"), ("s3", "B")], name="activates")
    >>> _ = nl.add_tuple_set([("s1", "x"), ("s2", "y"), ("s3", "x")], name="mentions")
    >>> nl.execute_squall_program(
    ...     "define as Cooccurrence "
    ...     "for every Region and for every Term "
    ...     "where a Selected_study activates the Region and mentions the Term."
    ... )
    >>> sorted(
    ...     nl.solve_all()["cooccurrence"]
    ...     .as_pandas_dataframe().apply(tuple, axis=1).tolist()
    ... )
    [('A', 'x'), ('A', 'y'), ('B', 'x')]


Probabilistic n-ary rules
.........................

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("A",), ("B",)], name="region")
    >>> _ = nl.add_tuple_set([("x",), ("y",)], name="term")
    >>> _ = nl.add_tuple_set([("s1",), ("s2",), ("s3",), ("s4",)], name="study")
    >>> _ = nl.add_tuple_set(
    ...     [("s1", "A"), ("s2", "A"), ("s3", "B")], name="activates"
    ... )
    >>> _ = nl.add_tuple_set(
    ...     [("s1", "x"), ("s2", "y"), ("s3", "x")], name="mentions"
    ... )
    >>> result = nl.execute_squall_program(
    ...     "define as Selected_study as an equiprobable choice over every Study. "
    ...     "define as Cooccurrence "
    ...     "for every Region and for every Term "
    ...     "where a Selected_study activates the Region and mentions the Term. "
    ...     "define as Joint_prob with inferred probability "
    ...     "every Cooccurrence (?r; ?t). "
    ...     "obtain every Joint_prob (?r; ?t; ?p) as P."
    ... )
    >>> df = result.as_pandas_dataframe()
    >>> df.columns = ["r", "t", "p"]
    >>> sorted(df.itertuples(index=False, name=None))
    [('A', 'x', 0.25), ('A', 'y', 0.25), ('B', 'x', 0.25)]

Each region–term pair co-occurs in exactly one of the four equiprobable
studies, hence the probability 1/4.


Probabilistic choice (SQUALL)
.............................

An equiprobable choice can be defined directly in SQUALL, avoiding the
need for ``add_uniform_probabilistic_choice_over_set`` in Python:

    >>> from neurolang.expressions import Symbol
    >>>
    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("s1",), ("s2",), ("s3",)], name="study")
    >>> nl.execute_squall_program(
    ...     "define as Selected_study "
    ...     "as an equiprobable choice over every Study."
    ... )
    >>> sym = Symbol("selected_study")
    >>> sym in nl.program_ir.pchoice_pred_symbs
    True


Filtering with comparisons
..........................

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [("a",), ("b",), ("c",), ("d",)], name="item"
    ... )
    >>> _ = nl.add_tuple_set(
    ...     [("a", 0), ("a", 1), ("b", 2), ("c", 3)], name="item_count"
    ... )
    >>> nl.execute_squall_program(
    ...     "define as Large every Item "
    ...     "that has an item_count greater equal than 2."
    ... )
    >>> sorted(nl.solve_all()["large"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['b', 'c']


Aggregations
............

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [("a",), ("b",), ("c",), ("d",)], name="item"
    ... )
    >>> _ = nl.add_tuple_set(
    ...     [("a", 0), ("a", 1), ("b", 2), ("c", 3)], name="item_count"
    ... )
    >>> _ = nl.add_tuple_set([(i,) for i in range(5)], name="quantity")
    >>> nl.execute_squall_program(
    ...     "define as max_items for every Item ?i ;"
    ...     " where every Max of the Quantity where ?i item_count per ?i."
    ... )
    >>> solution = nl.solve_all()
    >>> sorted(
    ...     solution["max_items"].as_pandas_dataframe()
    ...     .apply(tuple, axis=1).tolist()
    ... )
    [('a', 1), ('b', 2), ('c', 3)]


Boolean connectives
...................

**Conjunction (and)**

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",), ("carol",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",), ("carol",)], name="plays")
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="runs")
    >>> nl.execute_squall_program(
    ...     "define as Player every person that plays. "
    ...     "define as PlayAndRun every Player that runs."
    ... )
    >>> sorted(nl.solve_all()["playandrun"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']

**Disjunction (or)**

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",), ("carol",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",)], name="plays")
    >>> _ = nl.add_tuple_set([("bob",)], name="runs")
    >>> nl.execute_squall_program(
    ...     "define as PlayOrRun every person that plays or runs."
    ... )
    >>> sorted(nl.solve_all()["playorrun"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice', 'bob']


The syntax embedded in Python
-----------------------------

Rules can also be written as Python expressions, without a program text.
``nl.environment`` creates symbols as they are named: ``e.x`` is a variable,
``e.connected[...] = ...`` a fact or a rule:

    >>> nl = NeurolangDL()
    >>> with nl.environment as e:
    ...     e.connected[0, 1] = True
    ...     e.connected[1, 2] = True
    ...     e.connected[2, 3] = True
    ...     e.reachable[e.x, e.y] = e.connected[e.x, e.y]
    ...     e.reachable[e.x, e.y] = e.reachable[e.x, e.z] & e.connected[e.z, e.y]

The program is in memory, to be explored:

    >>> print(nl.symbols['connected'])
    connected: typing.AbstractSet[typing.Tuple[int, int]] = (0, 1) ...
    >>> for rule in nl.current_program:
    ...     print(rule)
    reachable(x, y) ← connected(x, y)
    reachable(x, y) ← ( reachable(x, z) ) ∧ ( connected(z, y) )

and queried:

    >>> with nl.environment as e:
    ...     res = nl.query((e.x, e.y), e.reachable(e.x, e.y))
    >>> print(res)
       x  y
    0  0  1
    1  0  2
    2  0  3
    3  1  2
    4  1  3
    5  2  3

``&`` is a conjunction, ``|`` a disjunction and ``~`` a negation, as in the
text syntax; ``with nl.scope as e:`` writes rules that are forgotten at the
end of the block.

Functions of your own
.....................

A Python function added with ``add_symbol`` is a function of the language,
or an aggregation when it takes an iterable:

    >>> from typing import Iterable
    >>> nl = NeurolangDL()
    >>> nl.add_tuple_set([(0, 1), (1, 2), (2, 3)], name='connected')
    connected: typing.AbstractSet[typing.Tuple[int, int]] = (0, 1) ...
    >>> @nl.add_symbol
    ... def agg_count(x: Iterable) -> int:
    ...     return len(set(x))
    >>> with nl.environment as e:
    ...     e.reachable[e.x, e.y] = e.connected[e.x, e.y]
    ...     e.reachable[e.x, e.y] = e.reachable[e.x, e.z] & e.connected[e.z, e.y]
    ...     e.count_arrivals[e.x, agg_count(e.y)] = e.reachable(e.x, e.y)
    ...     counts = nl.query((e.x, e.c), e.count_arrivals(e.x, e.c) & (e.x < 2))
    >>> print(counts)
       x  c
    0  0  3
    1  1  2

Constraints and open knowledge
..............................

``add_constraint(antecedent, consequent)`` adds a tuple-generating
dependency: the variables of the consequent only are existentially
quantified. Every person has a parent, even if the data does not name it,
:math:`\forall x\, \mathrm{person}(x) \rightarrow \exists y\,
\mathrm{parent}(y, x)`:

    >>> nl = NeurolangPDL()
    >>> with nl.environment as e:
    ...     e.person['Pat'] = True
    ...     e.person['Chris'] = True
    ...     _ = nl.add_constraint(e.person[e.x], e.parent[e.y, e.x])
    ...     e.has_parent[e.x] = e.person[e.x] & e.parent[e.y, e.x]
    ...     res = nl.query((e.x,), e.has_parent(e.x))
    >>> print(res)
           x
    0    Pat
    1  Chris


The REPL's commands in Python
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - In the REPL
     - In Python
   * - ``.load PATH NAME``
     - ``nl.add_tuple_set(pandas.read_csv(PATH), name=NAME)``
   * - ``%%`` documentation, ``.predicates``
     - ``nl.document_predicate(name, description, arguments)``,
       ``nl.predicate_documentation()``
   * - ``.import("m.dl")``
     - ``nl.import_datalog_module("m.dl")``; ``with
       nl.program_ir.importing_from("main.dl"):`` runs a program read from
       ``main.dl`` with its imports relative to the file
   * - ``.clear``
     - ``with nl.scope:`` around the rules to forget
   * - ``.ra on``
     - ``nl.execute_datalog_program(program, show_ra=True)``
   * - the relational algebra backend
     - ``neurolang.config.config.set_query_backend("polars")``
       (:doc:`performance`)

The gallery's examples (:doc:`auto_examples/index`) are complete Python
scripts: atlases, Neurosynth meta-analyses, SQUALL programs, text
embeddings.
