.. _tutorial-python:

NeuroLang in Python
===================

The rest of this documentation types queries into the REPL (:doc:`repl`).
The same engine is a Python object, for scripts and notebooks, for data
that is not in an engine, and for functions of your own. This page shows how,
in this order: an engine of the registry, Datalog text, your own relations
and probabilistic choices, the syntax embedded in Python, and what the REPL's
dot-commands do in Python.

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
with probabilities of their own, the first column holding the probability;
``load_ontology`` reads an OWL or RDF ontology into relations, as the
``ontologies`` of an engine (:doc:`tutorial_cognitive_atlas`).


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
