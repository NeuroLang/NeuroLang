SQUALL: Controlled English for NeuroLang
========================================

SQUALL (*Semantically controlled Query-Answerable Logical Language*) lets you
write NeuroLang queries and rules in plain English sentences instead of
symbolic Datalog notation.  Under the hood, each sentence is translated to a
NeuroLang logical expression using Montague semantics in
Continuation-Passing Style.

This tutorial walks through every supported construct with runnable examples.
All code blocks below can be executed with ``pytest --doctest-glob=doc/tutorial_squall.rst``.

.. contents:: Contents
   :local:
   :depth: 2


Setup
-----

Every example uses the high-level ``NeurolangPDL`` frontend::

    >>> from neurolang.frontend import NeurolangPDL


1. Basic Sentences
------------------

The simplest SQUALL sentence consists of a **subject** (a variable or
literal) and a **verb** (a unary predicate).

Variables are written with a ``?`` prefix.  String literals are enclosed in
single quotes.  The following example checks that a sentence about a named
entity fires as expected::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",)], name="plays")
    >>> result = nl.execute_squall_program("obtain every plays.")
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']

A **transitive** verb takes an object.  Transitive verbs used as binary
predicates are prefixed with ``~`` to indicate that argument order is
inverted (so ``x ~sings y`` maps to ``sings(y, x)``)::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> _ = nl.add_tuple_set([("jazz",)], name="genre")
    >>> _ = nl.add_tuple_set([("alice", "jazz")], name="sings")
    >>> nl.execute_squall_program(
    ...     "define as Performer every person ?x that a Genre ?y ~sings."
    ... )
    >>> sorted(nl.solve_all()["performer"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']


2. Quantifiers
--------------

SQUALL supports four determiners: ``every``, ``a``/``an``/``some``, ``no``,
and ``the``.

**Universal — every**

``every person plays`` means *for all x: if person(x) then plays(x)*::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",)], name="plays")
    >>> nl.execute_squall_program("define as Active every person that plays.")
    >>> sorted(nl.solve_all()["active"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']

**Existential — a / an / some**

``a person plays`` asserts existence.  In a query, only items with an
associated count are returned::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("a",), ("b",), ("c",)], name="item")
    >>> _ = nl.add_tuple_set([("a", 1), ("b", 2)], name="item_count")
    >>> result = nl.execute_squall_program(
    ...     "obtain every item ?i that has an item_count ?c."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['a', 'b']

**Negative — no**

``no`` inside a relative clause expresses negation-as-failure.  Items that
have *no* associated count are returned::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("a",), ("b",), ("c",)], name="item")
    >>> _ = nl.add_tuple_set([("a", 1), ("b", 2)], name="item_count")
    >>> result = nl.execute_squall_program(
    ...     "obtain every item ?i that has no item_count ?c."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['c']

**Named variables with quantifiers**

Variables can be named explicitly using ``?name`` labels directly after the
noun.  The label binds the variable so it can be reused elsewhere in the
sentence::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",)], name="plays")
    >>> nl.execute_squall_program(
    ...     "define as Active every person ?p that plays."
    ... )
    >>> sorted(nl.solve_all()["active"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']


3. Relative Clauses
-------------------

Relative clauses restrict the noun they modify.  They are introduced by
``that``, ``which``, ``who``, or ``where``.

**Intransitive VP relative clause**

``every person that plays`` — for every x: person(x) and plays(x)::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",)], name="plays")
    >>> nl.execute_squall_program(
    ...     "define as PlayerPerson every person that plays."
    ... )
    >>> sorted(nl.solve_all()["playerperson"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']

**Transitive VP relative clause (passive-like)**

``~verb`` signals a transitive (binary) predicate used in *passive* position:
``a study ~reports voxel`` reads ``reports(study, voxel)`` with argument order
reversed.  The rule below collects (study, voxel) pairs via an explicit
multi-variable head::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("v1",), ("v2",), ("v3",)], name="voxel")
    >>> _ = nl.add_tuple_set([("s1",), ("s2",)], name="study")
    >>> _ = nl.add_tuple_set([("s1", "v1"), ("s2", "v2")], name="reports")
    >>> nl.execute_squall_program(
    ...     "define as reported for every Study ?s ; with every Voxel ?v that ?s reports."
    ... )
    >>> sorted(
    ...     nl.solve_all()["reported"]
    ...     .as_pandas_dataframe().apply(tuple, axis=1).tolist()
    ... )
    [('s1', 'v1'), ('s2', 'v2')]

**Nested relative clauses**

Relative clauses can be nested by using an intermediate IDB predicate as the
noun.  The example below defines ``selected_player`` from the intersection of
two independent predicates::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",), ("carol",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",), ("carol",)], name="plays")
    >>> _ = nl.add_tuple_set([("alice",)], name="selected")
    >>> nl.execute_squall_program(
    ...     "define as PlayingSelected every selected that plays."
    ... )
    >>> sorted(nl.solve_all()["playingselected"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']

**Negative relative clause**

``does not VP`` expresses negation-as-failure on a unary predicate::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",)], name="plays")
    >>> nl.execute_squall_program(
    ...     "define as NotPlaying every person that does not plays."
    ... )
    >>> sorted(nl.solve_all()["notplaying"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['bob']

**Possessive relative clause — whose**

``whose NG2 VP`` expresses a possessive relationship via a binary noun.
``define as published every person whose writer plays.`` means: for every
person x, there exists a y such that writer(x, y) and plays(y) — and that
person is ``published``::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> _ = nl.add_tuple_set([("alice", "carol"), ("bob", "dave")], name="writer")
    >>> _ = nl.add_tuple_set([("carol",)], name="plays")
    >>> nl.execute_squall_program(
    ...     "define as published every person whose writer plays."
    ... )
    >>> sorted(nl.solve_all()["published"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']


4. Tuple (Multi-dimensional) Subjects
--------------------------------------

When a noun denotes a multi-dimensional entity (e.g. a voxel with x, y, z
coordinates), a parenthesised tuple of labels can follow the noun.  The
variables bind to the respective columns of the relation::

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

The compiler generates one binding per coordinate variable and produces a
single conjunction for the body.


5. Defining Rules with ``define as``
-------------------------------------

The ``define as`` prefix turns a sentence into a Datalog **rule definition**.

**Simple unary rule**::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",), ("carol",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",), ("carol",)], name="plays")
    >>> nl.execute_squall_program("define as Active every person that plays.")
    >>> sorted(nl.solve_all()["active"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice', 'carol']

**End-to-end execution**

The rule fires for every person that satisfies ``plays``::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",), ("carol",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",), ("carol",)], name="plays")
    >>> nl.execute_squall_program("define as Active every person that plays.")
    >>> solution = nl.solve_all()
    >>> sorted(solution["active"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice', 'carol']


6. Multi-Variable Rules and Joins
----------------------------------

N-ary rules use ``for every NOUN ; with every NOUN`` (or other prepositions)
to bind multiple variables into the head::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("a",), ("b",), ("c",)], name="item")
    >>> _ = nl.add_tuple_set(
    ...     [("a", 0), ("a", 1), ("b", 2), ("c", 3)], name="item_count"
    ... )
    >>> _ = nl.add_tuple_set([(i,) for i in range(5)], name="quantity")
    >>> nl.execute_squall_program(
    ...     "define as merge for every Item ?i ;"
    ...     " with every Quantity that ?i item_count"
    ... )
    >>> solution = nl.solve_all()
    >>> sorted(solution["merge"].as_pandas_dataframe().apply(tuple, axis=1).tolist())
    [('a', 0), ('a', 1), ('b', 2), ('c', 3)]


7. Filtering with Comparisons
------------------------------

Relative clauses can include comparison predicates using the keywords
``greater``, ``lower``, ``equal``, combined with optional ``equal`` and
``not``, followed by ``than`` or ``to`` and an operand.

Comparison keywords:

* ``greater than``
* ``greater equal than``
* ``lower than``
* ``lower equal than``
* ``equal to``
* ``not equal to``

The following rule selects items whose ``item_count`` is at least 2::

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

Item ``"d"`` is absent because it has no ``item_count`` entry.


8. Querying with ``obtain``
----------------------------

The ``obtain`` keyword introduces a **query** rather than a rule.
``execute_squall_program`` returns a ``NamedRelationalAlgebraFrozenSet``
directly::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [("a",), ("b",), ("c",), ("d",)], name="item"
    ... )
    >>> _ = nl.add_tuple_set(
    ...     [("a", 0), ("a", 1), ("b", 2), ("c", 3)], name="item_count"
    ... )
    >>> result = nl.execute_squall_program(
    ...     "obtain every Item that has an item_count."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['a', 'b', 'c']

Item ``"d"`` is absent because it has no ``item_count`` entry.

**Mixing rules and queries**

A single program can contain both ``define as`` rules and an ``obtain``
clause::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",)], name="plays")
    >>> result = nl.execute_squall_program(
    ...     "define as Active every person that plays. "
    ...     "obtain every Active."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']

**MARG queries — conditional probability**

The ``with probability … conditioned to …`` form defines a *marginal* (MARG)
conditional probability relation.  The engine rewrites it into numerator,
denominator and final ratio rules automatically, adding a probability column as
the last argument.

.. code-block:: python

   nl = NeurolangPDL()
   nl.add_tuple_set([("s1",), ("s2",)], name="study")
   nl.add_tuple_set([("s1",)], name="selected_study")
   nl.add_tuple_set([("s2",)], name="open_world_studies")
   nl.add_tuple_set([("s1",), ("s2",)], name="reported")
   nl.execute_squall_program(
       "define as Prob_report with probability every Reported ?s "
       "conditioned to every Selected_study ?s that Open_world_studies."
   )

The relation ``prob_report`` will contain one column per variable plus a
final probability column computed as
``P(reported | selected_study & open_world_studies)``.


9. Aggregations
----------------

Aggregations summarise a set of values into a single result per group.
The syntax follows the pattern::

    define as RESULT for every SUBJECT ;
        where every AGG_FUNC of the MEASURE where CONDITION per SUBJECT.

Supported aggregation functions: ``count``, ``sum``, ``max``, ``min``,
``average``.

The following rule computes the maximum ``item_count`` value per item::

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

Item ``"d"`` is absent because it has no ``item_count``.

**Global aggregation (no groupby)**

When no ``per`` clause is given, the aggregation function receives *all free
variables* of the source relation.  Any callable registered in the engine's
symbol table can be used as the aggregation functor:

.. code-block:: python

   from neurolang.expressions import Symbol, Constant

   nl = NeurolangPDL()
   nl.add_tuple_set([("a",), ("b",), ("c",)], name="item")
   nl.symbol_table[Symbol("collect_all")] = Constant(lambda vals: sorted(vals))
   nl.execute_squall_program(
       "define as Result every Collect_all of the Item."
   )

The relation ``result`` will contain the output of ``collect_all`` applied over
all tuples from ``item``.


10. Multiple Rules in One Program
-----------------------------------

Separate rules with a full stop.  The parser processes each rule and walks
them all into the engine::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",)], name="plays")
    >>> _ = nl.add_tuple_set([("bob",)], name="runs")
    >>> nl.execute_squall_program(
    ...     "define as Active every person that plays. "
    ...     "define as Fast every person that runs."
    ... )
    >>> solution = nl.solve_all()
    >>> sorted(solution["active"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']
    >>> sorted(solution["fast"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['bob']


11. Boolean Connectives in Relative Clauses
--------------------------------------------

Relative clauses support ``and`` (conjunction) and ``or`` (disjunction).
With conjunction, two rules can be combined step by step — define an
intermediate predicate and then constrain further::

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

With ``or``, the individual must satisfy at least one of the conditions::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",), ("carol",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",)], name="plays")
    >>> _ = nl.add_tuple_set([("bob",)], name="runs")
    >>> nl.execute_squall_program(
    ...     "define as PlayOrRun every person that plays or runs."
    ... )
    >>> sorted(nl.solve_all()["playorrun"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice', 'bob']


12. ``for … , …`` Quantification
----------------------------------

A sentence can be prefixed with ``for NOUN_PHRASE ,`` to bind the outer
variable first.  In ``define as`` rule definitions, the equivalent is to name
the variable explicitly after the noun using the ``?var`` label.  The example
below demonstrates named variable binding, which is the standard way to refer
to a variable in the rule body::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",)], name="plays")
    >>> nl.execute_squall_program(
    ...     "define as Active every person ?p that plays."
    ... )
    >>> sorted(nl.solve_all()["active"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']


13. Reserved Words and Quoting
--------------------------------

SQUALL reserves many common English words as keywords (``every``, ``a``,
``the``, ``that``, ``is``, ``has``, ``not``, ``and``, ``or``, ``where``,
``who``, ``which``, etc.).  If a predicate or entity name coincides with a
reserved word, wrap it in backticks::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",)], name="from")
    >>> result = nl.execute_squall_program(
    ...     "obtain every `from`."
    ... )  # doctest: +ELLIPSIS
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']

Variable names use the ``?`` prefix and may contain letters, digits, and
underscores::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("s001",), ("s002",)], name="study")
    >>> result = nl.execute_squall_program(
    ...     "obtain every study ?study_id."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['s001', 's002']

String literals use single quotes and may contain spaces::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("neuro study",), ("other",)], name="study")
    >>> result = nl.execute_squall_program(
    ...     "obtain every study that is 'neuro study'."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['neuro study']


15. Neuroimaging Domain Examples
----------------------------------

This section mirrors the patterns used in the actual NeuroLang examples
(``examples/squall_examples.py``, ``examples/plot_neurosynth_implementation.py``)
and shows how the same queries are expressed in SQUALL.

**Finding activated voxels reported by studies**

Each study in NeuroSynth reports ``(study, voxel)`` pairs.  We want to
collect every voxel that at least one study has reported as activated.
The predicate ``?s reports ?v`` maps to ``reports(s, v)``::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [("s1",), ("s2",), ("s3",)], name="study"
    ... )
    >>> _ = nl.add_tuple_set(
    ...     [("v1",), ("v2",), ("v3",)], name="voxel"
    ... )
    >>> _ = nl.add_tuple_set(
    ...     [("s1", "v1"), ("s2", "v1"), ("s2", "v2")], name="reports"
    ... )
    >>> nl.execute_squall_program(
    ...     "define as Activated every Voxel ?v that a Study ?s reports."
    ... )
    >>> sorted(nl.solve_all()["activated"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['v1', 'v2']

.. note::

   The tilde (``~``) *reverses* argument order.  Use it when the EDB stores
   ``(voxel, study)`` so that ``?v ~reports ?s`` maps to ``reports(v, s)``
   which the engine sees as ``reports(voxel, study)``.

**Filtering by study category (two-rule chain)**

Select a subset of studies by a category predicate, then collect the
voxels those studies report::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [("s1",), ("s2",)], name="auditory_study"
    ... )
    >>> _ = nl.add_tuple_set(
    ...     [("v1",), ("v2",), ("v3",)], name="voxel"
    ... )
    >>> _ = nl.add_tuple_set(
    ...     [("s1", "v1"), ("s2", "v2"), ("s2", "v3")], name="reports"
    ... )
    >>> nl.execute_squall_program(
    ...     "define as Auditory_voxel every Voxel ?v "
    ...     "that an Auditory_study ?s reports."
    ... )
    >>> sorted(nl.solve_all()["auditory_voxel"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['v1', 'v2', 'v3']

The two-rule chain pattern is the SQUALL equivalent of:

.. code-block:: text

   auditory_voxel(v) :- voxel(v), auditory_study(s), reports(s, v).

**Atlas region filtering — registering a custom predicate**

Custom Python functions registered in ``symbol_table`` become body
predicates.  The example below uses the built-in ``startswith`` to
filter atlas region names:

.. code-block:: python

   from neurolang.expressions import Symbol, Constant

   nl = NeurolangPDL()
   nl.symbol_table[Symbol("startswith")] = Constant(str.startswith)
   nl.add_tuple_set(
       [("L S_temporal_sup",), ("R S_temporal_sup",), ("L G_frontal_sup",)],
       name="atlas_label"
   )
   # SQUALL (intransitive predicate acting on the registered function):
   # define as Left_label every Atlas_label ?label that startswith 'L '.
   #
   # NOTE: string literals in body predicates currently require a
   # comparison rel_comp form; the above is illustrative — see the
   # Gap Report (Section 16) for the current limitation.

.. note::

   String-literal arguments to arbitrary body predicates are not yet
   supported.  The workaround is to pre-filter the EDB in Python before
   calling ``execute_squall_program``.

**Multi-variable brain activation rule (tuple subject)**

When voxels are stored as ``(x, y, z)`` coordinate triples, the tuple
label syntax binds all three columns at once::

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [("s1",), ("s2",)], name="study"
    ... )
    >>> _ = nl.add_tuple_set(
    ...     [(0, 1, 2), (3, 4, 5), (6, 7, 8)], name="voxel"
    ... )
    >>> _ = nl.add_tuple_set(
    ...     [("s1", 0, 1, 2), ("s2", 3, 4, 5)], name="focus_reported"
    ... )
    >>> nl.execute_squall_program(
    ...     "define as Activation every Voxel (?x; ?y; ?z) "
    ...     "that a Study ?s focus_reported."
    ... )
    >>> sorted(
    ...     nl.solve_all()["activation"]
    ...     .as_pandas_dataframe().apply(tuple, axis=1).tolist()
    ... )
    [(0, 1, 2), (3, 4, 5)]

The tuple subject ``(?x; ?y; ?z)`` binds the three coordinate columns of
``voxel`` and re-uses those variables in the ``focus_reported`` body
(column order: study, x, y, z).

**Conditional probability (MARG) — activation probability**

The MARG form computes the conditional probability that a study-voxel
pair is activated given a conditioning predicate.  The full NeuroSynth
equivalent (see ``examples/plot_neurosynth_implementation.py``) is:

.. code-block:: python

   nl = NeurolangPDL()
   nl.add_uniform_probabilistic_choice_over_set(
       [("s1",), ("s2",), ("s3",)], name="selected_study"
   )
   nl.add_tuple_set([("s1", 0, 1, 2), ("s2", 3, 4, 5)],
                    name="focus_reported")
   nl.add_tuple_set([("s1",), ("s2",)], name="term_assoc")
   nl.execute_squall_program(
       "define as Activation every Focus_reported (?s; ?x; ?y; ?z) "
       "that Selected_study. "
       "define as Prob_map with probability every Activation (?s; ?x; ?y; ?z) "
       "conditioned to every Term_assoc ?s."
   )

The relation ``prob_map`` will contain ``(s, x, y, z, probability)``
triples where the last column is
``P(focus_reported(s,x,y,z) | term_assoc(s))``.

.. note::

   A full probabilistic solve requires ``NeurolangPDL.solve_all()``
   after ``execute_squall_program`` and a CPLogic-compatible EDB loaded
   via ``add_uniform_probabilistic_choice_over_set``.


16. Missing SQUALL Syntax — Gap Report
-----------------------------------------

The following Datalog / IR patterns appear in codebase examples with their
current status as of 2026-04-17:

.. list-table:: SQUALL gap report (updated 2026-04-17)
   :header-rows: 1
   :widths: 40 15 45

   * - Feature
     - Status
     - Notes
   * - ``rule_body2_cond`` two-sided conditioned NP
     - ✅ Fixed
     - ``define as X with probability every A conditioned to every B`` now routes to ``rule_op_marg``
   * - Function calls in rule body (e.g. ``euclidean(?x,?y)``)
     - ✅ Fixed
     - Use ``rel_fun_call``: ``every A that euclidean(?x,?y) holds``
   * - Comparison against computed variable (``?w greater than ?threshold``)
     - ✅ Confirmed working
     - Use ``rel_comp`` with a label in the RHS ``op`` position
   * - Anonymous wildcard ``_`` in n-ary predicates
     - ✅ Confirmed working
     - Each ``_`` creates a distinct fresh symbol; ``every Study that _ activates`` works
   * - Variable/expression as explicit probability (``with probability ?p``)
     - ✅ Fixed
     - ``vpdo_explicit_prob_v1/vn`` now accept any NP including labels
   * - Skolem-like functional terms in rule head
     - ❌ Not supported
     - Requires IR changes beyond transformer scope

.. rubric:: Examples — previously missing, now working

**Symmetric conditioned probability:**

.. code-block:: text

    define as spread with probability every virus conditioned to every study.

**Function call in body:**

.. code-block:: text

    define as Close every Pair that euclidean(?x, ?y) holds.

**Variable probability:**

.. code-block:: text

    define as Probable every Study that activates with probability ?p.

**Anonymous wildcard:**

.. code-block:: text

    define as HasActivation every Study that _ activates.


17. Using the IR Builder (``with nl.environment as e:``)
=========================================================

Every SQUALL sentence is compiled into NeuroLang's intermediate representation
(IR).  You can also write IR directly using the **environment context manager**.
This is useful when:

- a pattern has no SQUALL syntax yet;
- you need to mix Python logic with declarative rules;
- you want to inspect or reuse the IR objects from a rule.

**Scope vs Environment**

``nl.scope`` — symbols are popped from the symbol table when the ``with``
block exits (clean, no side effects).

``nl.environment`` — symbols persist in the symbol table after exit
(use when rules must be visible to later ``solve_all()`` calls).

Both use the same ``e.<Name>`` attribute syntax.

**Rule equivalence cheat-sheet**

**Simple unary rule**

SQUALL:

.. code-block:: text

    define as Active every person that plays.

IR builder:

.. code-block:: python

    with nl.environment as e:
        e.active[e.x] = e.person(e.x) & e.plays(e.x)
    sol = nl.solve_all()

**Binary / n-ary rule**

SQUALL:

.. code-block:: text

    define as author_of for every Paper ?p ; where every Author ?a ; where ?a wrote ?p.

IR builder:

.. code-block:: python

    with nl.environment as e:
        e.author_of[e.p, e.a] = e.wrote(e.a, e.p)

**Probabilistic fact**

SQUALL:

.. code-block:: text

    define as probably activates every study.

IR builder:

.. code-block:: python

    from neurolang.probabilistic.expressions import ProbabilisticFact
    from neurolang.expressions import Symbol

    with nl.environment as e:
        p = Symbol.fresh()
        e.activates[e.s] = ProbabilisticFact(p, e.study(e.s))

**Marginalisation (MARG) query**

SQUALL:

.. code-block:: text

    define as prob_map with probability every focus_reported (?x; ?y; ?z; ?s)
        conditioned to every selected_study ?s that open_world_studies.

IR builder:

.. code-block:: python

    from neurolang.probabilistic.expressions import ProbabilisticQuery, Condition, PROB

    with nl.environment as e:
        x, y, z, s = e.x, e.y, e.z, e.s
        e.prob_map[x, y, z, s, ProbabilisticQuery(PROB, (x, y, z, s))] = Condition(
            e.focus_reported(x, y, z, s),
            e.selected_study(s) & e.open_world_studies(s)
        )

**Aggregation**

SQUALL:

.. code-block:: text

    define as max_items for every Item ?i ;
        where every Max of the Quantity where ?i item_count per ?i.

IR builder:

.. code-block:: python

    from neurolang.datalog.aggregation import AggregationApplication
    from neurolang.expressions import Constant

    with nl.environment as e:
        q = e.q
        e.max_items[e.i, AggregationApplication(Constant(max), (q,))] = (
            e.item(e.i) & e.item_count(e.i, q)
        )
