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

Every example imports the ``parser`` function and, for end-to-end execution
examples, the ``Chase`` solver and a ``RegionFrontendCPLogicSolver`` engine::

    >>> from neurolang.frontend.datalog.squall_syntax_lark import parser, SquallProgram
    >>> from neurolang.frontend.probabilistic_frontend import RegionFrontendCPLogicSolver
    >>> from neurolang.datalog.chase import Chase
    >>> from neurolang.expressions import Symbol, Constant
    >>> from neurolang.datalog import Implication


1. Basic Sentences
------------------

The simplest SQUALL sentence consists of a **subject** (a variable or
literal) and a **verb** (a unary predicate).

Variables are written with a ``?`` prefix::

    >>> result = parser("squall ?s reports")
    >>> from neurolang.expressions import FunctionApplication
    >>> assert isinstance(result, FunctionApplication)
    >>> assert "reports" in repr(result)
    >>> assert "s" in repr(result)

String literals are enclosed in single quotes::

    >>> result = parser("squall 'alice' plays")
    >>> assert "alice" in repr(result)
    >>> assert "plays" in repr(result)

A **transitive** verb takes an object.  Transitive verbs that are used as
binary predicates are prefixed with ``~`` to disambiguate them from
intransitive verbs::

    >>> result = parser("squall ?x ~sings ?y")
    >>> assert "sings" in repr(result)
    >>> assert "x" in repr(result)
    >>> assert "y" in repr(result)


2. Quantifiers
--------------

SQUALL supports four determiners: ``every``, ``a``/``an``/``some``, ``no``,
and ``the``.

**Universal — every**

``every person plays`` means *for all x: if person(x) then plays(x)*.
Parsed as ``UniversalPredicate(x, Implication(plays(x), person(x)))``.

::

    >>> from neurolang.logic import UniversalPredicate
    >>> result = parser("squall every person plays")
    >>> assert isinstance(result, UniversalPredicate)
    >>> assert "person" in repr(result)
    >>> assert "plays" in repr(result)

**Existential — a / an / some**

``a person plays`` means *there exists x such that person(x) and plays(x)*.

::

    >>> from neurolang.logic import ExistentialPredicate
    >>> result = parser("squall a person plays")
    >>> assert isinstance(result, ExistentialPredicate)
    >>> assert "person" in repr(result)
    >>> assert "plays" in repr(result)

**Negative — no**

``no person plays`` means *there is no x such that person(x) and plays(x)*.

::

    >>> from neurolang.datalog import Negation
    >>> result = parser("squall no person plays")
    >>> assert isinstance(result, Negation)
    >>> assert "person" in repr(result)

**Named variables with quantifiers**

Variables can be named explicitly using ``?name`` labels directly after the
noun.  The label binds the variable so it can be reused elsewhere in the
sentence::

    >>> result = parser("squall every person ?p plays")
    >>> assert isinstance(result, UniversalPredicate)
    >>> assert "p" in repr(result)


3. Relative Clauses
-------------------

Relative clauses restrict the noun they modify.  They are introduced by
``that``, ``which``, ``who``, or ``where``.

**Intransitive VP relative clause**

``every person that plays runs`` — for every x: person(x) and plays(x) →
runs(x)::

    >>> result = parser("squall every person that plays runs")
    >>> assert isinstance(result, UniversalPredicate)
    >>> assert "person" in repr(result)
    >>> assert "plays" in repr(result)
    >>> assert "runs" in repr(result)

**Transitive VP relative clause (passive-like)**

``every voxel that a study ~reports activates`` — for every voxel x, if
there is a study s that reports x, then x activates::

    >>> result = parser("squall every voxel that a study ~reports activates")
    >>> assert "voxel" in repr(result)
    >>> assert "reports" in repr(result)
    >>> assert "activates" in repr(result)

**Nested relative clauses**

Relative clauses can be nested arbitrarily::

    >>> result = parser(
    ...     "squall every voxel that a study that ~mentions a word "
    ...     "that is 'auditory' ~reports activates"
    ... )
    >>> assert "voxel" in repr(result)
    >>> assert "mentions" in repr(result)
    >>> assert "auditory" in repr(result)

**Negative relative clause**

``no`` inside a relative clause expresses negation-as-failure::

    >>> result = parser(
    ...     "squall every voxel that a study that ~reports no region "
    ...     "~reports activates"
    ... )
    >>> assert "voxel" in repr(result)
    >>> assert "activates" in repr(result)

**Possessive relative clause — whose**

``whose NG2 VP`` expresses a possessive relationship via a binary noun.
``define as published every person whose writer plays.`` means: for every
person x, there exists a y such that writer(x, y) and plays(y) — and that
person is ``published``::

    >>> result = parser("define as published every person whose writer plays.")
    >>> assert "person" in repr(result)
    >>> assert "writer" in repr(result)
    >>> assert "plays" in repr(result)


4. Tuple (Multi-dimensional) Subjects
--------------------------------------

When a noun denotes a multi-dimensional entity (e.g. a voxel with x, y, z
coordinates), a parenthesised tuple of labels can follow the noun::

    >>> from neurolang.logic import UniversalPredicate
    >>> result = parser(
    ...     "squall every voxel (?x; ?y; ?z) that a study ?s ~reports activates"
    ... )
    >>> assert isinstance(result, UniversalPredicate)
    >>> assert "voxel" in repr(result)
    >>> assert "reports" in repr(result)
    >>> assert "activates" in repr(result)

The parser nests three ``UniversalPredicate`` wrappers — one per coordinate
variable — and generates a single conjunction for the body.


5. Defining Rules with ``define as``
-------------------------------------

The ``define as`` prefix turns a sentence into a Datalog **rule definition**.
The result is an ``Implication`` expression that can be walked into an engine.

**Simple unary rule**::

    >>> rule = parser("define as Active every person that plays.")
    >>> assert isinstance(rule, Implication)
    >>> assert "active" in repr(rule).lower()
    >>> assert "person" in repr(rule)
    >>> assert "plays" in repr(rule)

**End-to-end execution**

Let's build a small engine and verify the rule fires::

    >>> engine = RegionFrontendCPLogicSolver()
    >>> engine.add_extensional_predicate_from_tuples(
    ...     Symbol("person"), [("alice",), ("bob",), ("carol",)]
    ... )
    >>> engine.add_extensional_predicate_from_tuples(
    ...     Symbol("plays"), [("alice",), ("carol",)]
    ... )
    >>> rule = parser("define as Active every person that plays.")
    >>> _ = engine.walk(rule)
    >>> solution = Chase(engine).build_chase_solution()
    >>> assert solution[Symbol("active")].value == {("alice",), ("carol",)}


6. Multi-Variable Rules and Joins
----------------------------------

N-ary rules use ``for every NOUN ; with every NOUN`` (or other prepositions)
to bind multiple variables into the head::

    >>> engine2 = RegionFrontendCPLogicSolver()
    >>> engine2.add_extensional_predicate_from_tuples(
    ...     Symbol("item"), [("a",), ("b",), ("c",)]
    ... )
    >>> engine2.add_extensional_predicate_from_tuples(
    ...     Symbol("item_count"),
    ...     [("a", 0), ("a", 1), ("b", 2), ("c", 3)]
    ... )
    >>> engine2.add_extensional_predicate_from_tuples(
    ...     Symbol("quantity"), [(i,) for i in range(5)]
    ... )
    >>> rule = parser(
    ...     "define as merge for every Item ?i ;"
    ...     " with every Quantity that ?i item_count"
    ... )
    >>> assert isinstance(rule, Implication)
    >>> _ = engine2.walk(rule)
    >>> solution = Chase(engine2).build_chase_solution()
    >>> assert solution[Symbol("merge")].value == {
    ...     ("a", 0), ("a", 1), ("b", 2), ("c", 3)
    ... }


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

    >>> engine3 = RegionFrontendCPLogicSolver()
    >>> engine3.add_extensional_predicate_from_tuples(
    ...     Symbol("item"), [("a",), ("b",), ("c",), ("d",)]
    ... )
    >>> engine3.add_extensional_predicate_from_tuples(
    ...     Symbol("item_count"),
    ...     [("a", 0), ("a", 1), ("b", 2), ("c", 3)]
    ... )
    >>> rule = parser(
    ...     "define as Large every Item "
    ...     "that has an item_count greater equal than 2."
    ... )
    >>> assert isinstance(rule, Implication)
    >>> _ = engine3.walk(rule)
    >>> solution = Chase(engine3).build_chase_solution()
    >>> assert solution[Symbol("large")].value == {("b",), ("c",)}


8. Querying with ``obtain``
----------------------------

The ``obtain`` keyword introduces a **query** rather than a rule.  The parser
returns a ``SquallProgram`` containing a ``Query`` expression that can be
executed against an engine.

::

    >>> from neurolang.expressions import Query
    >>> prog = parser("obtain every Item that has an item_count.")
    >>> assert isinstance(prog, SquallProgram)
    >>> assert len(prog.queries) == 1
    >>> assert isinstance(prog.queries[0], Query)

**End-to-end**::

    >>> engine4 = RegionFrontendCPLogicSolver()
    >>> engine4.add_extensional_predicate_from_tuples(
    ...     Symbol("item"), [("a",), ("b",), ("c",), ("d",)]
    ... )
    >>> engine4.add_extensional_predicate_from_tuples(
    ...     Symbol("item_count"),
    ...     [("a", 0), ("a", 1), ("b", 2), ("c", 3)]
    ... )
    >>> prog = parser("obtain every Item that has an item_count.")
    >>> q = prog.queries[0]
    >>> head_sym = Symbol.fresh()
    >>> _ = engine4.walk(Implication(head_sym(q.head), q.body))
    >>> solution = Chase(engine4).build_chase_solution()
    >>> assert solution[head_sym].value == {("a",), ("b",), ("c",)}

Item ``"d"`` is absent because it has no ``item_count`` entry.


9. Aggregations
----------------

Aggregations summarise a set of values into a single result per group.
The syntax follows the pattern::

    define as RESULT for every SUBJECT ;
        where every AGG_FUNC of the MEASURE where CONDITION per SUBJECT.

Supported aggregation functions: ``count``, ``sum``, ``max``, ``min``,
``average``.

The following rule computes the maximum ``item_count`` value per item::

    >>> engine5 = RegionFrontendCPLogicSolver()
    >>> engine5.add_extensional_predicate_from_tuples(
    ...     Symbol("item"), [("a",), ("b",), ("c",), ("d",)]
    ... )
    >>> engine5.add_extensional_predicate_from_tuples(
    ...     Symbol("item_count"),
    ...     [("a", 0), ("a", 1), ("b", 2), ("c", 3)]
    ... )
    >>> engine5.add_extensional_predicate_from_tuples(
    ...     Symbol("quantity"), [(i,) for i in range(5)]
    ... )
    >>> rule = parser(
    ...     "define as max_items for every Item ?i ;"
    ...     " where every Max of the Quantity where ?i item_count per ?i."
    ... )
    >>> assert isinstance(rule, Implication)
    >>> from neurolang.datalog.aggregation import AggregationApplication
    >>> agg_args = [a for a in rule.consequent.args
    ...             if isinstance(a, AggregationApplication)]
    >>> assert len(agg_args) == 1
    >>> assert agg_args[0].functor == Constant(max)
    >>> _ = engine5.walk(rule)
    >>> solution = Chase(engine5).build_chase_solution()
    >>> assert solution[Symbol("max_items")].value == {
    ...     ("a", 1), ("b", 2), ("c", 3)
    ... }

Item ``"d"`` is absent because it has no ``item_count``.


10. Multiple Rules in One Program
-----------------------------------

Separate rules with a full stop.  The parser returns a ``Union`` of
``Implication`` expressions::

    >>> from neurolang.datalog import Union
    >>> prog = parser(
    ...     "define as Active every person that plays. "
    ...     "define as Fast every person that runs."
    ... )
    >>> assert isinstance(prog, Union)
    >>> assert len(prog.formulas) == 2


11. Boolean Connectives in Relative Clauses
--------------------------------------------

Relative clauses support ``and`` (conjunction) and ``or`` (disjunction).
With conjunction, both conditions must hold for the same individual.
The conjunction binds two verb-phrase conditions directly in the relative
clause; the last verb before the period acts as the main VP::

    >>> result = parser("squall a person that runs and plays")
    >>> assert "person" in repr(result)
    >>> assert "plays" in repr(result)
    >>> assert "runs" in repr(result)

With ``or``, the individual must satisfy at least one of the conditions.
In this case a main VP must follow the disjunction::

    >>> from neurolang.logic import ExistentialPredicate
    >>> result = parser("squall a person that plays or runs walks")
    >>> assert "person" in repr(result)
    >>> assert "plays" in repr(result)
    >>> assert "runs" in repr(result)


12. ``for … , …`` Quantification
----------------------------------

A sentence can be prefixed with ``for NOUN_PHRASE ,`` to bind the outer
variable first::

    >>> result = parser("squall for every person ?p, ?p plays")
    >>> assert isinstance(result, UniversalPredicate)
    >>> assert "p" in repr(result)
    >>> assert "plays" in repr(result)


13. Reserved Words and Quoting
--------------------------------

SQUALL reserves many common English words as keywords (``every``, ``a``,
``the``, ``that``, ``is``, ``has``, ``not``, ``and``, ``or``, ``where``,
``who``, ``which``, etc.).  If a predicate or entity name coincides with a
reserved word, wrap it in backticks::

    >>> result = parser("squall every `from` plays")  # doctest: +ELLIPSIS
    >>> assert "from" in repr(result)

Variable names use the ``?`` prefix and may contain letters, digits, and
underscores::

    >>> result = parser("squall ?study_id reports")
    >>> assert "study_id" in repr(result)

String literals use single quotes and may contain spaces::

    >>> result = parser("squall 'neuro study' plays")
    >>> assert "neuro study" in repr(result)


14. Known Limitations
----------------------

The following constructs are **parsed** by the grammar but their semantics
are not yet fully implemented:

* **Conditioned rules** — ``define as probably … conditioned to …``:
  the conditioning noun phrase is silently discarded.
* **Inverse transitive prefix ``~``** — the ``~`` prefix disambiguates
  transitive verbs (it is required before a transitive verb name to tell the
  parser the verb takes an object) but does **not** reverse argument order.
  ``~author(x, y)`` and ``author(x, y)`` are identical in the IR.
* **``rule_body2_cond``** — the grammar rule for two-sided conditioned NPs
  has no transformer handler and will produce a raw list if reached.

These are tracked as stubs in the module docstring of
``neurolang/frontend/datalog/squall_syntax_lark.py``.
