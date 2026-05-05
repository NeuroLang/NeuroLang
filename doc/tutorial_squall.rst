SQUALL: Controlled English for NeuroLang
========================================

SQUALL (*Semantically controlled Query-Answerable Logical Language*) lets you
write NeuroLang queries and rules in plain English sentences instead of
symbolic Datalog notation.  Under the hood, each sentence is translated to a
NeuroLang logical expression using Montague semantics in
Continuation-Passing Style.

This tutorial is a **SQUALL language reference** — each section shows SQUALL
programs as standalone text.  The section *Running SQUALL Programs from Python*
explains how to execute them from the NeuroLang Python interface.  All examples
are validated with doctests (run with ``pytest --doctest-glob=doc/tutorial_squall.rst``).

.. contents:: Contents
   :local:
   :depth: 2


Setup
-----

All examples assume the ``NeurolangPDL`` frontend is imported:

    >>> from neurolang.frontend import NeurolangPDL


1. Basic Sentences
------------------

The simplest SQUALL sentence consists of a **subject** (a variable or
literal) and a **verb** (a unary predicate).

Variables are written with a ``?`` prefix.  String literals are enclosed in
single quotes.

**Unary query**

.. code-block:: squall

    obtain every plays.

Result: all entities in the ``plays`` relation (e.g. ``alice`` if ``plays`` contains ``("alice",)``).

A **transitive** verb takes an object.  Transitive verbs used as binary
predicates are prefixed with ``~`` to indicate that argument order is
inverted (so ``x ~sings y`` maps to ``sings(y, x)``)::

.. code-block:: squall

    define as Performer every person ?x that a Genre ?y ~sings.

.. code-block:: squall

    obtain every Performer.

Result: ``performer`` contains ``alice`` if ``person`` contains ``alice`` and ``bob``,
``genre`` contains ``("jazz",)``, and ``sings`` contains ``("alice", "jazz")``.


2. Quantifiers
--------------

SQUALL supports four determiners: ``every``, ``a``/``an``/``some``, ``no``,
and ``the``.

**Universal — every**

``every person plays`` means *for all x: if person(x) then plays(x)*::

.. code-block:: squall

    define as Active every person that plays.

.. code-block:: squall

    obtain every Active.

Result: ``active`` contains ``alice`` if ``person`` contains ``alice`` and ``bob``,
and ``plays`` contains ``("alice",)``.

**Existential — a / an / some**

``a person plays`` asserts existence.  In a query, only items with an
associated count are returned::

.. code-block:: squall

    obtain every item ?i that has an item_count ?c.

Result: the query returns ``a`` and ``b`` if ``item`` contains ``a, b, c`` and
``item_count`` maps ``a→1``, ``b→2``. Item ``c`` is absent because it has no
``item_count`` entry.

**Negative — no**

``no`` inside a relative clause expresses negation-as-failure.  Items that
have *no* associated count are returned::

.. code-block:: squall

    obtain every item ?i that has no item_count ?c.

Result: the query returns ``c`` if ``item`` contains ``a, b, c`` and
``item_count`` maps ``a→1``, ``b→2``. Items ``a`` and ``b`` are excluded because
they have associated counts.

**Named variables with quantifiers**

Variables can be named explicitly using ``?name`` labels directly after the
noun.  The label binds the variable so it can be reused elsewhere in the
sentence::

.. code-block:: squall

    define as Active every person ?p that plays.

.. code-block:: squall

    obtain every Active.

Result: ``active`` contains ``alice`` if ``person`` contains ``alice`` and ``bob``,
and ``plays`` contains ``("alice",)``.


3. Relative Clauses
-------------------

Relative clauses restrict the noun they modify.  They are introduced by
``that``, ``which``, ``who``, or ``where``.

**Intransitive VP relative clause**

``every person that plays`` — for every x: person(x) and plays(x)::

.. code-block:: squall

    define as PlayerPerson every person that plays.

.. code-block:: squall

    obtain every PlayerPerson.

Result: ``playerperson`` contains ``alice`` if ``person`` contains ``alice`` and ``bob``,
and ``plays`` contains ``("alice",)``.

**Transitive VP relative clause (passive-like)**

``~verb`` signals a transitive (binary) predicate used in *passive* position:
``a study ~reports voxel`` reads ``reports(study, voxel)`` with argument order
reversed.  The rule below collects (study, voxel) pairs via an explicit
multi-variable head::

.. code-block:: squall

    define as reported for every Study ?s ; with every Voxel ?v that ?s reports.

.. code-block:: squall

    obtain every reported.

Result: ``reported`` contains ``("s1", "v1")`` and ``("s2", "v2")`` if
``voxel`` contains ``v1, v2, v3``, ``study`` contains ``s1, s2``, and
``reports`` contains ``("s1", "v1")`` and ``("s2", "v2")``.

**Nested relative clauses**

Relative clauses can be nested by using an intermediate IDB predicate as the
noun.  The example below defines ``selected_player`` from the intersection of
two independent predicates::

.. code-block:: squall

    define as PlayingSelected every selected that plays.

.. code-block:: squall

    obtain every PlayingSelected.

Result: ``playingselected`` contains ``alice`` if ``person`` contains ``alice``,
``bob``, and ``carol``; ``plays`` contains ``alice`` and ``carol``; and
``selected`` contains ``alice``.

**Negative relative clause**

``does not VP`` expresses negation-as-failure on a unary predicate::

.. code-block:: squall

    define as NotPlaying every person that does not plays.

.. code-block:: squall

    obtain every NotPlaying.

Result: ``notplaying`` contains ``bob`` if ``person`` contains ``alice`` and ``bob``,
and ``plays`` contains ``("alice",)``.

**Possessive relative clause — whose**

``whose NG2 VP`` expresses a possessive relationship via a binary noun.
``define as published every person whose writer plays.`` means: for every
person x, there exists a y such that writer(x, y) and plays(y) — and that
person is ``published``::

.. code-block:: squall

    define as published every person whose writer plays.

.. code-block:: squall

    obtain every published.

Result: ``published`` contains ``alice`` if ``person`` contains ``alice`` and ``bob``,
``writer`` contains ``("alice", "carol")`` and ``("bob", "dave")``, and
``plays`` contains ``("carol",)``.


4. Tuple (Multi-dimensional) Subjects
--------------------------------------

When a noun denotes a multi-dimensional entity (e.g. a voxel with x, y, z
coordinates), a parenthesised tuple of labels can follow the noun.  The
variables bind to the respective columns of the relation::

.. code-block:: squall

    define as active every voxel (?v; ?x; ?y; ?z).

.. code-block:: squall

    obtain every active.

Result: ``active`` contains ``("v1", 0, 0, 1)`` and ``("v2", 1, 2, 3)`` if
``voxel`` contains those tuples.

The compiler generates one binding per coordinate variable and produces a
single conjunction for the body.

**Anonymous wildcard ``_`` in tuple labels**

Use ``_`` inside a tuple label to match a column in the body without
projecting it into the rule head.  Each ``_`` creates a distinct fresh
variable — the column is consumed in the join but dropped from the output.

This is particularly useful when a base relation has more columns than
needed in the derived predicate.  For example, ``peak_reported`` stores
``(i, j, k, study_id)`` but we want ``activation`` to contain only the
three spatial coordinates::

.. code-block:: squall

    define as Activation every Peak_reported (?i; ?j; ?k; _).

.. code-block:: squall

    obtain every Activation.

Result: ``activation`` contains ``(10, 20, 30)`` and ``(11, 21, 31)`` if
``peak_reported`` contains ``(10, 20, 30, "s1")`` and ``(11, 21, 31, "s2")``.
The study-id column is consumed in the join but dropped from the head.

The study-id column is matched in the body by the fresh symbol produced for
``_``, but it does not appear in the ``activation`` head.  Multiple ``_``
wildcards in the same tuple each get a distinct fresh variable.


5. Defining Rules with ``define as``
-------------------------------------

The ``define as`` prefix turns a sentence into a Datalog **rule definition**.

**Simple unary rule**

.. code-block:: squall

    define as Active every person that plays.

.. code-block:: squall

    obtain every Active.

Result: ``active`` contains ``alice`` and ``carol`` if ``person`` contains
``alice``, ``bob``, and ``carol``, and ``plays`` contains ``alice`` and ``carol``.


6. Multi-Variable Rules and Joins
----------------------------------

N-ary rules use ``for every NOUN ; with every NOUN`` (or other prepositions)
to bind multiple variables into the head::

.. code-block:: squall

    define as merge for every Item ?i ; with every Quantity that ?i item_count.

.. code-block:: squall

    obtain every merge.

Result: ``merge`` contains ``("a", 0)``, ``("a", 1)``, ``("b", 2)``, and
``("c", 3)`` if ``item`` contains ``a, b, c``, ``item_count`` contains those
tuples, and ``quantity`` contains ``0`` through ``4``.


6b. Compound Quantifiers and Anaphora
--------------------------------------

When a rule head needs more than one variable, the **compound quantifier**
syntax chains ``for every`` clauses with ``and``, followed by a ``where``
sentence that describes the join condition.  This reads much more naturally
than the semicolon-based multi-variable form::

.. code-block:: squall

    define as Cooccurrence for every Region ?r and for every Term ?t
        where a Selected_study ?s activates ?r and mentions ?t.

.. code-block:: squall

    obtain every Cooccurrence.

Result: ``cooccurrence`` contains ``("A", "x")``, ``("A", "y")``, and ``("B", "x")``
when ``region`` contains ``A`` and ``B``, ``term`` contains ``x`` and ``y``,
``selected_study`` contains ``s1``, ``s2``, ``s3``, ``activates`` contains
``("s1", "A")``, ``("s2", "A")``, ``("s3", "B")``, and ``mentions`` contains
``("s1", "x")``, ``("s2", "y")``, ``("s3", "x")``.

The ``for every Region ?r and for every Term ?t`` part binds ``r`` and
``t`` into the head.  The ``where`` sentence is the rule body; it can
use any of the usual SQUALL constructs (relative clauses, comparisons,
etc.).  Explicit labels after each noun let you refer to the same
variable in the body.

**Anaphoric definite references**

Inside the ``where`` sentence, ``the Noun`` can refer back to the variable
introduced by a preceding ``for every Noun``.  This is called **anaphora**
resolution.  The example below says exactly the same thing as the previous
one but uses ``the Region`` and ``the Term`` instead of explicit labels::

.. code-block:: squall

    define as Cooccurrence
        for every Region and for every Term
        where a Selected_study activates the Region and mentions the Term.

.. code-block:: squall

    obtain every Cooccurrence.

Result: same as above — ``cooccurrence`` contains ``("A", "x")``, ``("A", "y")``,
and ``("B", "x")`` under the same EDB assumptions.

The transformer remembers which variable ``for every Region`` bound and
re-uses that same symbol when ``the Region`` appears later in the sentence.
If ``the Noun`` is used when no matching ``for every Noun`` is in scope, it
falls back to the normal existential behaviour (creating a fresh variable).

.. note::

   Anaphora works within a single rule or query only — there is no
   inter-sentence scope yet.


6c. Probabilistic N-ary Rules
-------------------------------

``with inferred probability``, ``with probability``, and ``probably`` can
now appear on n-ary heads built with compound quantifiers.  The syntax is
identical to the unary case; the engine automatically handles the extra
head variables::

.. code-block:: squall

    define as Joint_prob with inferred probability
        for every Region ?r and for every Term ?t
        where ?r cooccurs ?t.

    obtain every Joint_prob (?r; ?t; ?p).

Result: ``joint_prob`` contains ``("A", "x", 1.0)``, ``("A", "y", 1.0)``,
and ``("B", "x", 1.0)`` when ``region`` contains ``A`` and ``B``, ``term``
contains ``x`` and ``y``, and ``cooccurs`` contains ``("A", "x")``,
``("A", "y")``, and ``("B", "x")``. The ``obtain`` clause returns these
three tuples with inferred probability ``1.0``.

When the body contains existentials (e.g. ``a Selected_study``), use an
intermediate deterministic rule to flatten the body first, then define the
probabilistic rule over the intermediate relation.  This pattern is shown in
Section 15.


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

.. code-block:: squall

    define as Large every Item that has an item_count greater equal than 2.

.. code-block:: squall

    obtain every Large.

Result: ``large`` contains ``b`` and ``c`` if ``item`` contains ``a, b, c, d``
and ``item_count`` maps ``a→0,1``, ``b→2``, ``c→3``.

Item ``"d"`` is absent because it has no ``item_count`` entry.


8. Querying with ``obtain``
----------------------------

The ``obtain`` keyword introduces a **query** rather than a rule.
``execute_squall_program`` returns a ``NamedRelationalAlgebraFrozenSet``
directly::

.. code-block:: squall

    obtain every Item that has an item_count.

Result: the query returns ``a``, ``b``, and ``c`` if ``item`` contains
``a, b, c, d`` and ``item_count`` maps ``a→0,1``, ``b→2``, ``c→3``.

Item ``"d"`` is absent because it has no ``item_count`` entry.

**Mixing rules and queries**

A single program can contain both ``define as`` rules and an ``obtain``
clause::

.. code-block:: squall

    define as Active every person that plays.
    obtain every Active.

Result: the query returns ``alice`` if ``person`` contains ``alice`` and ``bob``
and ``plays`` contains ``("alice",)``.

**MARG queries — conditional probability**

The ``with probability … conditioned to …`` form defines a *marginal* (MARG)
conditional probability relation.  The engine rewrites it into numerator,
denominator and final ratio rules automatically, adding a probability column as
the last argument.

When the conditioned relation has more columns than the MARG head needs, use
``_`` wildcards to drop the extra columns::

.. code-block:: squall

    define as Activation every Peak_reported (?i; ?j; ?k; ?s)
        such that ?s is a Selected_study.

    define as Term_association every Term_in_study_tfidf (?s; ?t; ?tfidf)
        such that ?s is a Selected_study.

    define as Activation_given_term with probability
        every Activation (?i; ?j; ?k; _)
        conditioned to every Term_association (?s; ?t; _) such that ?t is 'auditory'.

Here ``every Activation (?i; ?j; ?k; _)`` matches the 4-column ``activation``
relation (i, j, k, study_id) but projects out only the 3 spatial coordinates
into the MARG head.  Similarly ``every Term_association (?s; ?t; _)`` reads all
three columns of the relation but drops the ``tfidf`` weight from the
conditioning formula.

The relation ``activation_given_term`` will have columns
``(i, j, k, probability)`` where the last column is
``P(activation(i,j,k) | term_association(s,t)  ∧  t = 'auditory')``.

.. note::

   When using MARG with tuple-labeled relations, the arity of the conditioned
   and conditioning noun-phrases must exactly match the corresponding relation
   arities.  Use ``_`` for columns that exist in the body relation but should
   not appear in the head.


9. Aggregations
----------------

Aggregations summarise a set of values into a single result per group.
The syntax follows the pattern::

    define as RESULT for every SUBJECT ;
        where every AGG_FUNC of the MEASURE where CONDITION per SUBJECT.

Supported aggregation functions: ``count``, ``sum``, ``max``, ``min``,
``average``.

The following rule computes the maximum ``item_count`` value per item::

.. code-block:: squall

    define as max_items for every Item ?i ;
        where every Max of the Quantity where ?i item_count per ?i.

.. code-block:: squall

    obtain every max_items.

Result: ``max_items`` contains ``("a", 1)``, ``("b", 2)``, and ``("c", 3)``
if ``item`` contains ``a, b, c, d``, ``quantity`` contains ``0`` through ``4``,
and ``item_count`` maps ``a→0,1``, ``b→2``, ``c→3``. Item ``"d"`` is absent
because it has no ``item_count`` entry.

Item ``"d"`` is absent because it has no ``item_count``.

**Global aggregation (no groupby)**

When no ``per`` clause is given, the aggregation function receives *all free
variables* of the source relation.  Any callable registered in the engine's
symbol table can be used as the aggregation functor.

.. code-block:: squall

    define as Result every Collect_all of the Item.

Result: ``result`` contains the collected output of ``collect_all`` applied over
all tuples from ``item``. This requires ``collect_all`` to be registered as an
aggregation functor in the engine's symbol table.


10. Multiple Rules in One Program
-----------------------------------

Separate rules with a full stop.  The parser processes each rule and walks
them all into the engine::

.. code-block:: squall

    define as Active every person that plays.
    define as Fast every person that runs.

.. code-block:: squall

    obtain every Active.
    obtain every Fast.

Result: ``active`` contains ``alice`` and ``fast`` contains ``bob`` if
``person`` contains ``alice`` and ``bob``, ``plays`` contains ``("alice",)``,
and ``runs`` contains ``("bob",)``.


11. Boolean Connectives in Relative Clauses
--------------------------------------------

Relative clauses support ``and`` (conjunction) and ``or`` (disjunction).
With conjunction, two rules can be combined step by step — define an
intermediate predicate and then constrain further::

.. code-block:: squall

    define as Player every person that plays.
    define as PlayAndRun every Player that runs.

.. code-block:: squall

    obtain every PlayAndRun.

Result: ``playandrun`` contains ``alice`` if ``person`` contains ``alice``,
``bob``, and ``carol``; ``plays`` contains ``alice`` and ``carol``; and
``runs`` contains ``alice`` and ``bob``.

With ``or``, the individual must satisfy at least one of the conditions::

.. code-block:: squall

    define as PlayOrRun every person that plays or runs.

.. code-block:: squall

    obtain every PlayOrRun.

Result: ``playorrun`` contains ``alice`` and ``bob`` if ``person`` contains
``alice``, ``bob``, and ``carol``; ``plays`` contains ``("alice",)``; and
``runs`` contains ``("bob",)``.


12. ``for … , …`` Quantification
----------------------------------

A sentence can be prefixed with ``for NOUN_PHRASE ,`` to bind the outer
variable first.  In ``define as`` rule definitions, the equivalent is to name
the variable explicitly after the noun using the ``?var`` label.  The example
below demonstrates named variable binding, which is the standard way to refer
to a variable in the rule body::

.. code-block:: squall

    define as Active every person ?p that plays.

.. code-block:: squall

    obtain every Active.

Result: ``active`` contains ``alice`` if ``person`` contains ``alice`` and ``bob``
and ``plays`` contains ``("alice",)``.


13. Reserved Words and Quoting
--------------------------------

SQUALL reserves many common English words as keywords (``every``, ``a``,
``the``, ``that``, ``is``, ``has``, ``not``, ``and``, ``or``, ``where``,
``who``, ``which``, etc.).  If a predicate or entity name coincides with a
reserved word, wrap it in backticks::

.. code-block:: squall

    obtain every `from`.

Result: ``from`` contains ``alice`` if the EDB relation ``from`` contains
``("alice",)``.

Variable names use the ``?`` prefix and may contain letters, digits, and
underscores::

.. code-block:: squall

    obtain every study ?study_id.

Result: the query returns ``s001`` and ``s002`` if ``study`` contains those
tuples.

String literals use single quotes and may contain spaces::

.. code-block:: squall

    obtain every study that is 'neuro study'.

Result: the query returns ``neuro study`` if ``study`` contains
``("neuro study",)`` and ``("other",)``.


15. Neuroimaging Domain Examples
----------------------------------

This section mirrors the patterns used in the actual NeuroLang examples
(``examples/squall_examples.py``, ``examples/plot_neurosynth_implementation.py``)
and shows how the same queries are expressed in SQUALL.

**Finding activated voxels reported by studies**

Each study in NeuroSynth reports ``(study, voxel)`` pairs.  We want to
collect every voxel that at least one study has reported as activated.
The predicate ``?s reports ?v`` maps to ``reports(s, v)``::

.. code-block:: squall

    define as Activated every Voxel ?v that a Study ?s reports.

.. code-block:: squall

    obtain every Activated.

Result: ``activated`` contains ``v1`` and ``v2`` if ``study`` contains
``s1, s2, s3``, ``voxel`` contains ``v1, v2, v3``, and ``reports`` contains
``("s1", "v1")``, ``("s2", "v1")``, and ``("s2", "v2")``.

.. note::

   The tilde (``~``) *reverses* argument order.  Use it when the EDB stores
   ``(voxel, study)`` so that ``?v ~reports ?s`` maps to ``reports(v, s)``
   which the engine sees as ``reports(voxel, study)``.

**Filtering by study category (two-rule chain)**

Select a subset of studies by a category predicate, then collect the
voxels those studies report::

.. code-block:: squall

    define as Auditory_voxel every Voxel ?v
        that an Auditory_study ?s reports.

.. code-block:: squall

    obtain every Auditory_voxel.

Result: ``auditory_voxel`` contains ``v1``, ``v2``, and ``v3`` if
``auditory_study`` contains ``s1`` and ``s2``, ``voxel`` contains ``v1, v2, v3``,
and ``reports`` contains ``("s1", "v1")``, ``("s2", "v2")``, and
``("s2", "v3")``.

The two-rule chain pattern is the SQUALL equivalent of:

.. code-block:: text

   auditory_voxel(v) :- voxel(v), auditory_study(s), reports(s, v).

**Atlas region filtering — registering a custom predicate**

Custom predicates registered in the engine's symbol table can be used in
SQUALL body positions.  The example below assumes ``startswith`` is available
as a registered predicate::

.. code-block:: squall

    define as Left_label every Atlas_label ?label that startswith 'L '.

.. code-block:: squall

    obtain every Left_label.

Result: ``left_label`` contains ``"L S_temporal_sup"`` and ``"L G_frontal_sup"``
if ``atlas_label`` contains ``("L S_temporal_sup",)``, ``("R S_temporal_sup",)``,
and ``("L G_frontal_sup",)`` and ``startswith`` is registered as a binary
predicate.

.. note::

   String-literal arguments to arbitrary body predicates are not yet
   supported.  The workaround is to pre-filter the EDB in Python before
   calling ``execute_squall_program``.

**Multi-variable brain activation rule (tuple subject)**

When voxels are stored as ``(x, y, z)`` coordinate triples, the tuple
label syntax binds all three columns at once::

.. code-block:: squall

    define as Activation every Voxel (?x; ?y; ?z)
        that a Study ?s focus_reported.

.. code-block:: squall

    obtain every Activation.

Result: ``activation`` contains ``(0, 1, 2)`` and ``(3, 4, 5)`` if ``study``
contains ``s1`` and ``s2``, ``voxel`` contains ``(0, 1, 2)``, ``(3, 4, 5)``,
and ``(6, 7, 8)``, and ``focus_reported`` contains ``("s1", 0, 1, 2)`` and
``("s2", 3, 4, 5)``.

The tuple subject ``(?x; ?y; ?z)`` binds the three coordinate columns of
``voxel`` and re-uses those variables in the ``focus_reported`` body
(column order: study, x, y, z).

**Conditional probability (MARG) — activation probability**

The MARG form computes the conditional probability that a voxel is activated
given a conditioning predicate.  The full NeuroSynth forward-model pattern
(see ``examples/plot_squall_neurosynth.py``) is:

.. code-block:: squall

    define as Activation every Peak_reported (?i; ?j; ?k; ?s)
        such that ?s is a Selected_study.

    define as Term_association every Term_in_study_tfidf (?s; ?t; ?tfidf)
        such that ?s is a Selected_study.

    define as Activation_given_term with probability
        every Activation (?i; ?j; ?k; _)
        conditioned to every Term_association (?s; ?t; _) such that ?t is 'auditory'.

    define as Activation_given_term_image
        every Agg_create_region_overlay of the Activation_given_term (?i; ?j; ?k; ?p).

    obtain every Activation_given_term_image (?x).

Key points:

* ``(?i; ?j; ?k; _)`` on the conditioned side — the ``_`` drops the
  study-id column so the MARG head is ``(i, j, k, PROB(i,j,k))``.
* ``(?s; ?t; _)`` on the conditioning side — the ``_`` drops the
  ``tfidf`` weight column; only ``(s, t)`` appear in the conditioning formula.
* The ``obtain`` clause at the end causes ``execute_squall_program`` to return
  the query result directly as a ``NamedRelationalAlgebraFrozenSet``.

.. note::

   A full probabilistic solve requires a CPLogic-compatible EDB loaded via
   ``add_uniform_probabilistic_choice_over_set``.  See
   ``examples/plot_squall_neurosynth.py`` for the complete runnable example.

**Compound quantifier example — co-occurrence of region and term**

The following pattern (taken from ``examples/plot_squall_bayes_factor_decoding.py``)
uses compound quantifiers and anaphora to express a ternary join in plain
English with zero explicit variables::

.. code-block:: squall

    define as Cooccurrence
        for every Region and for every Term
        where a Selected_study activates the Region and mentions the Term.

    define as Joint_probability with inferred probability
        every Cooccurrence (?r; ?t).

* ``for every Region and for every Term`` binds both variables into the head.
* ``the Region`` and ``the Term`` are resolved anaphorically — they refer back
to the variables bound by the two ``for every`` clauses, so no explicit labels
are needed.
* ``activates the Region`` maps to ``activates(s, r)`` and ``mentions the Term``
maps to ``mentions(s, t)``; the study variable ``s`` is introduced by
``a Selected_study`` and exists only in the body.
* The intermediate ``cooccurrence`` rule has a flat head, so it can be queried
with the standard probabilistic solver.  The ``joint_probability`` rule then
adds the marginal probability column.


16. Missing SQUALL Syntax — Gap Report
-----------------------------------------

The following Datalog / IR patterns appear in codebase examples with their
current status as of 2026-04-20:

.. list-table:: SQUALL gap report (updated 2026-04-20)
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
   * - Anonymous wildcard ``_`` in tuple labels (``(?i; ?j; ?k; _)``)
     - ✅ Fixed
     - Each ``_`` creates a distinct fresh variable matched in the body but dropped from the head; works in both conditioned and conditioning NPs of MARG rules
   * - Variable/expression as explicit probability (``with probability ?p``)
     - ✅ Fixed
     - ``vpdo_explicit_prob_v1/vn`` now accept any NP including labels
   * - ``obtain`` clause returning results directly
     - ✅ Fixed
     - ``execute_squall_program`` returns a ``NamedRelationalAlgebraFrozenSet`` when a single ``obtain`` is present
   * - Compound quantifiers (`for every X and for every Y where …`)
     - ✅ Fixed
     - Added ``rule_body2``, ``quant_list``, ``quant_clause`` grammar; ``rule_opnn_compound`` transformer
   * - Anaphoric definite references (`the Noun` → bound variable)
     - ✅ Fixed
     - ``_symbol_scope`` tracks noun-to-variable mapping per rule; ``det_the`` resolves from scope
   * - Probabilistic n-ary predicates (`with inferred probability` on n-ary heads)
     - ✅ Fixed
     - ``rule_opnn_prob``, ``rule_opnn_marg``, ``rule_opnn_per_compound`` handlers; no engine changes needed
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
