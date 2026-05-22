SQUALL: Controlled English for NeuroLang
========================================

SQUALL (*Semantically controlled Query-Answerable Logical Language*) lets you
write NeuroLang queries and rules in plain English sentences instead of
symbolic Datalog notation.  Under the hood, each sentence is translated to a
NeuroLang logical expression using Montague semantics in
Continuation-Passing Style.

This tutorial is a **SQUALL language reference** — each section shows SQUALL
programs as standalone text.  All examples are validated with doctests
(run with ``pytest --doctest-glob=doc/tutorial_squall.rst``).

.. contents:: Contents
   :local:
   :depth: 2


Part 1: Getting Started
========================

1.1 Setup
----------

All examples assume the ``NeurolangPDL`` frontend is imported:

    >>> from neurolang.frontend import NeurolangPDL

1.2 Your First Query
---------------------

The simplest SQUALL sentence consists of a **subject** (a variable or
literal) and a **verb** (a unary predicate).

.. code-block:: squall

    obtain every plays.

Result: all entities in the ``plays`` relation.

1.3 Running from Python
------------------------

The general workflow is:

1. Create a ``NeurolangPDL`` engine.
2. Register EDB facts with ``add_tuple_set``.
3. Execute a SQUALL program string with ``execute_squall_program``.
4. Inspect results with ``solve_all()`` or via the direct return from
   ``obtain`` queries.

Full Python API details are in `appendix-a`_.


Part 2: Nouns and Quantification
==================================

2.1 Determiners
----------------

SQUALL supports four determiners: ``every``, ``a``/``an``/``some``, ``no``,
and ``the``.

**Universal — every**

``every person plays`` means *for all x: if person(x) then plays(x)*.

.. code-block:: squall

    define as Active every person that plays.

.. code-block:: squall

    obtain every Active.

Result: ``active`` contains ``alice`` if ``person`` contains ``alice`` and ``bob``,
and ``plays`` contains ``("alice",)``.

**Existential — a / an / some**

``a person plays`` asserts existence.  In a query, only items with an
associated count are returned.

.. code-block:: squall

    obtain every item ?i that has an item_count ?c.

Result: the query returns ``a`` and ``b`` if ``item`` contains ``a, b, c`` and
``item_count`` maps ``a→1``, ``b→2``.

**Negative — no**

``no`` inside a relative clause expresses negation-as-failure.

.. code-block:: squall

    obtain every item ?i that has no item_count ?c.

Result: the query returns ``c`` if ``item`` contains ``a, b, c`` and
``item_count`` maps ``a→1``, ``b→2``.

2.2 Named Variables — ``?label``
---------------------------------

Variables can be named explicitly using ``?name`` labels directly after the
noun.  The label binds the variable so it can be reused elsewhere in the
sentence.

.. code-block:: squall

    define as Active every person ?p that plays.

.. code-block:: squall

    obtain every Active.

Result: ``active`` contains ``alice`` if ``person`` contains ``alice`` and ``bob``,
and ``plays`` contains ``("alice",)``.

2.3 Tuple Subjects and Wildcard ``_``
--------------------------------------

When a noun denotes a multi-dimensional entity, a parenthesised tuple of
labels can follow the noun.  The variables bind to the respective columns.

.. code-block:: squall

    define as active every voxel (?v; ?x; ?y; ?z).

.. code-block:: squall

    obtain every active.

Result: ``active`` contains ``("v1", 0, 0, 1)`` and ``("v2", 1, 2, 3)`` if
``voxel`` contains those tuples.

Use ``_`` inside a tuple label to match a column without projecting it into
the rule head.  Each ``_`` creates a distinct fresh variable.

.. code-block:: squall

    define as Activation every Peak_reported (?i; ?j; ?k; _).

.. code-block:: squall

    obtain every Activation.

Result: ``activation`` contains ``(10, 20, 30)`` and ``(11, 21, 31)`` if
``peak_reported`` contains ``(10, 20, 30, "s1")`` and ``(11, 21, 31, "s2")``.

2.4 Possessive NP — ``NP2 of NP``
-----------------------------------

A binary noun (``noun2``) can be turned into a full noun phrase with
``DET noun2 of NP``.  The possessor is the outer NP; the possession is
quantified by the determiner.

.. code-block:: squall

    define as AuthoredPaper every paper that a writer of every author writes.

.. code-block:: squall

    obtain every AuthoredPaper.

Result: ``authoredpaper`` contains ``p1`` if ``paper`` contains ``p1``,
``author`` contains ``alice``, and ``writer`` contains
``("alice", "p1")``.

2.5 Dimensional Annotation — ``in ND``
---------------------------------------

A noun phrase can be annotated with ``in ND`` (e.g. ``in 3D``) to
document that the relation is *N*-dimensional.  The annotation is
syntactically accepted but carries no additional semantic content —
it serves as human-readable documentation.

.. code-block:: squall

    define as ActiveVoxel every Voxel in 3D that a Study reports.

.. code-block:: squall

    obtain every ActiveVoxel.

Result: ``activevoxel`` contains every voxel reported by at least one study.
The ``in 3D`` annotation is ignored by the engine.

2.6 Reserved Words and Quoting
--------------------------------

SQUALL reserves many common English words as keywords (``every``, ``a``,
``the``, ``that``, ``is``, ``has``, ``not``, ``and``, ``or``, ``where``,
``who``, ``which``, etc.).  If a predicate or entity name coincides with a
reserved word, wrap it in backticks.

.. code-block:: squall

    obtain every `from`.

Result: ``from`` contains ``alice`` if the EDB relation ``from`` contains
``("alice",)``.

Variable names use the ``?`` prefix and may contain letters, digits, and
underscores.

.. code-block:: squall

    obtain every study ?study_id.

Result: the query returns ``s001`` and ``s002`` if ``study`` contains those
tuples.

String literals use single quotes and may contain spaces.

.. code-block:: squall

    obtain every study that is 'neuro study'.

Result: the query returns ``neuro study`` if ``study`` contains
``("neuro study",)`` and ``("other",)``.


Part 3: Verbs
==============

3.1 Intransitive Verbs
-----------------------

The simplest verb is an **intransitive** predicate: one argument (the subject).

.. code-block:: squall

    define as PlayerPerson every person that plays.

.. code-block:: squall

    obtain every PlayerPerson.

Result: ``playerperson`` contains ``alice`` if ``person`` contains ``alice`` and ``bob``,
and ``plays`` contains ``("alice",)``.

.. _part-3-verb-args:

3.2 Transitive Verbs
--------------------

A **transitive** verb takes an object noun phrase following it, just as in
English.  The verb applies to its subject (the head noun) and its object
(the noun phrase after the verb) in natural subject-verb-object order.

.. code-block:: squall

    define as Performer every person that sings a Genre.

.. code-block:: squall

    obtain every Performer.

Result: ``performer`` contains ``alice`` if ``person`` contains ``alice`` and ``bob``,
``genre`` contains ``("jazz",)``, and ``sings`` contains ``("alice", "jazz")``.

The rule reads: *"every person that sings a genre"* — ``sings(person, genre)``
maps the subject (``person``) to the first argument and the object (``genre``)
to the second argument of the binary ``sings`` predicate.

.. note::

   When the EDB predicate stores its arguments in the **reverse** order
   (e.g. ``reports`` stores ``(study, voxel)`` and you want to query from
   the voxel's perspective), use the ``~`` prefix to invert the argument
   order::

       every Voxel that a Study ~reports.

   reads as *"a study reports a voxel"* in English, but maps to
   ``reports(voxel, study)`` — the ``~`` swaps the arguments so the
   subject (``Voxel``) becomes the second argument and the object
   (``Study``) becomes the first.

.. _3-3-auxiliaries:

3.3 Auxiliaries — ``does`` / ``is`` / ``has``
----------------------------------------------

``does not VP`` expresses negation-as-failure on a unary predicate.

.. code-block:: squall

    define as NotPlaying every person that does not plays.

.. code-block:: squall

    obtain every NotPlaying.

Result: ``notplaying`` contains ``bob`` if ``person`` contains ``alice`` and ``bob``,
and ``plays`` contains ``("alice",)``.

3.4 Possessive VP — ``has NP2``
---------------------------------

``has DET noun2`` expresses a possessive verb phrase.  The subject *has* a
thing related to it by the binary noun ``noun2``.

.. code-block:: squall

    define as Author every person that has a publication.

.. code-block:: squall

    obtain every Author.

Result: ``author`` contains ``alice`` if ``person`` contains ``alice``
and ``bob``, and ``publication`` (a binary relation) contains
``("alice", "paper1")``.

With an optional relative clause on the possessed noun:

.. code-block:: squall

    define as ProlificAuthor every person
        that has a publication that is highly_cited.

Result: persons for whom at least one entry in ``publication`` has a
related ``highly_cited`` fact.

3.5 Existential — ``there is NP``
-----------------------------------

``there is NP`` / ``there are NP`` asserts that at least one entity
matching the noun phrase exists.  The sentence is true whenever the NP is
non-empty.

.. code-block:: squall

    define as HasPlayer every Game that there is a Player.

.. code-block:: squall

    obtain every HasPlayer.

Result: ``hasplayer`` contains ``chess`` if ``game`` contains ``chess``
and ``go``, ``player`` contains ``("alice", "chess")``, but ``go`` has
no players.

3.7 Arithmetic Expressions
----------------------------

A ``?label is <expression>`` clause can assign the result of an arithmetic
expression to a variable.  The expression supports ``+``, ``-``, ``*``, ``/``
with standard operator precedence; parentheses are supported for grouping.

.. code-block:: squall

    define as Bayes_factor (?r; ?t; ?bf)
        where Joint_probability (?r, ?t, ?p_rt)
        and Region_probability (?r, ?p_r)
        and Term_probability (?t, ?p_t)
        and ?bf is (?p_rt / ?p_r) / ((?p_t - ?p_rt) / (1.0 - ?p_r)).

Result: ``bayes_factor`` contains ``("region_A", "term_x", 6.5)`` if the
joint probability is 0.6, the region probability is 0.3, and the term
probability is 0.8.

The ``is`` clause translates to an ``eq`` builtin with the arithmetic
expression tree as the second argument.  The expression is evaluated during
the chase using Python's ``operator`` module functions (``truediv``,
``sub``, etc.).

A full runnable example is in
``examples/plot_squall_bayes_factor_decoding.py``.

.. note::

    Arithmetic expressions currently support **numeric types only**.
    Non-numeric ``?label is 'string'`` is handled as a constant equality
    (see section 3.6 above).  The two uses share the same ``is`` keyword
    but produce different internal representations.

3.6 Inline Type Guard — ``where ?x is a Noun``
------------------------------------------------

Inside a relative clause, ``where (?i; ?j; ?k) is a Noun`` asserts that
the tuple belongs to the relation named by ``Noun``.

.. code-block:: squall

    define as SelectedPeak every Peak_reported (?i; ?j; ?k; ?s)
        where (?i; ?j; ?k; ?s) is a Activation.

.. code-block:: squall

    obtain every SelectedPeak.

Result: ``selectedpeak`` contains every tuple ``(i, j, k, s)`` that
appears in both ``peak_reported`` and ``activation``.

The scalar form works too: ``where ?s is a Selected_study`` asserts
that ``s`` belongs to ``selected_study``.


Part 4: Relative Clauses
==========================

Relative clauses restrict the noun they modify.  They are introduced by
``that``, ``which``, ``who``, or ``where``.

4.1 Intransitive VP
--------------------

``every person that plays`` — for every x: person(x) and plays(x).

.. code-block:: squall

    define as PlayerPerson every person that plays.

.. code-block:: squall

    obtain every PlayerPerson.

4.2 Transitive VP
------------------

A transitive verb takes an object NP after it in natural order (see
:ref:`part-3-verb-args`).  The rule below collects (study, voxel) pairs
via an explicit multi-variable head.

.. code-block:: squall

    define as reported for every Study ?s ; with every Voxel ?v that ?s reports.

.. code-block:: squall

    obtain every reported.

Result: ``reported`` contains ``("s1", "v1")`` and ``("s2", "v2")`` if
``voxel`` contains ``v1, v2, v3``, ``study`` contains ``s1, s2``, and
``reports`` contains ``("s1", "v1")`` and ``("s2", "v2")``.

4.3 Negative — ``does not VP``
--------------------------------

.. code-block:: squall

    define as NotPlaying every person that does not plays.

.. code-block:: squall

    obtain every NotPlaying.

Result: ``notplaying`` contains ``bob`` if ``person`` contains ``alice`` and ``bob``
and ``plays`` contains ``("alice",)``.

4.4 Possessive — ``whose NG2 VP`` and ``NP2 of which VP``
-----------------------------------------------------------

**whose NG2 VP**

``every person whose writer plays`` means: for every person x, there exists
a y such that writer(x, y) and plays(y).

.. code-block:: squall

    define as published every person whose writer plays.

.. code-block:: squall

    obtain every published.

Result: ``published`` contains ``alice`` if ``person`` contains ``alice`` and ``bob``,
``writer`` contains ``("alice", "carol")`` and ``("bob", "dave")``, and
``plays`` contains ``("carol",)``.

**NP2 of which VP**

``NP2 of which VP`` is the *relative* form of possessive NPs.  It attaches to a
head noun and reads: *"whose NP2 VP"*.

.. code-block:: squall

    define as ActiveGame every game, a player of which plays.

.. code-block:: squall

    obtain every ActiveGame.

Result: ``activegame`` contains ``chess`` if ``game`` contains ``chess``
and ``go``, ``player`` contains ``("chess", "alice")``, and
``plays`` contains ``("alice",)``.

4.5 Adjective — ``that is ADJ``
---------------------------------

An identifier used in adjective position (after ``that is``) acts as a unary
predicate.

.. code-block:: squall

    define as ActivePerson every person that is active.

.. code-block:: squall

    obtain every ActivePerson.

Result: ``activeperson`` contains ``alice`` if ``person`` contains
``alice`` and ``bob``, and ``active`` contains ``("alice",)``.

4.6 Function-Call Guard — ``func(?x, ?y) holds``
--------------------------------------------------

An arbitrary relation can be invoked in a relative clause with explicit
label arguments using ``identifier(?label, ...) holds``.

.. code-block:: squall

    define as Close every Pair ?p that euclidean(?x, ?y) holds.

.. code-block:: squall

    obtain every Close.

Result: ``close`` contains all pairs ``p`` for which
``euclidean(x, y)`` holds in the EDB.

4.7 Nested Relative Clauses
-----------------------------

Relative clauses can be nested by using an intermediate IDB predicate as the
noun.

.. code-block:: squall

    define as PlayingSelected every selected that plays.

.. code-block:: squall

    obtain every PlayingSelected.

Result: ``playingselected`` contains ``alice`` if ``person`` contains ``alice``,
``bob``, and ``carol``; ``plays`` contains ``alice`` and ``carol``; and
``selected`` contains ``alice``.

4.8 Comparisons
----------------

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

The following rule selects items whose ``item_count`` is at least 2.

.. code-block:: squall

    define as Large every Item that has an item_count greater equal than 2.

.. code-block:: squall

    obtain every Large.

Result: ``large`` contains ``b`` and ``c`` if ``item`` contains ``a, b, c, d``
and ``item_count`` maps ``a→0,1``, ``b→2``, ``c→3``.


Part 5: Connectives
====================

5.1 Conjunction — ``and``
--------------------------

With conjunction, two predicates are combined with ``and``.  A two-step
approach: define an intermediate predicate and constrain further.

.. code-block:: squall

    define as Player every person that plays.
    define as PlayAndRun every Player that runs.

.. code-block:: squall

    obtain every PlayAndRun.

Result: ``playandrun`` contains ``alice`` if ``person`` contains ``alice``,
``bob``, and ``carol``; ``plays`` contains ``alice`` and ``carol``; and
``runs`` contains ``alice`` and ``bob``.

5.2 Disjunction — ``or``
--------------------------

With ``or``, the individual must satisfy at least one of the conditions.

.. code-block:: squall

    define as PlayOrRun every person that plays or runs.

.. code-block:: squall

    obtain every PlayOrRun.

Result: ``playorrun`` contains ``alice`` and ``bob`` if ``person`` contains
``alice``, ``bob``, and ``carol``; ``plays`` contains ``("alice",)``; and
``runs`` contains ``("bob",)``.

5.3 Negation — ``not``
-----------------------

``does not VP`` expresses negation-as-failure.  See `3-3-auxiliaries`_ for
details and examples.


Part 6: Rules
==============

6.1 Simple Unary Rules
-----------------------

The ``define as`` prefix turns a sentence into a Datalog **rule definition**.

.. code-block:: squall

    define as Active every person that plays.

.. code-block:: squall

    obtain every Active.

Result: ``active`` contains ``alice`` and ``carol`` if ``person`` contains
``alice``, ``bob``, and ``carol``, and ``plays`` contains ``alice`` and ``carol``.

6.2 Multi-Variable Rules
-------------------------

N-ary rules use ``for every NOUN ; with every NOUN`` (or other prepositions)
to bind multiple variables into the head.

.. code-block:: squall

    define as merge for every Item ?i ; with every Quantity that ?i item_count.

.. code-block:: squall

    obtain every merge.

Result: ``merge`` contains ``("a", 0)``, ``("a", 1)``, ``("b", 2)``, and
``("c", 3)`` if ``item`` contains ``a, b, c``, ``item_count`` contains those
tuples, and ``quantity`` contains ``0`` through ``4``.

**Compound quantifiers**

When a rule head needs more than one variable, the **compound quantifier**
syntax chains ``for every`` clauses with ``and``, followed by a ``where``
sentence.

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

6.3 Anaphora — ``the Noun``
-----------------------------

Inside the ``where`` sentence, ``the Noun`` can refer back to the variable
introduced by a preceding ``for every Noun``.  This is called **anaphora**
resolution.

.. code-block:: squall

    define as Cooccurrence
        for every Region and for every Term
        where a Selected_study activates the Region and mentions the Term.

.. code-block:: squall

    obtain every Cooccurrence.

Result: same as the compound quantifier example above — ``cooccurrence`` contains
``("A", "x")``, ``("A", "y")``, and ``("B", "x")``.

.. note::

   Anaphora works within a single rule or query only — there is no
   inter-sentence scope yet.

6.4 Fork Quantification — ``for NP , S``
-----------------------------------------

The ``for NP , S`` (fork) construction places a noun phrase as an
**outer sentence-level quantifier** over an otherwise independent
sentence ``S``.

.. code-block:: squall

    for every Person ?p, ?p plays.

This reads: *"for every person p, p plays"*.

The fork form is especially useful when ``S`` is a complex sentence
that is awkward to embed inside a relative clause.

.. code-block:: squall

    define as Reported
        for every Study ?s,
            a Voxel ?v that ?s reports.

.. code-block:: squall

    obtain every Reported.

Result: ``reported`` contains ``("s1", "v1")`` and ``("s2", "v2")`` if
``study`` contains ``s1`` and ``s2``, ``voxel`` contains ``v1``, ``v2``,
``v3``, and ``reports`` contains ``("s1", "v1")`` and ``("s2", "v2")``.

6.5 Multiple Rules in One Program
-----------------------------------

Separate rules with a full stop.  The parser processes each rule and walks
them all into the engine.

.. code-block:: squall

    define as Active every person that plays.
    define as Fast every person that runs.

.. code-block:: squall

    obtain every Active.
    obtain every Fast.

Result: ``active`` contains ``alice`` and ``fast`` contains ``bob`` if
``person`` contains ``alice`` and ``bob``, ``plays`` contains ``("alice",)``,
and ``runs`` contains ``("bob",)``.

6.6 Bare Predicate Calls in Rule Bodies
-----------------------------------------

A rule body can call any predicate (EDB or IDB) directly using the syntax
``PredicateName (?arg1, ?arg2, ...)`` — no verb, no relative clause, no
anaphora.  This is useful when a rule needs to join several predicates with
explicit variable bindings, especially for arithmetic expressions that
reference the variables.

.. code-block:: squall

    define as Bayes_factor (?r; ?t; ?bf)
        where Joint_probability (?r, ?t, ?p_rt)
        and Region_probability (?r, ?p_r)
        and Term_probability (?t, ?p_t)
        and ?bf is (?p_rt / ?p_r) / ((?p_t - ?p_rt) / (1.0 - ?p_r)).

Here ``Joint_probability (?r, ?t, ?p_rt)`` is a bare predicate call —
it binds ``?p_rt`` to the probability column of the ``joint_probability``
relation for the given region ``?r`` and term ``?t``.  The predicate name
matches the rule name defined elsewhere in the program (case-insensitive).

The arguments use **comma** separators ``(a, b, c)``, matching the
convention for rule head variables.

.. note::

    Bare predicate calls complement the compound quantifier and anaphora
    patterns (sections 6.2–6.3).  Use bare calls when you need explicit
    variable bindings across multiple predicates, and anaphora when you
    want the join to be implicit through natural language.


Part 7: Queries
================

7.1 Simple ``obtain``
----------------------

The ``obtain`` keyword introduces a **query** rather than a rule.
``execute_squall_program`` returns a ``NamedRelationalAlgebraFrozenSet``
directly.

.. code-block:: squall

    obtain every Item that has an item_count.

Result: the query returns ``a``, ``b``, and ``c`` if ``item`` contains
``a, b, c, d`` and ``item_count`` maps ``a→0,1``, ``b→2``, ``c→3``.

7.2 ``obtain … as Name``
-------------------------

An ``obtain`` result can be renamed with ``as Name``.

.. code-block:: squall

    obtain every Joint_prob (?r; ?t; ?p) as P.

Result: the result relation is named ``P``.

7.3 Mixing Rules and Queries
-----------------------------

A single program can contain both ``define as`` rules and an ``obtain``
clause.

.. code-block:: squall

    define as Active every person that plays.
    obtain every Active.

Result: the query returns ``alice`` if ``person`` contains ``alice`` and ``bob``
and ``plays`` contains ``("alice",)``.


Part 8: Probabilistic Rules
=============================

8.1 Probabilistic Facts — ``probably``
----------------------------------------

The ``probably`` keyword creates a probabilistic fact with a fresh
probability variable.

.. code-block:: squall

    define as probably activates every study.

Result: ``activates`` is a probabilistic predicate; the probability is
inferred at query time.

8.2 Explicit Probability — ``with probability NP``
----------------------------------------------------

``with probability`` followed by a conditioned/conditioning noun-phrase pair
defines a *marginal* (MARG) conditional probability relation.

.. code-block:: squall

    define as Activation_given_term with probability
        every Activation (?i; ?j; ?k; _)
        conditioned to every Term_association (?s; ?t; _) such that ?t is 'auditory'.

The relation ``activation_given_term`` will have columns
``(i, j, k, probability)`` where the last column is
``P(activation(i,j,k) | term_association(s,t) ∧ t = 'auditory')``.

The keyword ``given`` is a synonym for ``conditioned to``, and is often
more natural for spatial-prior and atlas-filtering patterns:

.. code-block:: squall

    define as Activation_map with inferred probability every Active_voxel (?i; ?j; ?k; _)
        given every Study_term (_; ?t) where ?t is 'emotion'.

This reads: *"the inferred probability of activation at (i, j, k) given
the study term is 'emotion'"*.  The ``_`` in each tuple label drops the
study-id column from the respective side.

.. note::

   When using MARG with tuple-labeled relations, the arity of the conditioned
   and conditioning noun-phrases must exactly match the corresponding relation
   arities.  Use ``_`` for columns that exist in the body relation but should
   not appear in the head.

8.3 Inferred Probability — ``with inferred probability``
---------------------------------------------------------

``with inferred probability`` on a compound-quantifier head generates a
marginal probability over the joint distribution.

.. code-block:: squall

    define as Joint_prob with inferred probability
        for every Region and for every Term
        where the Region cooccurs the Term.

    obtain every Joint_prob (?r; ?t; ?p).

Result: ``joint_prob`` contains ``("A", "x", 1.0)``, ``("A", "y", 1.0)``,
and ``("B", "x", 1.0)`` when ``region`` contains ``A`` and ``B``, ``term``
contains ``x`` and ``y``, and ``cooccurs`` contains ``("A", "x")``,
``("A", "y")``, and ``("B", "x")``.

8.4 Probabilistic N-ary Rules
-------------------------------

The ``probably`` keyword can prefix n-ary (multi-argument) head rules
using the semicolon body form.  This generates an
``Implication(ProbabilisticFact(p, head), body)`` where ``p`` is a
fresh probability variable inferred at query time.

.. code-block:: squall

    define as probably Cooccurrence for every Study ?s ; with every Term ?t
        that ?s mentions.

.. code-block:: squall

    obtain every Cooccurrence.

Result: ``cooccurrence`` contains probabilistic ``(s, t)`` pairs for
every study that mentions a term.  The probability column is added
automatically.

8.5 Caveats — Existentials in Bodies
--------------------------------------

When the body contains existentials (e.g. ``a Selected_study``), use an
intermediate deterministic rule to flatten the body first, then define the
probabilistic rule over the intermediate relation.  This pattern is shown
in `part-11`_.


Part 9: Aggregations
=====================

Aggregations summarise a set of values into a single result per group.
The syntax follows the pattern::

    define as RESULT for every SUBJECT ;
        where every AGG_FUNC of the MEASURE where CONDITION per SUBJECT.

Supported aggregation functions: ``count``, ``sum``, ``max``, ``min``,
``average``.

9.1 Grouped — ``per GROUPBY``
-------------------------------

The following rule computes the maximum ``item_count`` value per item.

.. code-block:: squall

    define as max_items for every Item ?i ;
        where every Max of the Quantity where ?i item_count per ?i.

.. code-block:: squall

    obtain every max_items.

Result: ``max_items`` contains ``("a", 1)``, ``("b", 2)``, and ``("c", 3)``
if ``item`` contains ``a, b, c, d``, ``quantity`` contains ``0`` through ``4``,
and ``item_count`` maps ``a→0,1``, ``b→2``, ``c→3``.

9.2 Global — No Groupby
------------------------

When no ``per`` clause is given, the aggregation function receives *all free
variables* of the source relation.  Any callable registered in the engine's
symbol table can be used as the aggregation functor.

.. code-block:: squall

    define as Result every Collect_all of the Item.

Result: ``result`` contains the collected output of ``collect_all`` applied over
all tuples from ``item``. This requires ``collect_all`` to be registered as an
aggregation functor in the engine's symbol table.


Part 10: Program-Level Features
=================================

10.1 Directives — ``#name(args).``
------------------------------------

A SQUALL program may include directive lines of the form
``#name(arg, ...)`` to pass structured metadata or configuration to the
engine.  Directives are parsed as ``FunctionApplication(name, args)``
and processed by ``execute_squall_program`` before rule walking.

The following directive is currently supported:

``#set_backend('backend')``
    Switch the relational algebra backend before walking the rules.
    ``backend`` may be ``'pandas'``, ``'dask'``, or ``'duckdb'``.

.. code-block:: squall

    #set_backend('pandas').
    define as Active every person that plays.
    obtain every Active.

Directives are written with a leading ``#``, a lowercase name, and a
parenthesised argument list using the same term syntax as the rest of
SQUALL (labels, literals, or identifiers).  The trailing ``.`` follows
the normal sentence rule.  Unknown directives are silently ignored.


.. _part-11:

Part 11: Neuroimaging Domain Examples
=======================================

This section mirrors the patterns used in the actual NeuroLang examples
(``examples/squall_examples.py``, ``examples/plot_neurosynth_implementation.py``)
and shows how the same queries are expressed in SQUALL.

11.1 Activated Voxels
----------------------

Each study in NeuroSynth reports ``(study, voxel)`` pairs.  We want to
collect every voxel that at least one study has reported as activated.

.. code-block:: squall

    define as Activated every Voxel that a Study reports.

.. code-block:: squall

    obtain every Activated.

Result: ``activated`` contains ``v1`` and ``v2`` if ``study`` contains
``s1, s2, s3``, ``voxel`` contains ``v1, v2, v3``, and ``reports`` contains
``("s1", "v1")``, ``("s2", "v1")``, and ``("s2", "v2")``.

.. note::

    The tilde (``~``) *reverses* argument order (see :ref:`part-3-verb-args`).
    Use it when the EDB stores ``(voxel, study)`` so that ``?v ~reports ?s``
    maps to ``reports(v, s)`` which the engine sees as ``reports(voxel, study)``.

11.2 Filtering by Study Category
----------------------------------

Select a subset of studies by a category predicate, then collect the
voxels those studies report.

.. code-block:: squall

    define as Auditory_voxel every Voxel
        that an Auditory_study reports.

.. code-block:: squall

    obtain every Auditory_voxel.

Result: ``auditory_voxel`` contains ``v1``, ``v2``, and ``v3`` if
``auditory_study`` contains ``s1`` and ``s2``, ``voxel`` contains ``v1, v2, v3``,
and ``reports`` contains ``("s1", "v1")``, ``("s2", "v2")``, and
``("s2", "v3")``.

The two-rule chain pattern is the SQUALL equivalent of:

.. code-block:: text

   auditory_voxel(v) :- voxel(v), auditory_study(s), reports(s, v).

11.3 Atlas Region Filtering
-----------------------------

Custom predicates registered in the engine's symbol table can be used in
SQUALL body positions.

.. code-block:: squall

    define as Left_label every Atlas_label ?label that startswith 'L '.

.. code-block:: squall

    obtain every Left_label.

Result: ``left_label`` contains ``"L S_temporal_sup"`` and ``"L G_frontal_sup"``
if ``atlas_label`` contains ``("L S_temporal_sup",)``, ``("R S_temporal_sup",)``,
and ``("L G_frontal_sup",)`` and ``startswith`` is registered as a binary
predicate.



11.4 Multi-Variable Brain Activation
--------------------------------------

When voxels are stored as ``(x, y, z)`` coordinate triples, the tuple
label syntax binds all three columns at once.

.. code-block:: squall

    define as Activation every Voxel (?x; ?y; ?z)
        that a Study focus_reported.

.. code-block:: squall

    obtain every Activation.

Result: ``activation`` contains ``(0, 1, 2)`` and ``(3, 4, 5)`` if ``study``
contains ``s1`` and ``s2``, ``voxel`` contains ``(0, 1, 2)``, ``(3, 4, 5)``,
and ``(6, 7, 8)``, and ``focus_reported`` contains ``("s1", 0, 1, 2)`` and
``("s2", 3, 4, 5)``.

11.5 Conditional Probability (Full Pattern)
--------------------------------------------

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
  ``tfidf`` weight column.
* The ``obtain`` clause causes ``execute_squall_program`` to return
  the query result directly as a ``NamedRelationalAlgebraFrozenSet``.

.. note::

   A full probabilistic solve requires a CPLogic-compatible EDB loaded via
   ``add_uniform_probabilistic_choice_over_set``.  See
   ``examples/plot_squall_neurosynth.py`` for the complete runnable example.

11.6 Compound Quantifier and Anaphora
---------------------------------------

The following pattern uses compound quantifiers and anaphora to express a
ternary join in plain English with zero explicit variables.

.. code-block:: squall

    define as Cooccurrence
        for every Region and for every Term
        where a Selected_study activates the Region and mentions the Term.

    define as Joint_probability with inferred probability
        every Cooccurrence (?r; ?t).

* ``for every Region and for every Term`` binds both variables into the head.
* ``the Region`` and ``the Term`` are resolved anaphorically.
* ``activates the Region`` maps to ``activates(s, r)`` and ``mentions the Term``
  maps to ``mentions(s, t)``; the study variable ``s`` is introduced by
  ``a Selected_study`` and exists only in the body.
* The intermediate ``cooccurrence`` rule has a flat head, so it can be queried
  with the standard probabilistic solver.


.. _appendix-a:

Appendix A: Running SQUALL from Python
========================================

All of the SQUALL snippets shown in the previous sections can be executed
from Python using the ``NeurolangPDL`` frontend.  The general workflow is:

1. Create a ``NeurolangPDL`` engine.
2. Register EDB facts with ``add_tuple_set``.
3. Execute a SQUALL program string with ``execute_squall_program``.
4. Inspect results with ``solve_all()`` or via the direct return from
   ``obtain`` queries.

The subsections below demonstrate each pattern.


Registering facts and running a simple rule
--------------------------------------------

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set(
    ...     [("alice",), ("bob",), ("carol",)], name="person"
    ... )
    >>> nl.add_tuple_set(
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
-----------------------------------------

    >>> result = nl.execute_squall_program(
    ...     "obtain every person that plays."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice', 'carol']


Multiple rules and an obtain clause in one program
---------------------------------------------------

    >>> result = nl.execute_squall_program(
    ...     "define as Player every person that plays. "
    ...     "define as Runner every person that runs. "
    ...     "obtain every Player."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice', 'carol']


Transitive verbs and binary predicates
---------------------------------------

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> nl.add_tuple_set([("jazz",)], name="genre")
    >>> nl.add_tuple_set([("alice", "jazz")], name="sings")
    >>> nl.execute_squall_program(
    ...     "define as Performer every person that sings a Genre."
    ... )
    >>> sorted(nl.solve_all()["performer"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']


Quantifiers: every, a, no
--------------------------

**Every**

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> nl.add_tuple_set([("alice",)], name="plays")
    >>> nl.execute_squall_program("define as Active every person that plays.")
    >>> sorted(nl.solve_all()["active"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']

**A / an / some**

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set([("a",), ("b",), ("c",)], name="item")
    >>> nl.add_tuple_set([("a", 1), ("b", 2)], name="item_count")
    >>> result = nl.execute_squall_program(
    ...     "obtain every item ?i that has an item_count ?c."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['a', 'b']

**No**

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set([("a",), ("b",), ("c",)], name="item")
    >>> nl.add_tuple_set([("a", 1), ("b", 2)], name="item_count")
    >>> result = nl.execute_squall_program(
    ...     "obtain every item ?i that has no item_count ?c."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['c']


Relative clauses and negation
------------------------------

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> nl.add_tuple_set([("alice",)], name="plays")
    >>> nl.execute_squall_program(
    ...     "define as NotPlaying every person that does not plays."
    ... )
    >>> sorted(nl.solve_all()["notplaying"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['bob']


Tuple (multi-dimensional) subjects
-----------------------------------

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set(
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
    >>> nl.add_tuple_set(
    ...     [(10, 20, 30, "s1"), (11, 21, 31, "s2")], name="peak_reported"
    ... )
    >>> nl.execute_squall_program(
    ...     "define as Activation every Peak_reported (?i; ?j; ?k; _)."
    ... )
    >>> solution = nl.solve_all()
    >>> sorted(solution["activation"].as_pandas_dataframe().apply(tuple, axis=1).tolist())
    [(10, 20, 30), (11, 21, 31)]


Compound quantifiers and anaphora
----------------------------------

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set([("A",), ("B",)], name="region")
    >>> nl.add_tuple_set([("x",), ("y",)], name="term")
    >>> nl.add_tuple_set([("s1",), ("s2",), ("s3",)], name="selected_study")
    >>> nl.add_tuple_set([("s1", "A"), ("s2", "A"), ("s3", "B")], name="activates")
    >>> nl.add_tuple_set([("s1", "x"), ("s2", "y"), ("s3", "x")], name="mentions")
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
--------------------------

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set([("A",), ("B",)], name="region")
    >>> nl.add_tuple_set([("x",), ("y",)], name="term")
    >>> nl.add_tuple_set(
    ...     [("A", "x"), ("A", "y"), ("B", "x")], name="cooccurs"
    ... )
    >>> result = nl.execute_squall_program(
    ...     "define as Joint_prob with inferred probability "
    ...     "for every Region ?r and for every Term ?t "
    ...     "where ?r cooccurs ?t. "
    ...     "obtain every Joint_prob (?r; ?t; ?p) as P."
    ... )
    >>> df = result.as_pandas_dataframe()
    >>> df.columns = ["r", "t", "p"]
    >>> sorted(df.itertuples(index=False, name=None))
    [('A', 'x', 1.0), ('A', 'y', 1.0), ('B', 'x', 1.0)]


Filtering with comparisons
---------------------------

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set(
    ...     [("a",), ("b",), ("c",), ("d",)], name="item"
    ... )
    >>> nl.add_tuple_set(
    ...     [("a", 0), ("a", 1), ("b", 2), ("c", 3)], name="item_count"
    ... )
    >>> nl.execute_squall_program(
    ...     "define as Large every Item "
    ...     "that has an item_count greater equal than 2."
    ... )
    >>> sorted(nl.solve_all()["large"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['b', 'c']


Aggregations
-------------

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set(
    ...     [("a",), ("b",), ("c",), ("d",)], name="item"
    ... )
    >>> nl.add_tuple_set(
    ...     [("a", 0), ("a", 1), ("b", 2), ("c", 3)], name="item_count"
    ... )
    >>> nl.add_tuple_set([(i,) for i in range(5)], name="quantity")
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
--------------------

**Conjunction (and)**

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set([("alice",), ("bob",), ("carol",)], name="person")
    >>> nl.add_tuple_set([("alice",), ("carol",)], name="plays")
    >>> nl.add_tuple_set([("alice",), ("bob",)], name="runs")
    >>> nl.execute_squall_program(
    ...     "define as Player every person that plays. "
    ...     "define as PlayAndRun every Player that runs."
    ... )
    >>> sorted(nl.solve_all()["playandrun"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice']

**Disjunction (or)**

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set([("alice",), ("bob",), ("carol",)], name="person")
    >>> nl.add_tuple_set([("alice",)], name="plays")
    >>> nl.add_tuple_set([("bob",)], name="runs")
    >>> nl.execute_squall_program(
    ...     "define as PlayOrRun every person that plays or runs."
    ... )
    >>> sorted(nl.solve_all()["playorrun"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice', 'bob']


Neuroimaging domain examples
-----------------------------

**Activated voxels**

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set(
    ...     [("s1",), ("s2",), ("s3",)], name="study"
    ... )
    >>> nl.add_tuple_set(
    ...     [("v1",), ("v2",), ("v3",)], name="voxel"
    ... )
    >>> nl.add_tuple_set(
    ...     [("s1", "v1"), ("s2", "v1"), ("s2", "v2")], name="reports"
    ... )
    >>> nl.execute_squall_program(
    ...     "define as Activated every Voxel ?v that a Study ?s reports."
    ... )
    >>> sorted(nl.solve_all()["activated"].as_pandas_dataframe().iloc[:, 0].tolist())
    ['v1', 'v2']

**Multi-variable activation (tuple subject)**

    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set(
    ...     [("s1",), ("s2",)], name="study"
    ... )
    >>> nl.add_tuple_set(
    ...     [(0, 1, 2), (3, 4, 5), (6, 7, 8)], name="voxel"
    ... )
    >>> nl.add_tuple_set(
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


Appendix B: IR Builder Cheat-Sheet
=====================================

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


Appendix C: Gap Report
========================

The following Datalog / IR patterns appear in codebase examples with their
current status:

.. list-table:: SQUALL gap report (updated 2026-05-21)
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
   * - ``obtain … as Name``
     - ✅ Fixed
     - ``query_as`` transformer; result named by user
   * - Compound quantifiers (``for every X and for every Y where …``)
     - ✅ Fixed
     - Added ``rule_body2``, ``quant_list``, ``quant_clause`` grammar; ``rule_opnn_compound`` transformer
   * - Anaphoric definite references (``the Noun`` → bound variable)
     - ✅ Fixed
     - ``_symbol_scope`` tracks noun-to-variable mapping per rule; ``det_the`` resolves from scope
   * - Probabilistic n-ary predicates (``with inferred probability`` on n-ary heads)
     - ✅ Fixed
     - ``rule_opnn_prob``, ``rule_opnn_marg``, ``rule_opnn_per_compound`` handlers; no engine changes needed
   * - String / numeric literals in body predicates (``startswith('L ')``)
     - ✅ Fixed
     - ``rel_fun_call`` grammar accepts ``literal`` arguments; parsed as ``Constant`` values
   * - ``#set_backend`` directive in SQUALL programs
     - ✅ Fixed
     - ``command()`` transformer builds ``FunctionApplication``; ``execute_squall_program`` calls ``config.set_query_backend()``
   * - ``given`` keyword as MARG conditioner (``… given every X …``)
     - ✅ Fixed
     - ``rule_op_marg`` and ``rule_body1_cond`` accept ``given`` as synonym for ``conditioned to``
   * - Rules + queries mixed in a single ``execute_squall_program`` call
     - ✅ Fixed
     - Probabilistic rules walked once in a shared scope; ``ForbiddenDisjunctionError`` from re-walk caught silently
    * - Arithmetic expressions in rule bodies (``?x is (a / b) - c``)
      - ✅ Fixed
      - ``s_label_is_expr`` grammar, ``rule_op_predicate_body`` transformer; arithmetic operators ``+``, ``-``, ``*``, ``/`` with standard precedence; parentheses supported
    * - Bare predicate calls in rule body (``Predicate (?a, ?b, ?c)``)
      - ✅ Fixed
      - ``s_predicate_call`` / ``s_predicate_call_upper`` grammar rules; ``rel_pred_body_call`` / ``rel_pred_body_call_upper`` transformers; arguments use comma separator
    * - Skolem-like functional terms in rule head
      - ❌ Not supported
      - Requires IR changes beyond transformer scope
