Datalog: A Rule-Based Query Language for NeuroLang
====================================================

In addition to the SQUALL controlled-English frontend, NeuroLang accepts a
standard text Datalog syntax parsed by the ``standard_syntax.py`` module.
This syntax is closer to the logical core of the system — it uses rule-based
notation with ``:-`` (implication) instead of English sentences.

This tutorial mirrors the Bayes Factor decoding example from the
:doc:`tutorial_squall` tutorial, showing the same problem expressed in text
Datalog.  Read the two side by side to see how SQUALL sentences map to
Datalog rules.

The full program we will build:

.. code-block:: none

    active_region(r) :- region(r), selected_study(s), activates(s, r).

    region_probability(r, p_r) :- PROB[active_region(r)] = p_r.

    mentioned_term(t) :- term(t), selected_study(s), mentions(s, t).

    term_probability(t, p_t) :- PROB[mentioned_term(t)] = p_t.

    cooccurrence(r, t) :-
        region(r), term(t), selected_study(s),
        activates(s, r), mentions(s, t).

    joint_probability(r, t, p_rt) :- PROB[cooccurrence(r, t)] = p_rt.

    bayes_factor(r, t, bf) :-
        joint_probability(r, t, p_rt),
        region_probability(r, p_r),
        term_probability(t, p_t),
        (bf == (p_rt / p_r) / ((p_t - p_rt) / (1.0 - p_r))).

    ans(r, t, bf) :- bayes_factor(r, t, bf), (r == 'right fusiform gyrus').

.. contents:: Contents
   :local:
   :depth: 2


Part 1: The Bayes Factor Problem
==================================

1.1 What We Are Computing
--------------------------

Given a set of brain-imaging studies, each reporting activation peaks in some
anatomical region and associated with cognitive terms, we want to find which
terms are specifically associated with the **right fusiform gyrus**.

The Bayes Factor quantifies the evidence:

.. math::

   \mathrm{BF}(r, t) =
   \frac{P(T{=}t \mid R{=}r)}{P(T{=}t \mid R{\neq}r)}
   =
   \frac{P(R,T)/P(R)}{(P(T) - P(R,T))/(1 - P(R))}

where :math:`P(R,T)` is the joint probability that a randomly chosen study
activates region :math:`R` and mentions term :math:`T`,
:math:`P(R)` is the marginal probability of activating :math:`R`, and
:math:`P(T)` of mentioning :math:`T`.

We will compute all three probability distributions as Datalog rules, then
combine them with arithmetic to produce :math:`\mathrm{BF}` — all inside
NeuroLang with zero post-hoc pandas computation.

1.2 The Data
-------------

The Datalog program works with six relations:

``study(study_id)``
    The set of all study identifiers.
``activates(study_id, region)``
    Activation foci from Neurosynth peaks, mapped to anatomical regions via
    the Julich-Brain atlas.
``mentions(study_id, term)``
    TF-IDF term associations from Neurosynth.
``region(region)``
    The set of all anatomical region names.
``term(term)``
    The set of all cognitive term names.
``selected_study(study_id)``
    A probabilistic uniform choice over all studies (registered from Python,
    see Part 5).

These are registered in Python before running Datalog:

.. code-block:: python

    from neurolang.frontend import NeurolangPDL

    nl = NeurolangPDL()
    nl.add_tuple_set(study_ids_df, name="study")
    nl.add_tuple_set(study_activates_df, name="activates")
    nl.add_tuple_set(study_mentions_df, name="mentions")
    nl.add_tuple_set(region_df, name="region")
    nl.add_tuple_set(term_df, name="term")
    # selected_study is registered separately (see Part 5)


Part 2: Datalog Preliminaries
===============================

2.1 Predicates as Relations
----------------------------

A **predicate** in Datalog names a relation — it is the equivalent of a table
name in SQL or a noun in SQUALL.  A predicate is written as an identifier
followed by parenthesised arguments:

.. code-block:: none

    region(x)
    study(s)
    activates(s, r)

``region(x)`` refers to the ``region`` EDB (extensional database) relation.
Arguments are **variables** — they bind to columns of the relation.

2.2 Rules and the ``:-`` Operator
-----------------------------------

A Datalog **rule** has the form:

.. code-block:: none

    head(x, y, ...) :- body1(x, z), body2(z, y), ...

which reads **"head holds if body1, body2, ... all hold"**.  The ``:-``
operator is logical implication — it is Datalog's only means of defining
new predicates.

Variables that appear in the head must also appear in the body (the
*range-restriction* property).  Variables used only in the body are
existentially quantified.

2.3 The Query Predicate — ``ans(...)``
----------------------------------------

The special predicate ``ans(...)`` marks a **query** in a Datalog program.
When ``execute_datalog_program`` encounters an ``ans`` rule, it runs the
query immediately and returns the result as a ``RelationalAlgebraFrozenSet``:

.. code-block:: python

    prog = """
        ans(x) :- region(x).
    """
    result = nl.execute_datalog_program(prog)
    # result is a RelationalAlgebraFrozenSet with all region names

If the program contains multiple rules without an ``ans`` rule, you can
retrieve results later with ``nl.solve_all()``.

2.4 Variables
--------------

Variables are written as identifiers (letters, digits, underscores).  Unlike
Prolog, NeuroLang's text Datalog does **not** distinguish variables from
constants by case — the same identifier can be used in predicate position
(functor) or argument position (variable) and context determines its role:

.. code-block:: none

    region(r)        % r is a variable in argument position
    term(t)          % t is a variable in argument position
    Region(R)        % upper-case identifiers work the same way
    bigger(x, y) :- size(x, a), size(y, b), (a > b).

Note that in ``bigger``, ``x`` and ``y`` are bound by the ``size`` atoms:
a comparison alone cannot bind a head variable (range restriction, §2.2).

For clarity, convention uses short lowercase names for variables that
correspond to columns (``r`` for region, ``t`` for term, ``s`` for study).

To ignore a column, use a variable name that appears only once (the engine
creates a fresh anonymous variable for each unique name):

.. code-block:: none

    ans(r) :- activates(_study, r).

Here ``_study`` appears only once and is treated as a wildcard.

2.5 Constants and Strings
--------------------------

Constants are written inline:

.. code-block:: none

    ans(x) :- region(x), (x == 'right fusiform gyrus').

String literals use single quotes ``'...'``.  Numeric constants are written
directly: ``42``, ``3.14``.  The ``==`` operator tests equality and works for
both strings and numbers (see Part 8 for more comparisons).  Comparisons in a
rule body must be enclosed in parentheses.


Part 3: Predicates and Simple Rules
=====================================

3.1 The Bayes Factor Relations
-------------------------------

The Bayes Factor problem needs two binary EDB relations, three unary EDB
relations and one probabilistic choice:

.. code-block:: none

    study(s)                % s is a study identifier
    region(r)               % r is an anatomical region name
    term(t)                 % t is a cognitive term name
    activates(s, r)         % study s activates region r
    mentions(s, t)          % study s mentions term t
    selected_study(s)       % s is a probabilistically chosen study

``selected_study`` is a **probabilistic predicate** — it is not a regular
EDB set but a uniform probabilistic choice over studies (registered from
Python, see Part 5).

3.2 Simple Unary Rules
-----------------------

The first step in the Bayes Factor program defines intermediate unary
predicates that flatten the existential study quantifier:

.. code-block:: none

    active_region(r) :- region(r), selected_study(s), activates(s, r).

This reads: *r is an active region if there exists a study s such that s is
selected and s activates r*.  The variable ``s`` appears only in the body —
it is existentially quantified away.

In Datalog notation:

.. math::

    \mathtt{active\_region}(r) \leftarrow
        \mathtt{region}(r) \land
        \mathtt{selected\_study}(s) \land
        \mathtt{activates}(s, r)

Similarly for terms:

.. code-block:: none

    mentioned_term(t) :- term(t), selected_study(s), mentions(s, t).


Part 4: Multi-Variable Rules
==============================

4.1 Compound Rules with Multiple Variables
-------------------------------------------

To compute the joint distribution we need a **binary** relation —
``cooccurrence(region, term)`` — that links each study's region and term:

.. code-block:: none

    cooccurrence(r, t) :-
        region(r), term(t), selected_study(s),
        activates(s, r), mentions(s, t).

This creates:

.. math::

    \mathtt{cooccurrence}(r, t) \leftarrow
        \mathtt{region}(r) \land \mathtt{term}(t) \land
        \mathtt{selected\_study}(s) \land
        \mathtt{activates}(s, r) \land \mathtt{mentions}(s, t).

The head variables ``r`` and ``t`` are bound by the first two body
predicates.  ``s`` appears only in the body and is existentially quantified.

4.2 Conjunction in the Body
-----------------------------

Multiple conditions in a rule body are joined with commas ``,`` (or the
``&`` operator, which is a synonym):

.. code-block:: none

    cooccurrence(r, t) :-
        region(r) & term(t) & selected_study(s) &
        activates(s, r) & mentions(s, t).

Both forms are equivalent.  The comma form is more common in Datalog
literature.

In the Bayes Factor program, the three intermediate deterministic rules
are passed as a single program string:

.. code-block:: none

    active_region(r) :- region(r), selected_study(s), activates(s, r).
    mentioned_term(t) :- term(t), selected_study(s), mentions(s, t).
    cooccurrence(r, t) :-
        region(r), term(t), selected_study(s),
        activates(s, r), mentions(s, t).


Part 5: Probabilistic Queries
===============================

Now we introduce probability — the core of the Bayes Factor computation.

5.1 The Probabilistic Choice
-----------------------------

Before probabilistic queries work, the engine needs to know which predicate
represents a probabilistic choice over studies.  This is registered from
Python:

.. code-block:: python

    nl.add_uniform_probabilistic_choice_over_set(
        study_ids_df, name="selected_study"
    )

This registers ``selected_study`` as a predicate where each study has equal
probability :math:`1/N`.  Once registered, ``selected_study(s)`` can appear
in rule bodies, and the PROB/MARG constructs can compute probabilities over
it.

5.2 PROB — Marginal Probability
---------------------------------

The ``PROB[pred(x, ...)] = p`` body predicate computes the marginal
probability of a predicate after marginalising over all unbound variables:

.. code-block:: none

    region_probability(r, p_r) :- PROB[active_region(r)] = p_r.

This binds ``p_r`` to :math:`P(\mathtt{active\_region}(r))` —
the fraction of selected studies that activate region ``r``.

The equivalent in the Bayes Factor program:

.. code-block:: none

    region_probability(r, p_r) :- PROB[active_region(r)] = p_r.
    term_probability(t, p_t) :- PROB[mentioned_term(t)] = p_t.
    joint_probability(r, t, p_rt) :- PROB[cooccurrence(r, t)] = p_rt.

A conditioning filter can be added with ``//`` (``|`` is accepted as a
synonym):

.. code-block:: none

    ans(r, p) :- PROB[active_region(r) // mentioned_term('face')] = p.

This computes the conditional probability
:math:`P(\mathtt{active\_region}(r) \mid \mathtt{mentioned\_term}(\mathit{face}))`,
i.e. the fraction of the studies mentioning *face* that activate ``r``.

5.3 MARG — Marginal Probability of a Conjunction
--------------------------------------------------

``MARG[...] = p`` computes the same probability as ``PROB``; it is mostly
used with a conjunction inside the brackets:

.. code-block:: none

    ans(r, p) :- MARG[selected_study(s), activates(s, r)] = p.

This computes :math:`P(\exists s.\ \mathtt{selected\_study}(s) \land
\mathtt{activates}(s, r))` for each region ``r``.  As with ``PROB``, adding
``//`` gives a conditional probability:

.. code-block:: none

    ans(r, p) :- MARG[active_region(r) // mentioned_term('face')] = p.

5.4 SUCC — Success Probability
-------------------------------

The parser accepts ``SUCC[...] = p`` (and ``SUCC[...] :- body``) for
compatibility, but these atoms are currently discarded before execution.
Use ``PROB`` instead.


Part 6: Connectives
=====================

6.1 Conjunction
----------------

Multiple body atoms are joined with ``,`` or ``&``:

.. code-block:: none

    p(x, y) :- a(x), b(x, y).
    p(x, y) :- a(x) & b(x, y).

Both are equivalent.  The comma is the standard Datalog convention.

6.2 Disjunction
----------------

Datalog expresses disjunction through multiple rules with the same head:

.. code-block:: none

    p(x) :- a(x).
    p(x) :- b(x).

``p`` holds if either ``a`` holds or ``b`` holds (or both).

6.3 Negation
-------------

Negation-as-failure uses the ``~`` prefix (the grammar also lists the
Unicode ``¬``, but ``~`` is the recommended form):

.. code-block:: none

    not_playing(x) :- person(x), ~plays(x).

This reads: *x is not playing if x is a person and there is no proof that
x plays* — standard negation-as-failure semantics.


Part 7: The Bayes Factor Formula
==================================

7.1 Arithmetic Expressions
---------------------------

Datalog rules support arithmetic expressions with ``+``, ``-``, ``*``,
``/``, and ``**`` (exponentiation), with standard operator precedence:

.. code-block:: none

    (bf == (p_rt / p_r) / ((p_t - p_rt) / (1.0 - p_r)))

Parentheses control grouping, and the whole comparison is itself enclosed in
parentheses.  The ``==`` operator binds the result of the expression to the
variable on the left.

7.2 The Complete Bayes Factor Rule
------------------------------------

The BF formula combines the three probability distributions with arithmetic:

.. code-block:: none

    bayes_factor(r, t, bf) :-
        joint_probability(r, t, p_rt),
        region_probability(r, p_r),
        term_probability(t, p_t),
        (bf == (p_rt / p_r) / ((p_t - p_rt) / (1.0 - p_r))).

Each ``probability`` atom provides a probability column (``p_rt``, ``p_r``,
``p_t``) that was created by the PROB construct in Part 5.  The last line
evaluates the Bayes Factor formula inline using arithmetic.


Part 8: Querying and Optimization
===================================

8.1 The ``ans`` Query Predicate
---------------------------------

The ``ans`` predicate marks a query.  When the Datalog program is executed
via ``nl.execute_datalog_program()``, any ``ans`` rule is evaluated
immediately and its result returned:

.. code-block:: none

    ans(r, t, bf) :-
        bayes_factor(r, t, bf),
        (r == 'right fusiform gyrus').

.. code-block:: python

    result = nl.execute_datalog_program(prog)
    df = result.as_pandas_dataframe()
    df.columns = ["region", "term", "bf"]
    df.sort_values("bf", ascending=False).head()

8.2 Filtering with Comparisons
--------------------------------

The ``(r == 'right fusiform gyrus')`` filter uses the equality comparison
operator.  Text Datalog supports the full comparison suite (always written
between parentheses in a rule body):

========= =================
Operator  Meaning
========= =================
``==``    equal
``!=``    not equal
``<``     less than
``<=``    less or equal
``>``     greater than
``>=``    greater or equal
========= =================

Comparisons work on both numeric and string arguments.

8.3 Magic-Sets Optimization
-----------------------------

The ``(r == 'right fusiform gyrus')`` filter does more than just post-process
— NeuroLang's **magic-sets** optimisation pushes the constant backwards
through all the rules in the chain:

1. The constant ``'right fusiform gyrus'`` is inlined into the
   ``bayes_factor`` predicate before the SIPS (Sideways Information Passing
   Strategy) pass.
2. The SIPS creates an adorned predicate with the first argument marked as
   *bound*.
3. Magic rules propagate the bound value down the dependency chain:
   ``magic_joint_probability(r)``, ``magic_region_probability(r)``,
   ``magic_cooccurrence(r)``, ``magic_active_region(r)``.
4. Each adorned rule filters its body with a magic predicate, so only rows
   where ``r = 'right fusiform gyrus'`` are evaluated.

The result: with 12 000+ studies and 1 million term-study pairs, the query
completes in approximately 30 seconds because only 4 298 studies activating
the right fusiform gyrus are evaluated.

.. note::

    Magic-sets works when the constant appears on an IDB (rule-defined)
    predicate, as it does here (``bayes_factor`` is defined by a rule).
    A constant on an EDB predicate does not restrict that predicate, but
    its rows bind its variables in the IDB predicates right of it: in
    ``ans(x) :- e("c", y) & p(y, x)``, with ``e`` an EDB predicate and
    ``p`` defined by rules, the values of ``y`` in the rows of ``e``
    with ``"c"`` are passed to ``p``.


Part 9: The Complete Program
==============================

Putting it all together:

.. code-block:: none

    % Deterministic intermediate rules
    active_region(r) :- region(r), selected_study(s), activates(s, r).
    mentioned_term(t) :- term(t), selected_study(s), mentions(s, t).

    cooccurrence(r, t) :-
        region(r), term(t), selected_study(s),
        activates(s, r), mentions(s, t).

    % Probabilistic rules — marginal probabilities
    region_probability(r, p_r) :- PROB[active_region(r)] = p_r.
    term_probability(t, p_t) :- PROB[mentioned_term(t)] = p_t.
    joint_probability(r, t, p_rt) :- PROB[cooccurrence(r, t)] = p_rt.

    % Bayes Factor formula
    bayes_factor(r, t, bf) :-
        joint_probability(r, t, p_rt),
        region_probability(r, p_r),
        term_probability(t, p_t),
        (bf == (p_rt / p_r) / ((p_t - p_rt) / (1.0 - p_r))).

    % Query — filter for the target region
    ans(r, t, bf) :- bayes_factor(r, t, bf), (r == 'right fusiform gyrus').

Running this on real Neurosynth data produces the same results as the
SQUALL program in :doc:`tutorial_squall` (Part 9); the top 10 terms are
shown here:

.. list-table:: Top terms by Bayes Factor for the right fusiform gyrus
   :header-rows: 1

   * - Term
     - Bayes Factor
   * - ffa
     - 11.69
   * - face ffa
     - 11.06
   * - fusiform face
     - 11.02
   * - fusiform gyri
     - 7.04
   * - fusiform
     - 6.43
   * - fusiform gyrus
     - 6.39
   * - word form
     - 6.20
   * - visual word
     - 5.99
   * - occipito temporal
     - 5.87
   * - visual stream
     - 5.22

All terms exceed the Jeffreys "substantial evidence" threshold of
:math:`\sqrt{10} \approx 3.16`.  The top cluster — FFA, face, fusiform
face — is exactly what the right fusiform gyrus is known for.


Part 10: Advanced Features
============================

This section documents features not used directly in the Bayes Factor
example but available in the text Datalog syntax.

10.1 Aggregation
-----------------

Aggregations summarise a set of values into a single result per group using
the ``AGGREGATE`` construct:

.. code-block:: none

    ans(x, cnt) :- AGGREGATE[x](body(x, y) @ count(y)) = cnt.

The group-by variables appear in square brackets.  The ``@`` sign separates
the body from the aggregation function call.

Supported functions: ``count``, ``sum``, ``max``, ``min``, ``mean``,
``avg``, ``std``.

Example — count studies per region:

.. code-block:: none

    study_count(r, cnt) :-
        AGGREGATE[r](activates(s, r) @ count(s)) = cnt.

Global aggregation (no group-by) uses empty brackets:

.. code-block:: none

    total_studies(cnt) :-
        AGGREGATE[](study(s) @ count(s)) = cnt.

An aggregation can also be written directly in the head of a rule, the
variables of the head being the group-by variables.  This form accepts
aggregation functions with constant parameters, such as
``percentile(v, q)``: the ``q``-th percentile of ``v``, with ``q`` in
``[0, 100]``, ignoring NaN values.  It keeps, for instance, the top 10% of
the scores of each query:

.. code-block:: none

    cutoff(q, percentile(v, 90)) :- score(q, s, v).
    top_score(q, s) :- score(q, s, v), cutoff(q, t), (v >= t).

Percentiles are linearly interpolated and ties are kept, so the top 10%
can hold slightly more than 10% of the rows.

10.2 Existential Predicates
----------------------------

The ``exists`` construct asserts that there exists a tuple satisfying a set
of conditions, without binding the existential variable in the rule head:

.. code-block:: none

    has_activations(r) :- region(r), exists(s; activates(s, r)).

This is equivalent to:

.. code-block:: none

    has_activations(r) :- region(r), activates(s, r).

The ``;`` separates the existential variable from the body predicates.
Multiple existential variables can be stacked:

.. code-block:: none

    ans(r) :- exists(s, t; activates(s, r), mentions(s, t)).

10.3 Comparisons
-----------------

Besides equality (``==``), the full comparison suite is available:

.. code-block:: none

    large_items(x) :- item(x), item_count(x, c), (c >= 2).
    not_empty(x) :- item_count(x, c), (c != 0).
    small(x) :- item(x), item_count(x, c), (c < 10).

Comparisons can be combined with conjunction in the same rule body.

10.4 Commands and Directives
-----------------------------

Commands use the ``.`` prefix and follow the Prolog convention.  The text
Datalog frontend currently executes two commands, which load external data
into a relation:

.. code-block:: none

    .load_csv(study_metadata, "https://example.com/metadata.tsv", sep="\t")
    .load_atlas(destrieux, "atlas.nii.gz", "labels.csv", "https://example.com/destrieux.tgz")

Any other command (for instance ``.set_backend('polars')``) is rejected with
an ``UnsupportedProgramError``.  To change the relational algebra backend
(``'pandas'`` or ``'polars'``) from Datalog code, use the SQUALL
``#set_backend`` directive or, from Python,
``neurolang.config.config.set_query_backend(...)``; see :doc:`performance`.
Note that the backend is a process-wide setting.

10.5 Probabilistic Facts
-------------------------

Explicit probabilistic facts can be written inline:

.. code-block:: none

    0.9 :: noisy_region('fusiform').

This creates a probabilistic fact with probability 0.9.

Probabilistic rules (with a probability expression instead of a fixed
probability) also exist:

.. code-block:: none

    p(x) :: exp(-d / 5.0) :- a(x, d), (d < 0.8).


10.6 Probabilistic Choices
---------------------------

Probabilistic choices represent mutually exclusive alternatives — exactly one
alternative from each choice group holds in any possible world.  This is
different from probabilistic facts (``::``), where each fact is an
independent Bernoulli trial.

The choice operators are ``^`` and ``:~:`` (both are equivalent):

**Choice fact** — a standalone choice with a fixed probability:

.. code-block:: none

    0.3 ^ choice('a').
    0.7 ^ choice('b').

This declares two mutually exclusive alternatives: ``choice('a')`` holds with
probability 0.3 and ``choice('b')`` holds with probability 0.7.  The
probabilities in a choice group must sum to 1.

**Choice rule** — a choice whose head depends on a body:

.. code-block:: none

    n_studies(n) :- AGGREGATE[](study(s) @ count(s)) = n.
    picked_study(s) ^ (1 / n) :- study(s), n_studies(n).

Here the probability expression ``(1 / n)`` is evaluated at runtime, with
``n`` bound in the body to the number of studies; each tuple in the body
produces one arm of the choice, so ``picked_study`` is a uniform choice over
studies.  The ``:~:`` operator can be used interchangeably:

.. code-block:: none

    picked_study(s) :~: (1 / n) :- study(s), n_studies(n).


.. _appendix-a-datalog:

Appendix A: Running Datalog from Python
=========================================

All of the Datalog syntax shown in this tutorial can be executed from Python
using the ``NeurolangPDL`` frontend.  The general workflow follows the same
pattern as SQUALL (see :ref:`part-9-appendix-a`), but using
``execute_datalog_program`` instead of ``execute_squall_program``.

The steps are:

1. Create a ``NeurolangPDL`` engine.
2. Register EDB facts with ``add_tuple_set``.
3. Optionally register probabilistic choices with
   ``add_uniform_probabilistic_choice_over_set``.
4. Execute a Datalog program string with ``execute_datalog_program``.
5. Inspect results from the direct return value (when ``ans`` is used) or
   with ``solve_all()``.

Registering facts and running a simple rule
--------------------------------------------

.. code-block:: python

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [("alice",), ("bob",), ("carol",)], name="person"
    ... )
    >>> _ = nl.add_tuple_set(
    ...     [("alice",), ("carol",)], name="plays"
    ... )
    >>> nl.execute_datalog_program(
    ...     "active(x) :- person(x), plays(x)."
    ... )
    >>> solution = nl.solve_all()
    >>> sorted(
    ...     solution["active"].as_pandas_dataframe().iloc[:, 0].tolist()
    ... )
    ['alice', 'carol']

Querying with ``ans`` (direct return)
--------------------------------------

When the program contains an ``ans`` rule, ``execute_datalog_program``
returns the result directly:

.. code-block:: python

    >>> result = nl.execute_datalog_program(
    ...     "ans(x) :- person(x), plays(x)."
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice', 'carol']

Multiple rules
---------------

Each rule goes on its own line:

.. code-block:: python

    >>> result = nl.execute_datalog_program(
    ...     """
    ...     player(x) :- person(x), plays(x).
    ...     non_player(x) :- person(x), ~plays(x).
    ...     ans(x) :- player(x).
    ...     """
    ... )
    >>> sorted(result.as_pandas_dataframe().iloc[:, 0].tolist())
    ['alice', 'carol']

Binary predicates
------------------

.. code-block:: python

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> _ = nl.add_tuple_set([("jazz",)], name="genre")
    >>> _ = nl.add_tuple_set([("alice", "jazz")], name="sings")
    >>> nl.execute_datalog_program(
    ...     "performer(x) :- person(x), sings(x, g)."
    ... )
    >>> sorted(
    ...     nl.solve_all()["performer"]
    ...     .as_pandas_dataframe().iloc[:, 0].tolist()
    ... )
    ['alice']

Negation
---------

.. code-block:: python

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("alice",), ("bob",)], name="person")
    >>> _ = nl.add_tuple_set([("alice",)], name="plays")
    >>> nl.execute_datalog_program(
    ...     "not_playing(x) :- person(x), ~plays(x)."
    ... )
    >>> sorted(
    ...     nl.solve_all()["not_playing"]
    ...     .as_pandas_dataframe().iloc[:, 0].tolist()
    ... )
    ['bob']

Filtering with comparisons
---------------------------

.. code-block:: python

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [("a",), ("b",), ("c",), ("d",)], name="item"
    ... )
    >>> _ = nl.add_tuple_set(
    ...     [("a", 0), ("a", 1), ("b", 2), ("c", 3)], name="item_count"
    ... )
    >>> nl.execute_datalog_program(
    ...     "large(x) :- item(x), item_count(x, c), (c >= 2)."
    ... )
    >>> sorted(
    ...     nl.solve_all()["large"]
    ...     .as_pandas_dataframe().iloc[:, 0].tolist()
    ... )
    ['b', 'c']

Aggregation
------------

.. code-block:: python

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set(
    ...     [("a",), ("b",), ("c",), ("d",)], name="item"
    ... )
    >>> _ = nl.add_tuple_set(
    ...     [("a", 0), ("a", 1), ("b", 2), ("c", 3)], name="item_count"
    ... )
    >>> result = nl.execute_datalog_program(
    ...     """
    ...     max_count(x, m) :- AGGREGATE[x](item_count(x, c) @ max(c)) = m.
    ...     ans(x, m) :- max_count(x, m).
    ...     """
    ... )
    >>> sorted(
    ...     result.as_pandas_dataframe().apply(tuple, axis=1).tolist()
    ... )
    [('a', 1), ('b', 2), ('c', 3)]

Probabilistic query (PROB)
---------------------------

.. code-block:: python

    >>> nl = NeurolangPDL()
    >>> _ = nl.add_tuple_set([("s1",), ("s2",), ("s3",)], name="study")
    >>> _ = nl.add_tuple_set(
    ...     [("s1", "A"), ("s2", "A"), ("s3", "B")], name="activates"
    ... )
    >>> _ = nl.add_uniform_probabilistic_choice_over_set(
    ...     [("s1",), ("s2",), ("s3",)], name="selected_study"
    ... )
    >>> result = nl.execute_datalog_program(
    ...     "ans(r, p) :- PROB[selected_study(s), activates(s, r)] = p."
    ... )
    >>> df = result.as_pandas_dataframe()
    >>> sorted(df.itertuples(index=False, name=None))
    [('A', 0.666...), ('B', 0.333...)]

.. note::

    Conjunctions inside ``PROB[...]`` use ``,`` as separator (same as Datalog
    rule bodies).  The more common pattern in the Bayes Factor program is to
    define an intermediate rule first and then query ``PROB[pred(x)] = p``
    on the single predicate.


.. _appendix-b-datalog:

Appendix B: Querying from the Command Line
============================================

The ``neurolang-query`` command-line tool lets you run Datalog (or SQUALL)
queries without writing Python code.  It picks up a preconfigured engine from
the :ref:`engine registry <neurolang-engine-registry>` and executes the query
directly.

Listing available engines and predicates::

    neurolang-query --list-engines
    neurolang-query --engine neurosynth --list-predicates

Running a Datalog query inline::

    neurolang-query --engine neurosynth \
        "ans(t) :- term_in_study_tfidf(s, t, w)"

Running a query from a file::

    neurolang-query --engine neurosynth -f query.dl

Sorting the output by a column of the answer (``--sort`` can be repeated;
append ``:desc`` for descending order)::

    neurolang-query --engine neurosynth \
        --sort "w:desc" \
        "ans(t, w) :- term_in_study_tfidf(s, t, w)"

Saving and viewing the results
------------------------------

``--save PATH`` also writes the result in a file, and ``--view`` shows
its brain image in the viewer of the interactive mode, as the REPL's
``.save`` and ``.view`` do for its last result (see ``.help save`` and
``.help view`` there for the details).  They need the ``tui`` extra
(``pip install "neurolang[tui]"``).

* ``--save out.csv`` or ``out.tsv`` (optionally ``.gz``) writes the table
  of the result, sorted as ``--sort`` asks.
* ``--save out.nii`` or ``out.nii.gz`` writes a NIfTI image on the grid of
  the engine's template.  The voxels are the columns of the result of a
  sort the engine declares as voxel indices, or as millimetres, such as
  ``i, j, k``; the value of a voxel is its probability or its one other
  numeric column, such as a count, else 1.  A SQUALL program of several
  ``obtain`` statements gives a file per statement.
* ``--view`` writes the viewer's HTML page, prints its path and opens it
  in a browser when the standard error is a terminal.

For instance, the map of the number of studies mentioning pain that
report each voxel::

    neurolang-query --engine neurosynth --save pain.nii.gz --view \
        "n_pain(i, j, k, count(s)) :- reports(s, i, j, k), mentions(s, 'pain').
         ans(i, j, k, n) :- n_pain(i, j, k, n)"

Both work with ``--ask``, which then runs the query it writes.

Inspecting query plans
----------------------

``--show-ra`` (``-Q``) prints, instead of the answer, the relational algebra
plan each rule of the query is evaluated with, stratum by stratum
(deterministic, probabilistic and post-probabilistic).  Plans are drawn as
trees; base relations are shown as atoms with their number of rows, the
predicate generated for the query is shown as ``ans``, probabilistic plans
name the solver that produced them (small dichotomy theorem, Dalvi–Suciu
lifted inference or weighted model counting) and label their probability
columns ``p``, and recursive rules are shown once, followed by the number of
iterations the fixpoint took.  The rewritten program is printed too when
``--show-rewritten`` is given.  Two flags refine the display:

* ``--ra-verbose`` prints the plans in full, without collapsing the renames
  and projections of base relations into atoms (``ρ[x, y](π[#0, #1](edge))``
  instead of ``edge(x, y)``);
* ``--ra-analyze`` runs the query and annotates every plan node with the
  number of rows it produced, like SQL's ``EXPLAIN ANALYZE``.  Counts that
  would require removing duplicates of values that are expensive to compare,
  such as regions, are upper bounds, printed ``‹≤N rows›``.

For instance::

    $ neurolang-query --engine neurosynth --show-ra --ra-analyze \
        "ans(l, s) :- reports(s, i, j, k), schaefer_voxels(i, j, k, l)."
    ── deterministic stratum ──
    ── rule reports(s, i, j, k) :-
        peak_reported(i, j, k, s) ──
    peak_reported(i, j, k, s) ‹447,477 rows›
    ── rule schaefer_voxels(I, J, K, l) :-
        schaefer(l, r),
        ⟨contains⟩(r, (I, J, K)) ──
    δ[r→('I', 'J', 'K')] ‹≤132,032 rows›
    └─ schaefer(l, r) ‹100 rows›
    ── rule ans(l, s) :-
        reports(s, i, j, k),
        schaefer_voxels(i, j, k, l) ──
    ⋈ ‹10,190 rows›
    ├─ reports(s, i, j, k) ‹447,477 rows›
    └─ schaefer_voxels(i, j, k, l) ‹132,032 rows›

From Python, pass ``show_ra=True`` to ``execute_datalog_program``,
``execute_squall_program`` or ``query``, or a
:class:`~neurolang.frontend.show_ra.ShowRAOptions` such as
``ShowRAOptions(collapse=False, analyze=True)``.

For the full list of options, see ``neurolang-query --help`` or the
:py:mod:`neurolang.utils.cli` module documentation.


Appendix C: Engine Configuration (engines.yaml)
===============================================

See :doc:`engine_configuration` for the full engine registry reference.
