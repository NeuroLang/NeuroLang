Datalog: A Rule-Based Query Language for NeuroLang
====================================================

Besides the SQUALL controlled English, NeuroLang reads Datalog: rules
written with ``:-`` (implication) instead of English sentences, closer to
the logical core of the system.

Every example of this tutorial runs in the REPL (:doc:`repl`), in the
browser or in ``neurolang-query -i --engine neurosynth``: type it at the
prompt, or follow *Run in the browser REPL* under it. An example that uses
the rules of earlier ones has no such link: type the earlier ones first.

This tutorial mirrors the Bayes Factor decoding example from the
:doc:`tutorial_squall` tutorial, showing the same problem expressed in text
Datalog.  Read the two side by side to see how SQUALL sentences map to
Datalog rules.

The full program we will build, on the studies of Neurosynth and the
Destrieux atlas (it takes about 20 s):

.. nl-repl::

    activates(s, r) :- destrieux_activated(r, s)
    region(r) :- destrieux(r, x)

    active_region(r) :- region(r), selected_study(s), activates(s, r)
    region_probability(r, p_r) :- PROB[active_region(r)] = p_r

    mentioned_term(t) :- term(t), selected_study(s), mentions(s, t)
    term_probability(t, p_t) :- PROB[mentioned_term(t)] = p_t

    cooccurrence(r, t) :-
        region(r), term(t), selected_study(s),
        activates(s, r), mentions(s, t)
    joint_probability(r, t, p_rt) :- PROB[cooccurrence(r, t)] = p_rt

    bayes_factor(r, t, bf) :-
        joint_probability(r, t, p_rt),
        region_probability(r, p_r),
        term_probability(t, p_t),
        (bf == (p_rt / p_r) / ((p_t - p_rt) / (1.0 - p_r)))

    ans(r, t, bf) :- bayes_factor(r, t, bf), (r == 'R G oc temp lat fusifor')

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

The program uses four relations of the ``neurosynth`` engine (``.predicates``
lists them all, with their columns):

``study(study_id)``
    The 14 371 studies of Neurosynth, by PubMed id.
``mentions(study_id, term)``
    The terms each study mentions (TF-IDF above 0.03).
``term(term)``
    The terms of Neurosynth's vocabulary.
``selected_study(study_id)``
    A uniform probabilistic choice over the studies (see Part 5).

and two it defines from the Destrieux atlas, the anatomical regions and the
studies whose peaks fall in them:

.. nl-repl::

    activates(s, r) :- destrieux_activated(r, s)
    region(r) :- destrieux(r, x)

The REPL answers ``defined activates/2`` and ``defined region/1``: the
rules stay in the session, and the examples below use them. The region we
look at is ``'R G oc temp lat fusifor'``, the right fusiform gyrus in the labels of
the atlas, which 2 836 of the studies activate.


Part 2: Datalog Preliminaries
===============================

2.1 Predicates as Relations
----------------------------

A **predicate** in Datalog names a relation — it is the equivalent of a table
name in SQL or a noun in SQUALL.  A predicate is written as an identifier
followed by parenthesised arguments:

.. code-block:: neurolang

    region(x)
    study(s)
    activates(s, r)

``region(x)`` refers to the ``region`` EDB (extensional database) relation.
Arguments are **variables** — they bind to columns of the relation.

2.2 Rules and the ``:-`` Operator
-----------------------------------

A Datalog **rule** has the form:

.. code-block:: neurolang

    head(x, y, ...) :- body1(x, z), body2(z, y), ...

which reads **"head holds if body1, body2, ... all hold"**.  The ``:-``
operator is logical implication — it is Datalog's only means of defining
new predicates.

Variables that appear in the head must also appear in the body (the
*range-restriction* property).  Variables used only in the body are
existentially quantified.

A body may be a **disjunction**, written ``|`` (or ``∨``); ``,`` binds tighter
than ``|``, and parentheses group:

.. code-block:: neurolang

    uncinate(x, h) :-
        (traverses(x, 'orbitofrontal_gyrus', h)
         | traverses(x, 'middle_frontal_gyrus', h)
         | traverses(x, 'inferior_frontal_gyrus', h)),
        endpoint_in(x, e, 'temporal', h).

stands for three rules, one per disjunct.  Each disjunct must bind the
variables of the head, and those of its negated atoms.  ``a(x), b(x) | c(x)``
is ``(a(x), b(x)) | c(x)``.  Inside parentheses a comparison is a
formula of its own, so that values are alternatives of a variable::

    ans(i, t) :- cogat_term(i, t), (t == 'motor execution' | t == 'motor planning')

and ``b(x), (a(x) | x < 3)`` is ``b(x), (a(x) | (x < 3))``; outside
parentheses a comparison keeps its own: ``b(x), (x < 3)``.

.. note::

   ``|`` is a disjunction everywhere, also inside ``PROB[...]`` and
   ``MARG[...]``: ``PROB[a(x) | b(x)] = p`` is the probability of
   ``a(x) | b(x)``.  A conditional probability is written with ``//`` only.
   Before, ``|`` was a synonym of ``//``.

2.3 The Query Predicate — ``ans(...)``
----------------------------------------

The special predicate ``ans(...)`` marks a **query**. The REPL runs it as
soon as it is entered and shows its answer as a table:

.. nl-repl::
    :norun:

    ans(x) :- region(x)

A rule without ``ans`` only defines a relation, for the queries to come.

A submission can hold several queries, each with a name in brackets,
``ans[name](...)``; the REPL shows each answer under its name, and ``.view``
shows them all at once, as one image per query, with the networks and
streamlines of the others:

.. code-block:: neurolang

    ans[left](x, y, z) :- activation(x, y, z, 'left hemisphere')
    ans[tract](s) :- uncinate(s, 'left')
    ans[network](r1, r2, p) :- coactivation(r1, r2, p)

The queries of a submission have each a name: ``ans`` stands for the one
without brackets, and two queries cannot share a name. They are solved
together, which is faster than one after the other when they share
predicates; see :ref:`several-queries`. In Python, a program's queries are
returned by ``execute_datalog_program`` (:doc:`tutorial_python`).

2.4 Variables
--------------

Variables are written as identifiers (letters, digits, underscores).  Unlike
Prolog, NeuroLang's text Datalog does **not** distinguish variables from
constants by case — the same identifier can be used in predicate position
(functor) or argument position (variable) and context determines its role:

.. code-block:: neurolang

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

.. code-block:: neurolang

    ans(r) :- activates(_study, r).

Here ``_study`` appears only once and is treated as a wildcard.

2.5 Constants and Strings
--------------------------

Constants are written inline:

.. code-block:: neurolang

    ans(x) :- region(x), (x == 'R G oc temp lat fusifor').

String literals use single quotes ``'...'``.  Numeric constants are written
directly: ``42``, ``3.14``.  The ``==`` operator tests equality and works for
both strings and numbers (see Part 8 for more comparisons).  Comparisons in a
rule body must be enclosed in parentheses.


.. _datalog-doc-comments:

2.6 Comments and Documentation
-------------------------------

A ``%`` starts a comment, to the end of the line, unless it is in a string
or an IRI.  A block of lines starting with ``%%`` right before a rule or a
fact documents its predicate.  Its first line may give the signature of the
predicate, the names of its arguments and, after a ``:``, their sorts; the
other lines are its description:

.. code-block:: neurolang

    %% uncinate(x: streamline, h: hemisphere)
    %% The uncinate fasciculus: streamlines joining the orbitofrontal
    %% cortex to the anterior temporal lobe through the temporal stem,
    %% in hemisphere h.
    uncinate(x, h) :- ...

Without a signature, the whole block is the description.  The engine keeps
the documentation (see :doc:`tutorial_python` for it in Python): the
query assistant and the REPL's ``.predicates`` show it, and a column of the
sort ``streamline`` is drawn as streamlines by ``.view``.  The rules are:

* The block must be right before the statement: a block followed by a blank
  line, a ``%`` comment or nothing is a comment, as a header of a file.  A
  ``%%`` after the start of a line, or inside a rule, is a comment too.
* It documents the predicate of the rule, fact, probabilistic fact or
  choice, or ``name := ...`` statement after it, and the following rules of
  that predicate need none; it documents nothing before a query or a
  command.
* The signature is the first line only if it reads ``name(...)`` with the
  name of that predicate, and then it must give its number of arguments.
  Each argument is a column: the three columns of a voxel are written
  ``i: voxel, j: voxel, k: voxel``.
* The blocks of a predicate add up: their descriptions follow one another,
  each once, and the arguments of the last signature are kept.
* The documentation follows the scopes of the program, as its rules do:
  that of a rule defined in ``with nl.scope:``, or in the REPL's session
  that ``.clear`` empties, goes with it.  A module imported with
  ``.import`` documents its predicates as the program does.
* The documentation of a predicate in the program wins over its
  description in the engine's configuration.

``.document(name, "description")`` in a program documents a predicate too.


Part 3: Predicates and Simple Rules
=====================================

3.1 The Bayes Factor Relations
-------------------------------

The Bayes Factor problem needs two binary relations, three unary ones and
one probabilistic choice, of the engine or defined in 1.2:

.. code-block:: neurolang

    study(s)                % s is a study identifier
    region(r)               % r is an anatomical region name
    term(t)                 % t is a cognitive term name
    activates(s, r)         % study s activates region r
    mentions(s, t)          % study s mentions term t
    selected_study(s)       % s is a probabilistically chosen study

``selected_study`` is a **probabilistic predicate** — it is not a regular
relation but a uniform probabilistic choice over studies (see Part 5).

3.2 Simple Unary Rules
-----------------------

The first step in the Bayes Factor program defines intermediate unary
predicates that flatten the existential study quantifier:

.. code-block:: neurolang

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

.. code-block:: neurolang

    mentioned_term(t) :- term(t), selected_study(s), mentions(s, t).


Part 4: Multi-Variable Rules
==============================

4.1 Compound Rules with Multiple Variables
-------------------------------------------

To compute the joint distribution we need a **binary** relation —
``cooccurrence(region, term)`` — that links each study's region and term:

.. code-block:: neurolang

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

.. code-block:: neurolang

    cooccurrence(r, t) :-
        region(r) & term(t) & selected_study(s) &
        activates(s, r) & mentions(s, t).

Both forms are equivalent.  The comma form is more common in Datalog
literature.

In the Bayes Factor program, the three intermediate deterministic rules
are passed as a single program string:

.. code-block:: neurolang

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

Probabilistic queries need a predicate that is a probabilistic choice over
the studies. The ``neurosynth`` engine has one, ``selected_study``: each of
its :math:`N` studies is the selected one with probability :math:`1/N`, and
exactly one is selected in each possible world. ``selected_study(s)`` can
appear in rule bodies, and PROB and MARG compute probabilities over it. A
choice of your own can be written in Datalog (10.6); the engine's comes
from its configuration (:doc:`engine_configuration`).

5.2 PROB — Marginal Probability
---------------------------------

The ``PROB[pred(x, ...)] = p`` body predicate computes the marginal
probability of a predicate after marginalising over all unbound variables:

.. code-block:: neurolang

    region_probability(r, p_r) :- PROB[active_region(r)] = p_r.

This binds ``p_r`` to :math:`P(\mathtt{active\_region}(r))` —
the fraction of selected studies that activate region ``r``.

The equivalent in the Bayes Factor program:

.. code-block:: neurolang

    region_probability(r, p_r) :- PROB[active_region(r)] = p_r.
    term_probability(t, p_t) :- PROB[mentioned_term(t)] = p_t.
    joint_probability(r, t, p_rt) :- PROB[cooccurrence(r, t)] = p_rt.

A conditioning filter can be added with ``//`` (``|`` is a disjunction,
not a conditioning, see 2.2):

.. code-block:: neurolang

    ans(r, p) :- PROB[active_region(r) // mentioned_term('face')] = p.

This computes the conditional probability
:math:`P(\mathtt{active\_region}(r) \mid \mathtt{mentioned\_term}(\mathit{face}))`,
i.e. the fraction of the studies mentioning *face* that activate ``r``.
Either side of ``//`` may be a conjunction, in parentheses or not:
``PROB[(a(x, s), (x < y)) // (b(s), c(s))] = p`` is
``PROB[a(x, s), (x < y) // b(s), c(s)] = p``.

5.3 MARG — Marginal Probability of a Conjunction
--------------------------------------------------

``MARG[...] = p`` computes the same probability as ``PROB``; it is mostly
used with a conjunction inside the brackets:

.. code-block:: neurolang

    ans(r, p) :- MARG[selected_study(s), activates(s, r)] = p.

This computes :math:`P(\exists s.\ \mathtt{selected\_study}(s) \land
\mathtt{activates}(s, r))` for each region ``r``.  As with ``PROB``, adding
``//`` gives a conditional probability:

.. code-block:: neurolang

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

.. code-block:: neurolang

    p(x, y) :- a(x), b(x, y).
    p(x, y) :- a(x) & b(x, y).

Both are equivalent.  The comma is the standard Datalog convention.

6.2 Disjunction
----------------

Datalog expresses disjunction through multiple rules with the same head:

.. code-block:: neurolang

    p(x) :- a(x).
    p(x) :- b(x).

``p`` holds if either ``a`` holds or ``b`` holds (or both).

6.3 Negation
-------------

Negation-as-failure uses the ``~`` prefix (the grammar also lists the
Unicode ``¬``, but ``~`` is the recommended form):

.. code-block:: neurolang

    silent_region(r) :- region(r), ~activates(s, r).

This reads: *r is a silent region if r is a region and there is no proof
that a study activates it* — standard negation-as-failure semantics. Every
variable of a negated atom must also appear in a positive atom of the rule,
or, as ``s`` here, appear nowhere else: it is then existentially quantified
inside the negation.


Part 7: The Bayes Factor Formula
==================================

7.1 Arithmetic Expressions
---------------------------

Datalog rules support arithmetic expressions with ``+``, ``-``, ``*``,
``/``, and ``**`` (exponentiation), with standard operator precedence:

.. code-block:: neurolang

    (bf == (p_rt / p_r) / ((p_t - p_rt) / (1.0 - p_r)))

Parentheses control grouping, and the whole comparison is itself enclosed in
parentheses.  The ``==`` operator binds the result of the expression to the
variable on the left.

7.2 The Complete Bayes Factor Rule
------------------------------------

The BF formula combines the three probability distributions with arithmetic:

.. code-block:: neurolang

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

The ``ans`` predicate marks a query, run as soon as it is entered; with the
rules of Parts 1–7 in the session:

.. nl-repl::
    :norun:

    ans(r, t, bf) :-
        bayes_factor(r, t, bf),
        (r == 'R G oc temp lat fusifor')
    .sort bf:desc

``.sort bf:desc`` shows the answer with the largest Bayes factors first; in
the browser, a click on the column's header does it.

8.2 Filtering with Comparisons
--------------------------------

The ``(r == 'R G oc temp lat fusifor')`` filter uses the equality comparison
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

The ``(r == 'R G oc temp lat fusifor')`` filter does more than just post-process
— NeuroLang's **magic-sets** optimisation pushes the constant backwards
through all the rules in the chain:

1. The constant ``'R G oc temp lat fusifor'`` is inlined into the
   ``bayes_factor`` predicate before the SIPS (Sideways Information Passing
   Strategy) pass.
2. The SIPS creates an adorned predicate with the first argument marked as
   *bound*.
3. Magic rules propagate the bound value down the dependency chain:
   ``magic_joint_probability(r)``, ``magic_region_probability(r)``,
   ``magic_cooccurrence(r)``, ``magic_active_region(r)``.
4. Each adorned rule filters its body with a magic predicate, so only rows
   where ``r = 'R G oc temp lat fusifor'`` are evaluated.

The result: of the 14 371 studies, only the 2 836 that activate the right
fusiform gyrus are evaluated for the joint probabilities, and the query
completes in about 20 seconds.

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

Putting it all together, the program at the top of the page:

.. nl-repl::

    % The regions and the studies activating them, from the Destrieux atlas
    activates(s, r) :- destrieux_activated(r, s)
    region(r) :- destrieux(r, x)

    % Deterministic intermediate rules
    active_region(r) :- region(r), selected_study(s), activates(s, r)
    mentioned_term(t) :- term(t), selected_study(s), mentions(s, t)
    cooccurrence(r, t) :-
        region(r), term(t), selected_study(s),
        activates(s, r), mentions(s, t)

    % Probabilistic rules — marginal probabilities
    region_probability(r, p_r) :- PROB[active_region(r)] = p_r
    term_probability(t, p_t) :- PROB[mentioned_term(t)] = p_t
    joint_probability(r, t, p_rt) :- PROB[cooccurrence(r, t)] = p_rt

    % Bayes Factor formula
    bayes_factor(r, t, bf) :-
        joint_probability(r, t, p_rt),
        region_probability(r, p_r),
        term_probability(t, p_t),
        (bf == (p_rt / p_r) / ((p_t - p_rt) / (1.0 - p_r)))

    % Query — filter for the target region
    ans(r, t, bf) :- bayes_factor(r, t, bf), (r == 'R G oc temp lat fusifor')
    .sort bf:desc

It answers 3 228 terms; the top 10:

.. list-table:: Top terms by Bayes Factor for the right fusiform gyrus
   :header-rows: 1

   * - Term
     - Bayes Factor
   * - ffa
     - 8.92
   * - face ffa
     - 8.43
   * - fusiform gyri
     - 7.50
   * - fusiform face
     - 7.34
   * - visual stream
     - 4.50
   * - fusiform
     - 4.18
   * - face recognition
     - 3.97
   * - ventral visual
     - 3.95
   * - fusiform gyrus
     - 3.88
   * - occipital temporal
     - 3.80

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

.. code-block:: neurolang

    ans(x, cnt) :- AGGREGATE[x](body(x, y) @ count(y)) = cnt.

The group-by variables appear in square brackets.  The ``@`` sign separates
the body from the aggregation function call.

Supported functions: ``count``, ``sum``, ``max``, ``min``, ``mean``,
``avg``, ``std``.

``count(y)`` counts the distinct values of ``y`` in each group, and
``count(y, z)`` the distinct pairs: ``count(s)`` over
``schaefer_activated(l, s)`` counts each study once, however many parcels
``l`` it activates.  The other functions take a value per row of the body:
``sum(w)`` over ``weight(s, w)`` adds the weights of all the studies ``s``,
equal or not, and ``mean(v)`` over the replicates ``rep(b, v)`` of a
resampling averages all of them.

Example — count studies per region:

.. code-block:: neurolang

    study_count(r, cnt) :-
        AGGREGATE[r](activates(s, r) @ count(s)) = cnt.

Global aggregation (no group-by) uses empty brackets:

.. code-block:: neurolang

    total_studies(cnt) :-
        AGGREGATE[](study(s) @ count(s)) = cnt.

An aggregation can also be written directly in the head of a rule, the
variables of the head being the group-by variables.  This form accepts
aggregation functions with constant parameters, such as
``percentile(v, q)``: the ``q``-th percentile of ``v``, with ``q`` in
``[0, 100]``, ignoring NaN values.  It keeps, for instance, the top 10% of
the scores of each query:

.. code-block:: neurolang

    cutoff(q, percentile(v, 90)) :- score(q, s, v).
    top_score(q, s) :- score(q, s, v), cutoff(q, t), (v >= t).

Percentiles are linearly interpolated and ties are kept, so the top 10%
can hold slightly more than 10% of the rows.

Arithmetic in the head of a rule, ``ans(x, n / 2) :- r(x, n)``, is not an
aggregation, and is refused: a value computed from each row is computed in
the body, and named in the head,
``ans(x, v) :- r(x, n), (v == n / 2)``; a value computed from an
aggregation, by a rule of its own over the aggregation's rule.

10.2 Existential Predicates
----------------------------

The ``exists`` construct asserts that there exists a tuple satisfying a set
of conditions, without binding the existential variable in the rule head:

.. code-block:: neurolang

    has_activations(r) :- region(r), exists(s; activates(s, r)).

This is equivalent to:

.. code-block:: neurolang

    has_activations(r) :- region(r), activates(s, r).

The ``;`` separates the existential variable from the body predicates.
Multiple existential variables can be stacked:

.. code-block:: neurolang

    ans(r) :- exists(s, t; activates(s, r), mentions(s, t)).

10.3 Comparisons
-----------------

Besides equality (``==``), the full comparison suite is available:

.. code-block:: neurolang

    large_items(x) :- item(x), item_count(x, c), (c >= 2).
    not_empty(x) :- item_count(x, c), (c != 0).
    small(x) :- item(x), item_count(x, c), (c < 10).

Comparisons can be combined with conjunction in the same rule body.

10.4 Commands and Directives
-----------------------------

Commands use the ``.`` prefix and follow the Prolog convention.  The text
Datalog frontend executes two commands which load external data into a
relation:

.. code-block:: neurolang

    .load_csv(study_metadata, "https://example.com/metadata.tsv", sep="\t")
    .load_atlas(destrieux, "atlas.nii.gz", "labels.csv", "https://example.com/destrieux.tgz")
    .load_atlas(fs, "aparc+aseg.mgz", "FreeSurferColorLUT.txt", kind="freesurfer", space="MNI152")
    .load_atlas(schaefer, kind="schaefer", n_rois=100)

``.load_atlas`` adds the relation ``name(label, region)`` of an atlas: an
image and a CSV of its labels at URLs (the first form, downloaded to
``~/neurolang_data``), local files of ``kind`` ``"freesurfer"`` (a colour
LUT), ``"bids"`` (a ``dseg.tsv``) or ``"tsv"``, the atlases of the engines
(``"destrieux"``, ``"schaefer"``, ``"tian"``) or a fetcher of nilearn
(``kind="nilearn", fetcher="harvard_oxford"``).  The engine records the
atlas, a :class:`~neurolang.atlases.Atlas`, by name, in its ``atlases``.

and ``.import``, which imports a *Datalog module*, a file of facts, rules
and commands, as if its statements were written in its place:

.. code-block:: neurolang

    .import("bundles/uncinate.dl")
    .import("bundles/freesurfer_queries.dl")
    ans(x) :- uf(x, "left").

* The path of a module is relative to the directory of the importing file:
  the program run with ``neurolang-query --file``, or the module importing
  another; otherwise it is relative to the working directory.  A module not
  found there is looked for in the directories an engine gives its modules
  (``module_directories`` of its solver): with the engine ``tracts``,
  ``.import("bundles/freesurfer_queries.dl")`` is one of its modules.
  ``package:path`` is the file ``path`` shipped in the Python package
  ``package``, found with :mod:`importlib.resources`, so that an engine or
  a package can ship modules without their users knowing where it is
  installed.
* A module is imported once in a program (by its resolved path): importing
  it again, from the program or from another module, does nothing.  The
  imports belong to the scope they are made in: leaving a ``with
  nl.scope:`` block, or ``.clear`` in the REPL, forgets them with the
  rules they defined, and the module can be imported again.
* A module imported while it is being imported is an import cycle, an error
  naming the modules of the cycle.
* A module defines; it does not query: a module with a query
  (``ans(...) :- ...``) is refused.  An error in a module names its file,
  with the line and column of a syntax error.

:doc:`tutorial_python` imports modules from Python.

Any other command (for instance ``.set_backend('polars')``) is rejected with
an ``UnsupportedProgramError``.  To change the relational algebra backend
(``'pandas'`` or ``'polars'``) from Datalog code, use the SQUALL
``#set_backend`` directive, or Python (:doc:`tutorial_python`); see
:doc:`performance`.
Note that the backend is a process-wide setting.

10.5 Probabilistic Facts
-------------------------

Explicit probabilistic facts can be written inline:

.. code-block:: neurolang

    0.9 :: noisy_region('fusiform').

This creates a probabilistic fact with probability 0.9.

Probabilistic rules (with a probability expression instead of a fixed
probability) also exist:

.. code-block:: neurolang

    p(x) :: exp(-d / 5.0) :- a(x, d), (d < 0.8).


10.6 Probabilistic Choices
---------------------------

Probabilistic choices represent mutually exclusive alternatives — exactly one
alternative from each choice group holds in any possible world.  This is
different from probabilistic facts (``::``), where each fact is an
independent Bernoulli trial.

The choice operators are ``^`` and ``:~:`` (both are equivalent):

**Choice fact** — a standalone choice with a fixed probability:

.. code-block:: neurolang

    0.3 ^ choice('a').
    0.7 ^ choice('b').

This declares two mutually exclusive alternatives: ``choice('a')`` holds with
probability 0.3 and ``choice('b')`` holds with probability 0.7.  The
probabilities in a choice group must sum to 1.

**Choice rule** — a choice whose head depends on a body:

.. code-block:: neurolang

    n_studies(n) :- AGGREGATE[](study(s) @ count(s)) = n.
    picked_study(s) ^ (1 / n) :- study(s), n_studies(n).

Here the probability expression ``(1 / n)`` is evaluated at runtime, with
``n`` bound in the body to the number of studies; each tuple in the body
produces one arm of the choice, so ``picked_study`` is a uniform choice over
studies.  The ``:~:`` operator can be used interchangeably:

.. code-block:: neurolang

    picked_study(s) :~: (1 / n) :- study(s), n_studies(n).

10.7 Confidence Intervals by Resampling
-----------------------------------------

The Bayes factors of Part 9 are computed from one sample of studies.  How
much they would change with another sample is answered by resampling the
studies: a *resampling* of the choice ``selected_study`` is a set of
replicates ``b``, each reweighting the studies as a bootstrap of them does.
It is declared once, in the engine's configuration
(:doc:`engine_configuration`) or from Python (:doc:`tutorial_python`): the
``neurosynth`` engine has ``boot_studies``, 500 replicates of a Bayesian
bootstrap of ``selected_study``.

``RESAMPLE[boot_studies](pred(x)) = b`` in a rule body is then
``pred(x)`` computed on replicate ``b``, for any ``pred`` defined over
``PROB`` of the choice, including the arithmetic over the probabilities.
The confidence interval is the percentiles of the replicates:

.. nl-repl::
    :norun:

    bf_rep(b, r, t, bf) :- RESAMPLE[boot_studies](bayes_factor(r, t, bf)) = b

    ans(t, bf, percentile(x, 2.5), percentile(x, 97.5)) :-
        bayes_factor('R G oc temp lat fusifor', t, bf),
        bf_rep(b, 'R G oc temp lat fusifor', t, x),
        (bf > 4)

With the rules of Part 9, this gives the 95% intervals of the six terms of
Bayes factor above 4, in under a minute: ``ffa``, 8.92 [5.96, 13.64];
``face ffa``, 8.43 [5.40, 13.23]; ``fusiform gyri``, 7.50 [4.90, 11.28];
``fusiform face``, 7.34 [5.39, 10.23]; ``visual stream``, 4.50 [2.92, 7.09];
``fusiform``, 4.18 [3.73, 4.69].

The replicates of a resampling are the same in every query, so that two
measures computed on them can be compared replicate by replicate, joining
them on ``b``:

.. nl-repl::
    :norun:

    difference(b, d) :-
        bf_rep(b, 'R G oc temp lat fusifor', 'ffa', x),
        bf_rep(b, 'R G oc temp lat fusifor', 'navigation', y),
        (d == log(x) - log(y))
    ans(percentile(d, 2.5), percentile(d, 97.5)) :- difference(b, d)

The interval of the log ratio, [0.30, 1.51], excludes 0: ``ffa`` is more
specific to the region than ``navigation`` (3.57), in this sample of studies.

The interval of a difference cannot be obtained from the intervals of the
two measures, which are computed on the same studies and are correlated.

**Schemes.**  ``scheme`` sets the weight of each study in a replicate:
``"bayesian"``, the Bayesian bootstrap (Rubin, 1981, Ann. Stat.), weights
of a uniform Dirichlet distribution, never zero, so that a rare term keeps
a defined Bayes factor; ``"multinomial"``, the classical bootstrap, how
many times the study is drawn among ``N`` draws with replacement;
``"poisson"``, a Poisson(1) count.  ``seed`` fixes the draws.

**How it is computed.**  If the choice ``C`` takes the outcome ``c`` with
probability :math:`P(C = c)` and the replicate ``b`` with probability
:math:`P_b(C = c) = w_{bc}\, P(C = c)`, then for any query ``q``, as the
choice holds for exactly one outcome in every world,

.. math::

   P_b(q) = \sum_c P_b(C = c)\, P(q \mid C = c)
          = \sum_c w_{bc}\, P(q \wedge C = c).

:math:`P(q \wedge C = c)` is ``q`` with the outcome of the choice as one
more answer variable, which the probabilistic solvers compute as any other
query, and the sum over ``c`` is an aggregation joined with the resampling
relation ``boot_studies(b, c, w)``.  Hence ``RESAMPLE`` is a rewriting of
the rules ``pred`` depends on and needs no other solver:

1. the rules below a ``PROB`` that depend on the choice take the outcome
   ``c`` as their first argument, the same for all the atoms of a body,
   since all the atoms of the choice hold for the same outcome;
2. each ``PROB`` over them becomes :math:`P(q \wedge C = c)`, then its sum
   weighted by each replicate; a conditional ``PROB[q // e]`` is the ratio
   of two of them;
3. the rules over these probabilities take the replicate ``b`` as their
   first argument, the same for all the atoms of a body, so that the
   probabilities a measure combines come from the same replicate.

The rewritten rules are named after the resampling (``bayes_factor@boot_studies``)
and are added to the program once, then reused by every ``RESAMPLE`` of the
same predicate.  Each replicate recomputes the measure, so that a query over
``n`` replicates costs about ``n`` times its rows: ask for one region or
term rather than for every voxel.


.. _appendix-a-datalog:

Appendix A: Running Datalog from Python
=========================================

Every program of this tutorial runs from Python too, as the text of
``execute_datalog_program`` on a :class:`~neurolang.frontend.NeurolangPDL`,
an engine of the registry or one with relations of your own: see
:doc:`tutorial_python`.


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

A result on the regions of an atlas, a column of region labels such as
``l`` of ``destrieux_activated(l, s)``, is painted region by region, on
the atlas's grid, with the value of its rows; ``--view`` starts with the
regions of the top 5% of the values, as for voxels (``--threshold`` changes
it), and with every region of a result without values.  The probability that a study mentioning language
activates each Destrieux region::

    neurolang-query --engine neurosynth --view \
        "active(l, s) :- destrieux_activated(l, s), selected_study(s)
         term_association(s, t) :- mentions(s, t), selected_study(s)
         ans(l, p) :- PROB[active(l, s) // term_association(s, 'language')] = p"

The viewer names the region of the result under the mouse, and under the
cursor below the views; its control *atlas colours* draws the regions in
the colours their atlas gives them: Schaefer's networks, FreeSurfer's for
Destrieux (``aparc.a2009s``), those of Tian's subcortex.  A result of
regions alone, ``ans(l) :- schaefer_label(l), startswith('17Networks LH
Default', l)``, starts in them.

A result of pairs of regions, or of points (``x1, y1, z1, x2, y2, z2``),
with a value is a network: ``--view`` draws a node at the centre of each
region and an edge per pair, its colour and thickness the value, from
the thinnest at the edge threshold to the thickest at the largest value,
in the 3D render of the brain and on the slices, and lists the strongest
edges.  The edge threshold, the strongest 5% by default (``--threshold``),
is changed in the viewer, under the views: "Edges from" and its slider
draw the network again with the edges above it.  On a slice, the mouse
over a node shows the name of its region, and over an edge the names of
the two regions it joins and its value.  The co-activation of the
Schaefer regions by studies about reading, the probability that a study
mentioning reading activates both regions of a pair::

    neurolang-query --engine neurosynth --view \
        "active(l, s) :- schaefer_activated(l, s), selected_study(s)
         co_active(l1, l2, s) :- active(l1, s), active(l2, s)
         term_association(s, t) :- mentions(s, t), selected_study(s)
         ans(l1, l2, p) :- (l1 < l2), PROB[co_active(l1, l2, s) // term_association(s, 'reading')] = p"

Both work with ``--ask``, which then runs the query it writes.

Where an answer comes from
--------------------------

In the interactive mode, ``.graph`` draws the last query, run or written by
the assistant with ``.ask``, in a page opened in the browser; the REPL in
the browser shows it in its *Graph* tab after each query. It reads from left
to right: the data of the engine (its relations, its random choices, such as
``selected_study``, and its probabilistic facts), the rules of the query and
of the engine they go through, and the answer. A rule lists the atoms of its
body, where the edges of the relations they use arrive, and how it combines
them: the variables it joins them on, and those it keeps or drops. A rule
using a random choice is probabilistic: its rows hold for some choices only
(orange edges). ``PROB[...]`` turns it into the probability of each of its
rows, given the condition after ``//`` (dashed edges): with one uniform
choice, such as ``selected_study``, the share of the studies where the body
and the condition hold among those where the condition holds. After a run,
the answer shows the distribution of its probabilities. Clicking a node
shows its rules and what it depends on. The relations that come from an
ontology, such as the triples of the Cognitive Atlas (``rdfs:label``), are
teal hexagons, apart from the blue cylinders of the regular tables.

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
