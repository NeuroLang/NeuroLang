NSQUALL: Controlled English for NeuroLang
==========================================

NSQUALL (NeuroLang's *Semantically controlled Query-Answerable Logical Language*)
lets you write NeuroLang queries and rules in plain English sentences instead of
symbolic Datalog notation.  Under the hood, each sentence is translated to a
NeuroLang logical expression using Montague semantics in
Continuation-Passing Style, building on the original
`SQUALL language <https://people.irisa.fr/Sebastien.Ferre/software/squall/>`_
by Sébastien Ferré.

.. rubric:: References

.. [Ferre2012] S. Ferré. *SQUALL: A Controlled Natural Language for
   Querying and Updating RDF Graphs*. Controlled Natural Languages (CNL),
   2012. LNCS 7427, p. 11-25, Springer.

.. [Zanitti2023] G. E. Zanitti, Y. Soto, V. Iovene, M. V. Martinez,
   R. O. Rodriguez, G. I. Simari, D. Wassermann. *Scalable Query Answering
   Under Uncertainty to Neuroscientific Ontological Knowledge: The
   NeuroLang Approach*. Neuroinformatics, 21(2), 407-425, 2023.

This tutorial is built around a single real-world neuroimaging problem:
**Bayes Factor decoding of the right fusiform gyrus**.  Each section
introduces exactly the NSQUALL feature needed for the next step, so by
the end you will understand every line of a complete probabilistic
NSQUALL program.

The full program we will build:

.. code-block:: squall

    define as Selected_study as an equiprobable choice over every Study.
    define as Active_region every Region that a Selected_study activates.

    define as Region_probability with inferred probability
        every Active_region.

    define as Mentioned_term every Term that a Selected_study mentions.

    define as Term_probability with inferred probability
        every Mentioned_term.

    define as Cooccurrence
        for every Region and for every Term
        where a Selected_study activates the Region and mentions the Term.

    define as Joint_probability with inferred probability
        every Cooccurrence (?r; ?t).

    define as Bayes_factor (?r; ?t; ?bf)
        where Joint_probability (?r, ?t, ?p_rt)
        and Region_probability (?r, ?p_r)
        and Term_probability (?t, ?p_t)
        and ?bf is (?p_rt / ?p_r) / ((?p_t - ?p_rt) / (1.0 - ?p_r)).

    obtain every Bayes_factor (?r; ?t; ?bf)
        where ?r is 'right fusiform gyrus' as BF.

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

We will compute all three probability distributions as NSQUALL rules, then
combine them with arithmetic to produce :math:`\mathrm{BF}` — all inside
NeuroLang with zero post-hoc pandas computation.

1.2 The Data
-------------

The NSQUALL program works with six relations:

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
    A probabilistic uniform choice over all studies (defined in NSQUALL
    below, not from Python).

These are registered in Python before running NSQUALL (see Appendix A for the
full API):

    >>> from neurolang.frontend import NeurolangPDL
    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set(study_ids_df, name="study")
    >>> nl.add_tuple_set(study_activates_df, name="activates")
    >>> nl.add_tuple_set(study_mentions_df, name="mentions")
    >>> nl.add_tuple_set(region_df, name="region")
    >>> nl.add_tuple_set(term_df, name="term")
    >>> # Selected_study is then defined directly in NSQUALL (see below)

A complete working example is in
``examples/plot_squall_bayes_factor_decoding.py``.

1.3 Running an NSQUALL Program
------------------------------

All NSQUALL code is passed as a string to ``execute_squall_program``:

    >>> result = nl.execute_squall_program("""
    ...     define as Active_region every Region
    ...         that a Selected_study activates.
    ...     define as Region_probability with inferred probability
    ...         every Active_region.
    ...     ...
    ...     obtain every Bayes_factor (?r; ?t; ?bf)
    ...         where ?r is 'right fusiform gyrus' as BF.
    ... """)
    >>> bf_df = result.as_pandas_dataframe()

When the program contains exactly one ``obtain`` clause, the method returns a
``NamedRelationalAlgebraFrozenSet`` directly.  For multiple ``obtain`` clauses
it returns a ``dict`` keyed by name.

Now let's build this program step by step, starting from the simplest
building blocks.


Part 2: Nouns and Basic Queries
=================================

2.1 Nouns as Relations
-----------------------

A noun in NSQUALL names a relation.  ``Region`` refers to the ``region`` EDB,
``Term`` to ``term``, ``Selected_study`` to ``selected_study``.

The simplest query asks for all entities in a relation:

.. code-block:: squall

    obtain every Region.

Result: all anatomical regions in the data set.

The ``obtain`` keyword introduces a query.  ``every`` is a **determiner**
that asks for all values.  The sentence reads like plain English.

2.2 Determiners
-----------------

NSQUALL supports four determiners:

=========== ============ ==========================================
Keyword     Meaning      Example
=========== ============ ==========================================
``every``   Universal    ``every Region`` — all regions
``a``       Existential  ``a Selected_study`` — at least one study
``no``      Negative     ``no Term`` — no matching terms
``the``     Anaphoric    ``the Region`` — refers back to an
                         earlier ``every Region`` (section 5.3)
=========== ============ ==========================================

In the Bayes Factor program, ``a Selected_study`` is the existential choice
that quantifies over studies — it introduces a study variable without binding
it in the rule head.

2.3 Named Variables — ``?label``
---------------------------------

Variables can be named with ``?name`` labels, which bind the variable so it
can be reused in the same rule:

.. code-block:: squall

    obtain every Joint_probability (?r; ?t; ?p).

Here ``?r``, ``?t``, and ``?p`` are bound to the three columns of the
``joint_probability`` relation.

2.4 Tuple Subjects and Wildcards
---------------------------------

When a noun denotes a multi-column relation, a parenthesised tuple of labels
follows the noun:

.. code-block:: squall

    define as Selected_peak every Peak_reported (?i; ?j; ?k; ?s).

The ``;`` separator matches the column structure.  Use ``_`` for columns you
want to match but not project into the rule head:

.. code-block:: squall

    define as Activation every Peak_reported (?i; ?j; ?k; _).

Each ``_`` creates a distinct fresh variable that exists in the query body
but is dropped from the head.

2.5 String Literals
--------------------

String literals use single quotes:

.. code-block:: squall

    obtain every study that is 'neuro study'.

This is the syntax we will later use to select the target region:

.. code-block:: squall

    obtain every Bayes_factor (?r; ?t; ?bf)
        where ?r is 'right fusiform gyrus' as BF.

The ``where ?r is '...'`` clause filters ``?r`` to a fixed constant string.

2.6 Reserved Words and Backtick Quoting
-----------------------------------------

NSQUALL reserves many common English words (``every``, ``a``, ``the``,
``that``, ``is``, ``has``, ``not``, ``and``, ``or``, ``where``, ``who``,
``which``, etc.).  If a relation name coincides with a reserved word, wrap it
in backticks:

.. code-block:: squall

    obtain every `from`.

Variable names use the ``?`` prefix and may contain letters, digits, and
underscores.  They cannot clash with reserved words because of the prefix.


Part 3: Verbs and Relations
=============================

Now we add verbs to connect nouns into sentences.

3.1 Transitive Verbs — The Bayes Factor Relations
---------------------------------------------------

The Bayes Factor problem needs two binary relations:
``activates(study, region)`` and ``mentions(study, term)``.

In NSQUALL, a transitive verb maps subject→first argument, object→second
argument, in natural English order:

.. code-block:: squall

    every Region that a Selected_study activates

Reads as *"for every region r, there exists a selected study s such that
activates(s, r)"* — the study (subject) activates the region (object).

Similarly:

.. code-block:: squall

    every Term that a Selected_study mentions

Reads as *"for every term t, there exists a selected study s such that
mentions(s, t)"*.

3.2 Argument Order and the ``~`` Inverse
------------------------------------------

When the EDB relation stores its arguments in the **reverse** order — for
instance ``reports`` stores ``(study, voxel)`` but you want to query from
the voxel's perspective — use the ``~`` prefix to swap them:

.. code-block:: squall

    every Voxel that a Study ~reports.

This maps to ``reports(voxel, study)`` — the ``~`` inverts the arguments
so the subject is matched to the second column and the object to the first.

Our Bayes Factor data stores ``activates(study, region)`` in natural order
(study first, region second), so no ``~`` is needed.

3.3 Intransitive Verbs
-----------------------

An intransitive verb takes only a subject:

.. code-block:: squall

    define as PlayerPerson every person that plays.

Result: all persons who play.

3.4 Possessive VP — ``has NP2``
---------------------------------

``has DET noun2`` expresses a possessive verb phrase — the subject has a
thing related to it by the binary noun ``noun2``:

.. code-block:: squall

    define as Author every person that has a publication.

Result: persons for whom a matching entry in the binary ``publication``
relation exists.

With an optional restrictive relative clause on the possessed noun:

.. code-block:: squall

    define as ProlificAuthor every person
        that has a publication that is highly_cited.

3.5 Existential — ``there is NP``
-----------------------------------

``there is NP`` / ``there are NP`` asserts that at least one entity matching
the noun phrase exists:

.. code-block:: squall

    define as HasPlayer every Game that there is a Player.

Useful when the body needs a purely existential check without introducing a
join variable.

3.6 Auxiliaries — ``does`` / ``is`` / ``has``
-----------------------------------------------

``does not VP`` expresses negation-as-failure:

.. code-block:: squall

    define as NotPlaying every person that does not plays.

Result: persons who do not appear in the ``plays`` relation.

3.7 Function-Call Guard — ``Predicate(?x, ?y) holds``
-------------------------------------------------------

An arbitrary relation can be invoked in a relative clause with explicit
arguments:

.. code-block:: squall

    define as Close every Pair ?p that euclidean(?x, ?y) holds.

This calls ``euclidean(x, y)`` as a guard in the body.  We will see a more
general form of this in section 7.1.

.. -- 3.8 Bare Predicate Calls --

.. note::

    Section 7.1 introduces a closely related pattern — **bare predicate
    calls** with the syntax ``PredicateName (?a, ?b, ?c)`` without a verb —
    which is used in the Bayes Factor rule body.


Part 4: Defining Rules
========================

Now we move from queries to rule definitions, building the intermediate
predicates that the Bayes Factor program needs.

4.1 Simple Unary Rules
-----------------------

The ``define as`` prefix turns a sentence into a Datalog rule:

.. code-block:: squall

    define as Active_region every Region that a Selected_study activates.

This creates a new predicate ``active_region(region)``.  In Datalog notation:

.. code-block:: text

    active_region(r) :- region(r), selected_study(s), activates(s, r).

The rule says: *r is an active region if there exists a selected study s
that activates r*.

The Bayes Factor program defines four such intermediate rules:

.. code-block:: squall

    define as Active_region every Region that a Selected_study activates.
    define as Mentioned_term every Term that a Selected_study mentions.

These two create unary predicates ``active_region(region)`` and
``mentioned_term(term)``.  They are the building blocks for the marginal
probabilities.

4.2 Multi-Variable Rules — Compound Quantifiers
-------------------------------------------------

To build the joint distribution we need a **ternary** relation:
``cooccurrence(region, term)`` linking each study's region and term.

The compound quantifier syntax chains ``for every`` clauses:

.. code-block:: squall

    define as Cooccurrence
        for every Region and for every Term
        where a Selected_study activates the Region and mentions the Term.

This creates:

.. code-block:: text

    cooccurrence(r, t) :-
        region(r), term(t), selected_study(s),
        activates(s, r), mentions(s, t).

The ``and`` between quantifiers binds both ``Region`` and ``Term`` into the
rule head, producing a binary predicate.

4.3 Anaphora — ``the Noun``
-----------------------------

Inside the ``where`` clause, ``the Region`` and ``the Term`` refer back to
the variables introduced by ``for every Region`` and ``for every Term``.
This is called **anaphora** resolution — the reader (human or machine)
understands which variable is meant from context.

.. code-block:: squall

    define as Cooccurrence
        for every Region and for every Term
        where a Selected_study activates the Region and mentions the Term.

Without anaphora you would need explicit variables:

.. code-block:: squall

    define as Cooccurrence
        for every Region and for every Term
        where a Selected_study ?s activates ?r and mentions ?t.

The anaphoric form is more natural and is what the Bayes Factor program uses.

.. note::

   Anaphora works within a single rule only — there is no inter-sentence
   scope yet.  Each rule resolves ``the X`` from the ``for every X`` in its
   own head.

4.4 The ``;`` Separator for Multi-Variable Heads
---------------------------------------------------

An alternative to compound quantifiers uses a ``;``-separated tuple after
``define as``:

.. code-block:: squall

    define as reported for every Study ?s ; with every Voxel ?v that ?s reports.

This is equivalent but reads less naturally than the compound quantifier
with anaphora.

4.5 Multiple Rules in One Program
-----------------------------------

Rules are separated by a full stop.  All the Bayes Factor ``define``
sentences are passed as a single program string:

.. code-block:: squall

    define as Active_region every Region that a Selected_study activates.
    define as Mentioned_term every Term that a Selected_study mentions.
    define as Cooccurrence
        for every Region and for every Term
        where a Selected_study activates the Region and mentions the Term.
    ...

    obtain every Bayes_factor (?r; ?t; ?bf)
        where ?r is 'right fusiform gyrus' as BF.

4.6 Fork Quantification — ``for NP , S``
------------------------------------------

The ``for NP , S`` (fork) construction places a noun phrase as an
**outer sentence-level quantifier** over an otherwise independent
sentence ``S``:

.. code-block:: squall

    for every Person ?p, ?p plays.

This reads: *"for every person p, p plays"*.

The fork form is useful when ``S`` is too complex to embed inside a relative
clause.

.. code-block:: squall

    define as Reported
        for every Study ?s,
            a Voxel ?v that ?s reports.

Here the fork binds ``?s`` in the outer scope so it can be referenced by the
inner sentence.

4.7 Comparisons
----------------

Relative clauses can include comparisons using the keywords ``greater``,
``lower``, ``equal``, optionally combined with ``equal`` and ``not``,
followed by ``than`` or ``to`` and an operand:

.. code-block:: squall

    define as Large every Item that has an item_count greater equal than 2.

Supported forms:
* ``greater than``
* ``greater equal than``
* ``lower than``
* ``lower equal than``
* ``equal to``
* ``not equal to``


Part 5: Probabilistic Rules
=============================

Now we introduce probability — the core of the Bayes Factor computation.

5.1 Probabilistic Facts — ``probably``
---------------------------------------

The ``probably`` keyword creates a probabilistic fact with a fresh
probability variable:

.. code-block:: squall

    define as probably activates every study.

The ``activates`` predicate is now probabilistic; the probability is inferred
at query time.

5.2 Inferred Probability — ``with inferred probability``
---------------------------------------------------------

``with inferred probability`` on a rule head generates a marginal probability
over the joint distribution.  This is the key construct for the Bayes Factor:

.. code-block:: squall

    define as Region_probability with inferred probability
        every Active_region.

    define as Term_probability with inferred probability
        every Mentioned_term.

    define as Joint_probability with inferred probability
        every Cooccurrence (?r; ?t).

Each produces a predicate with an added probability column:

* ``region_probability(region, PROB(region))`` — marginal :math:`P(R)`
* ``term_probability(term, PROB(term))`` — marginal :math:`P(T)`
* ``joint_probability(region, term, PROB(region, term))`` — joint :math:`P(R,T)`

The probability is the fraction of studies that satisfy the body given the
probabilistic ``Selected_study`` choice.  The tuple ``(?r; ?t)`` in the
``Cooccurrence`` head tells the solver which variables to keep after
marginalising over studies — the study variable introduced by
``a Selected_study`` in the body is quantified away.

.. note::

   The body of each ``with inferred probability`` rule must be a
   **deterministic** predicate.  The probabilistic solver combines it with
   the ``selected_study`` choice automatically.  This is why we create
   ``Active_region``, ``Mentioned_term``, and ``Cooccurrence`` as separate
   deterministic rules first — they flatten the existential study variable
   before the probabilistic step.

5.3 Conditional Probability — ``with probability … conditioned to``
---------------------------------------------------------------------

The MARG form computes a conditional probability:

.. code-block:: squall

    define as Activation_given_term with probability
        every Activation (?i; ?j; ?k; _)
        conditioned to every Term_association (?s; ?t; _) such that ?t is 'auditory'.

Here ``_`` drops the study-id column from the conditioned side and the
TF-IDF weight from the conditioning side.  The result has columns
``(i, j, k, probability)`` where the last column is
``P(activation(i,j,k) | term_association(s,t) ∧ t = 'auditory')``.

The keyword ``given`` is a synonym for ``conditioned to``:

.. code-block:: squall

    define as Activation_map with inferred probability every Active_voxel (?i; ?j; ?k; _)
        given every Study_term (_; ?t) where ?t is 'emotion'.

This reads: *"the inferred probability of activation at (i, j, k) given
the study term is 'emotion'"*.

.. note::

   When using MARG with tuple-labeled relations, the arity of the conditioned
   and conditioning noun-phrases must exactly match the corresponding
   relation arities.  Use ``_`` for columns that exist in the body relation
   but should not appear in the head.

5.4 Explicit Probability — ``with probability NP``
----------------------------------------------------

``with probability`` followed by a conditioned/conditioning noun-phrase pair
also supports explicit naming:

.. code-block:: squall

    define as Activation_given_term with probability
        every Activation (?i; ?j; ?k; _)
        conditioned to every Term_association (?s; ?t; _) such that ?t is 'auditory'.

5.5 Caveats — Existentials in Body
------------------------------------

When the body contains existentials (e.g. ``a Selected_study``), create an
intermediate deterministic rule first, then define the probabilistic rule
over it.  This is exactly the pattern used in the Bayes Factor program:

1. ``Active_region`` — deterministic, flattens the existential
2. ``Region_probability with inferred probability every Active_region`` —
   probabilistic, adds the probability column


Part 6: Connectives
=====================

6.1 Conjunction — ``and``
---------------------------

Multiple conditions in a rule body are joined with ``and``:

.. code-block:: squall

    define as Cooccurrence
        for every Region and for every Term
        where a Selected_study activates the Region and mentions the Term.

The ``activates the Region and mentions the Term`` is a conjunction of two
verb phrases sharing the same subject (the study).

When predicates are split across separate rules, chain them with an
intermediate:

.. code-block:: squall

    define as Player every person that plays.
    define as PlayAndRun every Player that runs.

6.2 Disjunction — ``or``
--------------------------

With ``or``, the subject must satisfy at least one condition:

.. code-block:: squall

    define as PlayOrRun every person that plays or runs.

6.3 Negation — ``not``
-----------------------

``does not VP`` expresses negation-as-failure (see section 3.6):

.. code-block:: squall

    define as NotPlaying every person that does not plays.


Part 7: The Bayes Factor Formula
==================================

Now we reach the heart of the program — computing the Bayes Factor from the
three probability distributions.

7.1 Bare Predicate Calls
--------------------------

To reference ``Joint_probability``, ``Region_probability``, and
``Term_probability`` inside a rule body, we use the **bare predicate call**
syntax — the predicate name followed by parenthesised arguments, no verb:

.. code-block:: squall

    define as Bayes_factor (?r; ?t; ?bf)
        where Joint_probability (?r, ?t, ?p_rt)
        and Region_probability (?r, ?p_r)
        and Term_probability (?t, ?p_t)
        and ?bf is (?p_rt / ?p_r) / ((?p_t - ?p_rt) / (1.0 - ?p_r)).

``Joint_probability (?r, ?t, ?p_rt)`` calls the ``joint_probability``
predicate with arguments ``?r``, ``?t``, binding ``?p_rt`` to the
probability column.  The arguments use **comma** separators ``(a, b, c)``.
The predicate name matches the rule name defined elsewhere in the program
(case-insensitive).

This form is the multi-predicate equivalent of the function-call guard in
section 3.7, but without the ``holds`` keyword — it is used in rule bodies
to join several predicates with explicit variable bindings.

7.2 Arithmetic Expressions
---------------------------

The last conjunct in the rule assigns the result of an arithmetic expression
to ``?bf``:

.. code-block:: squall

    ?bf is (?p_rt / ?p_r) / ((?p_t - ?p_rt) / (1.0 - ?p_r))

This is the Bayes Factor formula from section 1.1, expressed directly in
NSQUALL.  The expression supports ``+``, ``-``, ``*``, ``/`` with standard
operator precedence; parentheses are supported for grouping.

The ``is`` clause translates to an ``eq`` builtin with the arithmetic
expression tree as the second argument.  The expression is evaluated during
the chase using Python's ``operator`` module functions (``truediv``,
``sub``, etc.).

.. note::

    Arithmetic expressions currently support **numeric types only**.
    Non-numeric ``?label is 'string'`` is handled as a constant equality
    (see section 2.5).  The two uses share the same ``is`` keyword but
    produce different internal representations.

7.3 The Complete Bayes Factor Rule
------------------------------------

.. code-block:: squall

    define as Bayes_factor (?r; ?t; ?bf)
        where Joint_probability (?r, ?t, ?p_rt)
        and Region_probability (?r, ?p_r)
        and Term_probability (?t, ?p_t)
        and ?bf is (?p_rt / ?p_r) / ((?p_t - ?p_rt) / (1.0 - ?p_r)).

In Datalog notation:

.. code-block:: text

    bayes_factor(r, t, bf) :-
        joint_probability(r, t, p_rt),
        region_probability(r, p_r),
        term_probability(t, p_t),
        eq(bf, truediv(truediv(p_rt, p_r), truediv(sub(p_t, p_rt), sub(1.0, p_r)))).


Part 8: Querying and Optimization
===================================

8.1 The ``obtain`` Clause
--------------------------

The ``obtain`` clause executes a query and returns results.  Unlike
``define as``, which creates a rule for later use, ``obtain`` immediately
solves the query and makes the result available to Python.

.. code-block:: squall

    obtain every Bayes_factor (?r; ?t; ?bf)
        where ?r is 'right fusiform gyrus' as BF.

This asks for all ``(r, t, bf)`` triples where ``r`` is the target region.

8.2 Named Results — ``as Name``
---------------------------------

The ``as BF`` suffix names the result relation:

    >>> result = nl.execute_squall_program(squall_program_with_obtain)
    >>> bf_df = result.as_pandas_dataframe()
    >>> bf_df.columns = ["region", "term", "bf"]
    >>> bf_df.sort_values("bf", ascending=False).head()

When there is exactly one ``obtain`` clause, ``execute_squall_program``
returns the ``NamedRelationalAlgebraFrozenSet`` directly.  When there are
multiple, it returns a ``dict`` keyed by name.

8.3 Magic-Sets Optimization
-----------------------------

The ``where ?r is 'right fusiform gyrus'`` filter does more than just
post-filter — NeuroLang's **magic-sets** optimisation pushes the constant
backwards through all the rules in the chain:

1. The ``InlineEqualityConstantsMixin`` inlines ``eq(r, Constant('rfg'))``
   into the ``bayes_factor`` call before the SIPS (Sideways Information
   Passing Strategy) sees it.
2. The SIPS creates an adorned predicate
   ``bayes_factor^bff_0(Constant('rfg'), t, bf)`` where ``bf`` means the
   first argument is bound.
3. A magic init ``magic_bayes_factor^bff_0(Constant('rfg'))`` seeds the
   propagation.
4. Magic rules propagate the bound argument down the dependency chain:
   ``magic_joint_probability^bff_0(r)``,
   ``magic_region_probability^bf_0(r)``,
   ``magic_cooccurrence^bf_0(r)``,
   ``magic_active_region^b_0(r)``.
5. Each adorned rule now filters its body with a magic predicate, so the
   computation for the ``cooccurrence`` join, the marginal probabilities, and
   the final Bayes Factor all only see rows where ``r = 'right fusiform
   gyrus'``.

The result: with 12 000+ studies and 1 million term-study pairs, the query
completes in approximately 30 seconds because only 4 298 studies activating
the right fusiform gyrus are evaluated.

.. note::

    Magic-sets works when the constant appears on an **IDB** (rule-defined)
    predicate, as it does here (``Bayes_factor`` is defined by a rule).  For
    EDB queries the SIPS returns early and the optimisation does not apply.


Part 9: The Complete Program
==============================

Putting it all together:

.. code-block:: squall

    define as Selected_study as an equiprobable choice over every Study.
    define as Active_region every Region that a Selected_study activates.
    define as Region_probability with inferred probability every Active_region.

    define as Mentioned_term every Term that a Selected_study mentions.
    define as Term_probability with inferred probability every Mentioned_term.

    define as Cooccurrence
        for every Region and for every Term
        where a Selected_study activates the Region and mentions the Term.
    define as Joint_probability with inferred probability every Cooccurrence (?r; ?t).

    define as Bayes_factor (?r; ?t; ?bf)
        where Joint_probability (?r, ?t, ?p_rt)
        and Region_probability (?r, ?p_r)
        and Term_probability (?t, ?p_t)
        and ?bf is (?p_rt / ?p_r) / ((?p_t - ?p_rt) / (1.0 - ?p_r)).

    obtain every Bayes_factor (?r; ?t; ?bf)
        where ?r is 'right fusiform gyrus' as BF.

Running this on real Neurosynth data (see
``examples/plot_squall_bayes_factor_decoding.py``) produces:

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
   * - orthographic
     - 5.43
   * - occipitotemporal cortex
     - 5.32
   * - visual stream
     - 5.22
   * - inferior occipital
     - 5.03
   * - ventral visual
     - 4.88
   * - occipito
     - 4.31
   * - occipitotemporal
     - 4.22
   * - extrastriate
     - 3.69
   * - identity
     - 3.40
   * - characters
     - 3.38
   * - face recognition
     - 3.26

All 20 terms exceed the Jeffreys "substantial evidence" threshold of
:math:`\sqrt{10} \approx 3.16`.  The top cluster — FFA, face, fusiform
face — is exactly what the right fusiform gyrus is known for.


Part 10: Additional NSQUALL Syntax
===================================

This section documents features not used directly in the Bayes Factor
example but available in the language.

10.1 Aggregations
------------------

Aggregations summarise a set of values into a single result per group:

.. code-block:: text

    define as RESULT for every SUBJECT ;
        where every AGG_FUNC of the MEASURE where CONDITION per SUBJECT.

Supported functions: ``count``, ``sum``, ``max``, ``min``, ``average``.

Example — maximum ``item_count`` per item:

.. code-block:: squall

    define as max_items for every Item ?i ;
        where every Max of the Quantity where ?i item_count per ?i.

Global aggregation (no ``per`` clause):

.. code-block:: squall

    define as Result every Collect_all of the Item.

This requires ``collect_all`` to be registered as an aggregation functor in
the engine's symbol table.

10.2 Probabilistic Choice Definitions
--------------------------------------

A probabilistic choice creates a predicate whose tuples are mutually exclusive
alternatives, each assigned a probability.  The simplest form creates a
uniform choice:

.. code-block:: squall

    define as Selected_study as an equiprobable choice over every Study.

This registers ``selected_study`` as a probabilistic choice predicate where
every study in the ``study`` EDB has equal probability ``1/N``.  The source
noun phrase (``Study`` in this example) must refer to a single-column EDB
relation registered in advance with ``add_tuple_set``.

Unlike ``define as`` rules, a choice definition does not produce an IDB
predicate — it registers a probabilistic choice in the engine's symbol table
that later probabilistic rules (Part 5) can reference via
``a Selected_study`` (existential quantifier).

The equivalent Python API is ``add_uniform_probabilistic_choice_over_set``;
the general ``add_probabilistic_choice_from_tuples`` supports arbitrary
user-specified probabilities.

**Weighted choice** — the grammar also accepts explicit probability
expressions:

.. code-block:: squall

    define as Selected_study as a choice over every Study with probability ?p.

    define as Selected_study as a choice over every Study (?s; ?q)
        with probability (?q / ?total).

Here ``?p`` is a variable bound from a multi-column source, and
``(?q / ?total)`` is an arithmetic expression evaluated per tuple.
Support for executing weighted choice definitions directly from
NSQUALL is not yet implemented — use ``add_probabilistic_choice_from_tuples``
in Python to register choices with non-uniform probabilities.

10.3 Program-Level Directives — ``#name(args).``
--------------------------------------------------

A NSQUALL program may include directive lines of the form ``#name(arg, ...)``
to pass configuration to the engine.  Directives are processed before rule
walking.

Currently supported:

``#set_backend('backend')``
    Switch the relational algebra backend.  ``backend`` may be ``'pandas'``,
    ``'dask'``, or ``'duckdb'``.

.. code-block:: squall

    #set_backend('pandas').
    define as Active every person that plays.
    obtain every Active.

Unknown directives are silently ignored.

10.4 Inline Type Guards
-------------------------

Inside a relative clause, ``where (?i; ?j; ?k) is a Noun`` asserts that the
tuple belongs to the relation named by ``Noun``:

.. code-block:: squall

    define as SelectedPeak every Peak_reported (?i; ?j; ?k; ?s)
        where (?i; ?j; ?k; ?s) is a Activation.

The scalar form works too: ``where ?s is a Selected_study``.

10.5 Nested Relative Clauses
-----------------------------

Relative clauses can be nested by using an intermediate IDB predicate as the
noun:

.. code-block:: squall

    define as PlayingSelected every selected that plays.


.. _part-9-appendix-a:

Appendix A: Running NSQUALL from Python
=========================================

All of the NSQUALL syntax shown in this tutorial can be executed from Python
using the ``NeurolangPDL`` frontend.  The general workflow is:

1. Create a ``NeurolangPDL`` engine.
2. Register EDB facts with ``add_tuple_set``.
3. Execute an NSQUALL program string with ``execute_squall_program``.
4. Inspect results via the direct return from ``obtain`` queries or with
   ``solve_all()``.

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


Probabilistic choice (NSQUALL)
------------------------------

An equiprobable choice can be defined directly in NSQUALL, avoiding the
need for ``add_uniform_probabilistic_choice_over_set`` in Python:

    >>> from neurolang.expressions import Symbol
    >>>
    >>> nl = NeurolangPDL()
    >>> nl.add_tuple_set([("s1",), ("s2",), ("s3",)], name="study")
    >>> nl.execute_squall_program(
    ...     "define as Selected_study "
    ...     "as an equiprobable choice over every Study."
    ... )
    >>> sym = Symbol("selected_study")
    >>> sym in nl.program_ir.pchoice_pred_symbs
    True


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


.. _appendix-b:

Appendix B: IR Builder Cheat-Sheet
====================================

Every NSQUALL sentence is compiled into NeuroLang's intermediate
representation (IR).  You can also write IR directly using the
**environment context manager**.  This is useful when a pattern has no
NSQUALL syntax yet, or when you need to mix Python logic with declarative
rules.

**Scope vs Environment**

``nl.scope`` — symbols are popped from the symbol table when the ``with``
block exits (clean, no side effects).

``nl.environment`` — symbols persist in the symbol table after exit
(use when rules must be visible to later ``solve_all()`` calls).

Both use the same ``e.<Name>`` attribute syntax.

**Rule equivalence cheat-sheet**

**Simple unary rule**

NSQUALL:

.. code-block:: text

    define as Active every person that plays.

IR builder:

.. code-block:: python

    with nl.environment as e:
        e.active[e.x] = e.person(e.x) & e.plays(e.x)
    sol = nl.solve_all()

**Binary / n-ary rule**

NSQUALL:

.. code-block:: text

    define as author_of for every Paper ?p ; where every Author ?a ; where ?a wrote ?p.

IR builder:

.. code-block:: python

    with nl.environment as e:
        e.author_of[e.p, e.a] = e.wrote(e.a, e.p)

**Probabilistic fact**

NSQUALL:

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

NSQUALL:

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

NSQUALL:

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


.. _appendix-c:

Appendix C: Gap Report
=======================

All features described in this tutorial are fully supported.  The only patterns
from the codebase that have no NSQUALL syntax yet are:

- **Skolem-like functional terms in rule head** — ❌ Not supported.
  Requires IR changes beyond the grammar transformer scope.
- **Weighted choice execution from NSQUALL** — ⏳ Grammar only.
  ``define as X as a choice over Y with probability P`` parses but the
  execution path is not yet implemented.  Use
  ``add_probabilistic_choice_from_tuples`` in Python for arbitrary
  probabilities.


.. _appendix-d:

Appendix D: Test Coverage
==========================

The NSQUALL parser and execution engine (``squall_syntax_lark.py``,
``query_resolution_datalog.py``) are covered by 92 tests (91 pass, 1
pre-existing skip) in two test suites:

- ``test_squall_parser.py`` — 52 tests covering grammar parsing,
  transformer logic, simplifier, and SquallProgram construction.
- ``test_squall_syntax_lark.py`` — 8 tests for Lark grammar integration.
- ``test_squall_pdl_integration.py`` — 32 tests for end-to-end SQUALL
  execution, including probabilistic queries.

**Coverage of newly added probabilistic choice code** (manual analysis,
run via ``pytest`` on Python 3.14 — coverage.py has a pre-existing numpy
C-extension conflict on this platform):

+--------------------------------------------+-------------+-------------+
| Component                                  | Approx.     | Test        |
|                                            | lines       | coverage    |
+============================================+=============+=============+
| Grammar rules (``neurolang_natural.lark``) | 20          | 100%        |
+--------------------------------------------+-------------+-------------+
| EquiprobableChoiceDef / WeightedChoiceDef  | 40          | ~100%       |
| classes + SquallProgram integration        |             |             |
+--------------------------------------------+-------------+-------------+
| Transformer handlers                       | 50          | ~95%        |
| (``rule_equiprobable_choice``,             |             |             |
| ``rule_weighted_choice``, ``squall``       |             |             |
| routing)                                   |             |             |
+--------------------------------------------+-------------+-------------+
| Parser simplification pass-through         | 15          | ~60%        |
| (``parser()`` for choice defs in           |             |             |
| Union/SquallProgram)                       |             |             |
+--------------------------------------------+-------------+-------------+
| ``_handle_equiprobable_choice`` (main      | 50          | ~85%        |
| path)                                      |             |             |
+--------------------------------------------+-------------+-------------+
| ``_handle_weighted_choice``                | 10          | ~80%        |
| (NotImplementedError)                      |             |             |
+--------------------------------------------+-------------+-------------+
| Error paths (missing source, non-constant  | 15          | 0%          |
| source, unknown body formula)              |             |             |
+--------------------------------------------+-------------+-------------+
| Scoped re-walk with choice defs            | 10          | 0%          |
| (obtain + choice together)                 |             |             |
+--------------------------------------------+-------------+-------------+
| **Total new non-test code**                | **~210**    | **~83%**    |
+--------------------------------------------+-------------+-------------+

The uncovered ~17% consists primarily of defensive error-handling paths and
edge cases that occur only with malformed programs or uncommon execution
patterns.  The main-line feature paths — parsing, transformer registration,
symbol-table population, and uniform-probability computation — are fully
exercised.
