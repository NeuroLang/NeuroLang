.. _probabilistic-lifting:

Lifted probabilistic inference
==============================

This page lists the strategies NeuroLang uses to compute the probability
of a query exactly by *lifted inference*, that is by translating the query
into a plan of relational algebra operations on probabilities, without
grounding the program into a propositional formula. For each strategy it
gives the formula applied, when it applies and the reference it comes from.

.. contents::
   :local:
   :depth: 2

Solvers and their order
-----------------------

:class:`~neurolang.frontend.NeurolangPDL` tries its solvers in order, and
uses the first one which accepts the query:

1. :mod:`neurolang.probabilistic.small_dichotomy_theorem_based_solver`, for
   hierarchical queries without self-joins (see `Hierarchical queries
   without self-joins`_);
2. :mod:`neurolang.probabilistic.dalvi_suciu_lift`, which first plans
   non-repeating queries with negation (see `Non-repeating queries with
   negation`_), and otherwise lifts unions of conjunctive queries with
   self-joins (see `Dalvi–Suciu lifting`_), extended to negations and
   comparisons which share relations with the rest of the query (see
   `Negations and comparisons sharing relations`_);
3. :mod:`neurolang.probabilistic.weighted_model_counting`, which is not
   lifted: it compiles the provenance of the query into a sentential
   decision diagram (SDD) [Darwiche2011]_ with pysdd and computes its
   weighted model count [Chavira2008]_. It answers every query of the
   language but can be exponential in the size of the data.

A lifted solver declines a query by raising an
``UnsupportedSolverError``, such as ``NonLiftableException``, and the next
one is then tried. Lifted plans are
computed on the provenance semiring of probabilities
(:mod:`neurolang.relational_algebra_provenance`), whose operations are
listed in `Operations of the lifted plans`_. Conditional queries (``//``,
``given``, ``conditioned to``) are answered as the ratio of two such
queries, the joint and the condition.

The probabilistic semantics are those of CP-Logic [Vennekens2009]_, and the
overall approach is described in [Zanitti2023]_. Probabilistic facts are
*tuple-independent* relations; probabilistic choices are relations with a
single key-less block, where exactly one tuple is true, a special case of
*block-independent disjoint* (BID) relations [Suciu2011]_.

Operations of the lifted plans
------------------------------

Each operation computes the probability :math:`p` of each tuple of its
result from the probabilities of its inputs.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Operation
     - Probability
     - Used for
   * - ``NaturalJoin`` (:math:`\bowtie`)
     - :math:`p_1 \, p_2`
     - independent conjunction
   * - ``Union``
     - :math:`1 - (1 - p_1)(1 - p_2)`
     - independent disjunction
   * - ``Difference``
     - :math:`p_1 \, (1 - p_2)`
     - independent negation, :math:`A \land \lnot B`
   * - ``Subtraction`` (:math:`-_P`)
     - :math:`p_1 - p_2`
     - :math:`P[A] - P[A \land B]`, for dependent :math:`A` and :math:`B`
   * - ``IndependentProjection`` (ind-∃-π)
     - :math:`1 - \prod_i (1 - p_i)`
     - existential quantifier on a separator variable
   * - ``IndependentProjectionUniversal`` (ind-∀-π)
     - :math:`\prod_i p_i`
     - universal quantifier on a separator variable
   * - ``DisjointProjection`` (disj-π)
     - :math:`\sum_i p_i`
     - projection of mutually exclusive tuples
   * - ``WeightedNaturalJoin`` (Σ)
     - :math:`\sum_j w_j \, p_j`
     - inclusion–exclusion with Möbius weights
   * - ``Support``
     - :math:`1`
     - the tuples of a plan, to give values to its variables

Preparing the query
-------------------

Before lifting, :func:`~neurolang.probabilistic.dalvi_suciu_lift.solve_succ_query`
rewrites the query as follows.

Flattening
   The rules the query depends on are unfolded into its body, giving a
   union of conjunctive queries in disjunctive normal form.

Absorption in negations
   Conjuncts asserted outside a negation are removed from it, by the
   absorption law

   .. math::

      A \land \lnot(A \land B) \equiv A \land \lnot B, \qquad
      A \land \lnot\exists y\,(A \land B) \equiv A \land \lnot\exists y\, B
      \quad (y \text{ not free in } A).

   The law holds in every interpretation, hence in every possible world.
   It removes, for instance, a probabilistic atom repeated within a
   negation, which can make the query non-repeating, the class of queries
   with negation for which Fink and Olteanu characterise liftability
   [Fink2016]_. A negated conjunction all of whose conjuncts are asserted
   is a contradiction and is left unchanged
   (:class:`~neurolang.logic.transformations.AbsorbAssertedConjunctsInNegations`).

Deterministic negations
   A negated conjunct :math:`\lnot N(\bar x)` whose atoms are all
   deterministic relations or comparisons is the same relation in every
   possible world. With :math:`G(\bar x)` the deterministic conjuncts of
   the query whose variables are among :math:`\bar x`, it is replaced by a
   fresh deterministic relation computed with relational algebra,

   .. math::

      D(\bar x) := G(\bar x) \land \lnot N(\bar x), \qquad
      G \land \lnot N \equiv G \land D,

   :math:`G` being conjoined within the existentials of :math:`N` so that
   they are range-restricted, as in
   :math:`\lnot\exists t_1\,(T(t_1) \land M(s, t_1) \land t_1 \neq t)`.
   The negation is left unchanged when no non-empty :math:`G` binds all of
   :math:`\bar x`, for instance when :math:`N` is closed
   (:mod:`neurolang.probabilistic.deterministic_subqueries`).

Probabilistic choices
   Two atoms of the same probabilistic choice relation are true together
   only if they are the same tuple, as exactly one tuple of the choice is
   true. Their arguments are therefore made equal
   (:func:`~neurolang.probabilistic.expression_processing.lift_optimization_for_choice_predicates`).

Shattering constants
   An atom with a constant, such as :math:`R(x, a)`, is replaced by an atom
   of a new relation holding the tuples of :math:`R` selected on that
   constant, so that the remaining atoms of :math:`R` refer to disjoint sets
   of tuples [Dalvi2012]_ (:mod:`neurolang.probabilistic.shattering`).

Ranking
   Atoms with a repeated variable, such as :math:`R(x, x)`, are replaced by
   atoms of a new relation holding the selected tuples, and the query must
   then be *ranked*: there is an order on the variables such that the
   arguments of every atom follow it [Dalvi2012]_
   (:mod:`neurolang.probabilistic.ranking`). Queries which are not ranked are
   declined.

Checks
   The query must have no built-in function, except comparisons within
   existential conjuncts (see `Negations and comparisons sharing
   relations`_). Unless the ``check_unate`` option of
   ``[PROBABILISTIC_SOLVER]`` is turned off (see :doc:`performance`), it
   must also use a single kind of quantifier and be *unate*, each relation
   occurring only positively or only negatively.
   The negations and comparisons of `Negations and comparisons sharing
   relations`_ are exempt from this check, as their rules check each
   query they lift.

Hierarchical queries without self-joins
---------------------------------------

For a conjunctive query in which no relation occurs twice, the small
dichotomy theorem applies [Dalvi2007]_ [Fink2014]_: writing :math:`at(x)`
for the atoms in which variable :math:`x` occurs, the query can be computed
in polynomial time if it is *hierarchical*, that is if for any two variables
:math:`x` and :math:`y`,

.. math::

   at(x) \subseteq at(y) \quad\text{or}\quad
   at(x) \supseteq at(y) \quad\text{or}\quad
   at(x) \cap at(y) = \emptyset,

and is #P-hard otherwise. The plan is built from the independent join
and existential projection only. This solver declines queries with
negations or disjunctions, which are left to the next solver; Fink and
Olteanu's extension of the theorem to negation is described in the next
section.

Non-repeating queries with negation
-----------------------------------

A query is *non-repeating* when none of its probabilistic relations
occurs twice in it. Fink and Olteanu show that a non-repeating query
without union, built from joins, projections and negations, on
tuple-independent relations, can be computed in polynomial time if it is
hierarchical, and is #P-hard otherwise [Fink2014]_. The planner below
also accepts disjunctions, comparisons, constants, deterministic relations
and probabilistic choices, which this theorem does not cover: its plans
are proved correct in `Correctness of the plans`_, and it is not claimed
to plan every tractable query. Before Dalvi–Suciu lifting,
:func:`~neurolang.probabilistic.non_repeating_lift.non_repeating_plan`
plans the non-repeating queries with a negation in this way. The negations
can be nested and can quantify variables, as in
:math:`\exists x\, R(x) \land \lnot\exists y\, (S(x, y) \land \lnot U(x,
y))`, whose quantifiers are mixed once the negations are moved to the
atoms, which the checks of Dalvi–Suciu lifting decline.

The query is first rewritten with two equivalences, which hold in every
possible world and keep it non-repeating:

* equalities with a quantified variable are eliminated in every
  subformula, :math:`\exists v\, (v = e \land \varphi) \equiv
  \varphi[e/v]` for a variable or a constant :math:`e`, as in
  :math:`\lnot\exists u\, (W(s, u) \land u = t) \equiv \lnot W(s, t)`;
* a negated disjunction is the conjunction of the negated disjuncts,
  :math:`\lnot(A \lor B) \equiv \lnot A \land \lnot B`, the disjuncts
  possibly having different free variables, as in
  :math:`M(s, t) \land \lnot(K(s) \lor G(s, t))`.

As the subformulas of a non-repeating query have different relations, they
are independent, and the plan is built with the independent operations
only:

* a conjunction is the join of its positive conjuncts, filtered by its
  comparisons, from which its negated conjuncts are removed with
  ``Difference``, :math:`P[A \land \lnot B] = P[A]\,(1 - P[B])`;
* a disjunction of formulas with the same free variables is their
  independent union;
* :math:`\exists \bar v\, (\varphi_1 \lor \varphi_2)` is planned as
  :math:`\exists \bar v\, \varphi_1 \lor \exists \bar v\, \varphi_2`,
  each disjunct keeping the quantifiers of its variables. For instance, a
  negated predicate defined by two rules, :math:`N(s) \leftarrow F(s)` and
  :math:`N(s) \leftarrow M(s, t)`, flattens to
  :math:`\lnot\exists t\, (F(s) \lor M(s, t))`, which is planned as
  :math:`\lnot F(s) \land \lnot\exists t\, M(s, t)`;
* :math:`\exists \bar v\, \varphi` is first split into the groups of
  conjuncts of :math:`\varphi` connected by the variables of
  :math:`\bar v`, the other conjuncts being moved out of the quantifier.
  Each group is then projected on one of its variables :math:`v` that
  occurs in every probabilistic atom of the group, including those within
  its negations: the events :math:`\varphi(a)` for different values
  :math:`a` concern different tuples, and the projection is independent,
  :math:`1 - \prod_a (1 - P[\varphi(a)])`. When a probabilistic choice
  conjunct of the group has :math:`v`, the events are mutually exclusive
  instead, and the projection is disjoint, :math:`\sum_a P[\varphi(a)]`.
  If no variable of the group occurs in all of its probabilistic atoms, as
  in the #P-hard
  :math:`\exists x\, R(x) \land \lnot\exists y\, (S(x, y) \land T(y))`,
  or if another condition fails, such as a comparison or a negation whose
  variables are not bound by the rest of its conjunction, or a choice atom
  with the variable that is not a conjunct of the group, no plan is given
  and the query is left to Dalvi–Suciu lifting.

Deterministic relations do not count as repetitions, and do not need to
have the projected variable.

In Datalog, the rules of a predicate share their head variables, so the
negation of a predicate defined by several rules always has disjuncts with
the same free variables once the quantifiers are pushed into them;
disjuncts with different free variables only arise in queries built with
the Python API.

Dalvi–Suciu lifting
-------------------

:func:`~neurolang.probabilistic.dalvi_suciu_lift.dalvi_suciu_lift`
implements the algorithm of Dalvi and Suciu for unions of conjunctive
queries [Dalvi2012]_, extended to probabilistic choices [Suciu2011]_. Its
rules are tried in the order below, each one calling the algorithm on the
subqueries it produces. A query to which no rule applies is marked
``NonLiftable`` and declined. Dalvi and Suciu prove their algorithm
complete for unions of conjunctive queries on tuple-independent
databases: a query on which it fails is #P-hard [Dalvi2012]_. The
implementation restricts some of its rules, inclusion–exclusion for
instance, so a query it declines is not necessarily #P-hard.

Base cases
   A single atom is read from its relation; a subquery whose atoms are all
   deterministic is computed by ordinary relational algebra, with
   probability 1.

Minimisation
   Redundant atoms and disjuncts are removed with the containment test of
   :mod:`neurolang.probabilistic.containment`, and the base cases are tried
   again.

Independent conjunction
   With the query in conjunctive normal form, if its conjuncts split into
   components with no probabilistic relation in common, they are
   independent:

   .. math::

      P[Q_1 \land Q_2] = P[Q_1]\,P[Q_2], \qquad
      P[Q_1 \land \lnot Q_2] = P[Q_1]\,(1 - P[Q_2]).

   Negated components sharing no relation with the rest are planned this
   way, with ``Difference``: the query is liftable when its positive and
   negated parts are (see `Correctness of the plans`_). Fink and Olteanu
   study the converse, for queries with negation without repeated
   relations [Fink2014]_ [Fink2016]_.

Independent disjunction
   With the query in disjunctive normal form, if its disjuncts split into
   components with no probabilistic relation in common:

   .. math::

      P[Q_1 \lor Q_2] = 1 - (1 - P[Q_1])(1 - P[Q_2]).

Separator variable
   A variable :math:`z` is a separator variable of :math:`\exists z\,Q(z)`
   if it occurs in every atom and, for each relation, in the same position
   of all of its atoms. The events :math:`Q(a)` for different values
   :math:`a` then concern different tuples and are independent:

   .. math::

      P[\exists z\, Q(z)] = 1 - \prod_a (1 - P[Q(a)]).

   The dual rule, for a universally quantified separator variable, gives
   :math:`\prod_a P[Q(a)]` [Suciu2020]_.

Disjoint projection
   If an atom of a probabilistic choice, or of a tuple-independent relation
   with constants in all its positions, occurs in a conjunctive query, the
   values of the projected variables give mutually exclusive events whose
   probabilities add up [Suciu2011]_ (section 4.3.1). For a union of
   conjunctive queries, the disjuncts with such an atom are separated
   with :math:`P[A \lor B] = P[A] + P[B] - P[A \land B]`.

Inclusion–exclusion
   For a conjunction :math:`Q_1 \land \dots \land Q_n` in conjunctive normal
   form,

   .. math::

      P\Big[\bigwedge_{i} Q_i\Big] =
      \sum_{\emptyset \neq S \subseteq \{1..n\}}
      (-1)^{|S|+1}\, P\Big[\bigvee_{i \in S} Q_i\Big].

   Terms whose disjunctions are equivalent are merged and their
   coefficients added, using the Möbius function of the lattice of the
   disjunctions ordered by containment [Dalvi2012]_. This cancellation
   removes the terms that may be #P-hard while their sum is not. The rule
   is applied only when all the conjuncts have the same free variables,
   for which the containment test is sound.

Negations and comparisons sharing relations
--------------------------------------------

:func:`~neurolang.probabilistic.dalvi_suciu_lift.lift_with_dependent_negations`
extends the algorithm to conjunctions :math:`A \land \lnot N` in which the
negated conjunct :math:`N` shares probabilistic relations with :math:`A`,
and to existential conjuncts with comparisons such as :math:`t \neq x`.
Such conjunctions fall outside unions of conjunctive queries, so the
completeness result above does not apply to them.
The rules below are tried in turn when the relations shared are
tuple-independent; a conjunction sharing a probabilistic choice is
declined. Every rule checks each query it lifts, and the lifted plans are
sound (see `Correctness of the plans`_), but the rules are not known to be
complete: some liftable conjunctions of this kind may still be declined
and left to weighted model counting. These rules are NeuroLang's own: the
references they cite give the techniques they build on, not their
correctness, which is proved below.

Comparisons in an existential conjunct
   For :math:`A \land \exists t\,(\psi(t) \land C)`, or its negation, where
   :math:`C` is a conjunction of comparisons between :math:`t`, the free
   variables and constants, e.g.
   :math:`P(x, y) \land \lnot\exists w\,(P(w, y) \land w \neq x)`. Following
   Olteanu and Huang [Olteanu2009]_, with :math:`L` the plan of :math:`A`,
   the event :math:`E = \exists t\,(\psi(t) \land C)` is computed as

   .. math::

      E = \text{ind-}\pi\;\sigma_C\big(\mathrm{lift}(\psi(t)) \bowtie
      \mathrm{support}(L)\big)

   and the plan is :math:`L \bowtie E`, or :math:`\mathrm{Difference}(L, E)`
   when negated. It applies when :math:`t` is a separator variable of
   :math:`\psi`, and when each probabilistic atom of :math:`\psi(t)` has
   :math:`t` in a position where the atoms of the same relation in
   :math:`A` have a term that :math:`C` makes different from :math:`t`, so
   that :math:`E` and :math:`A` concern different tuples.

Free-variable shattering
   For :math:`A \land \lnot\exists t\,\psi(t)` where the shared atoms have a
   free variable :math:`x` of :math:`A` in the position of :math:`t`, such
   as :math:`R(x) \land \lnot\exists t\,(R(t) \land S(x, t))`. Splitting the
   existential on :math:`t = x` and :math:`t \neq x`, as shattering does
   [Dalvi2012]_,

   .. math::

      \exists t\,\psi(t) \equiv \psi(x) \lor \exists t\,(t \neq x \land
      \psi(t)),

   gives the plan, with :math:`L = \mathrm{lift}(A \land \lnot\psi(x))`,

   .. math::

      \mathrm{Difference}\Big(L,\ \text{ind-}\pi\;\sigma_{t \neq x}
      \big(\mathrm{lift}(\psi(t)) \bowtie \mathrm{support}(L)\big)\Big).

   It applies when :math:`t` is a separator variable of :math:`\psi`, and
   when, for :math:`t \neq x`, each probabilistic atom of :math:`\psi(t)` is
   a different tuple from the atoms of the same relation in :math:`A` and
   :math:`\psi(x)`, the second part being then independent of :math:`L`.

Negation by subtraction
   For any :math:`A \land \lnot N`, as :math:`A \land N` implies :math:`A`,

   .. math::

      P[A \land \lnot N] = P[A] - P[A \land N],

   computed with ``Subtraction`` when both :math:`A` and :math:`A \land N`
   are liftable.

Existential shattering
   For a positive existential conjunct :math:`A \land \exists t\,\psi(t)`
   sharing relations with :math:`A`, which Dalvi–Suciu lifting declines,

   .. math::

      P[A \land \exists t\,\psi(t)] = P[A] - P[A \land \lnot\exists t\,
      \psi(t)],

   the latter being planned by free-variable shattering.

Correctness of the plans
------------------------

A rule is *liftable* when its plan computes the exact probability of the
query (soundness) in time polynomial in the size of the data, the query
being fixed (data complexity). This section proves both for the rules of
this page which are not covered by a cited theorem. Completeness, that
every tractable query is lifted, is not claimed for them.

The probabilistic model
   A tuple-independent relation has a Boolean random variable per tuple,
   these variables being independent. A probabilistic choice has one
   random variable whose value is the single tuple of the relation which
   is true. The variables of different relations are independent, and a
   possible world is a value for all of them. Deterministic relations are
   the same in every world.

   For a formula :math:`\varphi` and a tuple :math:`\theta` of values for
   its free variables, the event :math:`\varphi\theta` is the set of worlds
   in which it holds. It is determined by the random variables of the
   tuples :math:`R(\bar a)` such that an atom of :math:`\varphi\theta` can be
   :math:`R(\bar a)` for some values of its quantified variables: these are
   its *tuples*. A plan computes, for every :math:`\theta` in its support,
   :math:`P[\varphi\theta]`, and :math:`P[\varphi\theta] = 0` outside it.

   The rules identify the variables by their names, so the quantified
   variables are first renamed to fresh ones
   (:func:`~neurolang.logic.transformations.standardise_apart`, applied by
   :func:`~neurolang.probabilistic.dalvi_suciu_lift.solve_succ_query`): no
   substitution is captured and no atom has a variable on which it does
   not depend.

Lemma 1 (independence)
   If no two events of a family :math:`E_1, \dots, E_n` have a tuple of a
   tuple-independent relation in common, nor both have tuples of the same
   probabilistic choice, the events are mutually independent: they are
   determined by pairwise disjoint sets of independent random variables.
   Formulas without a common probabilistic relation satisfy this.

Lemma 2 (exclusivity)
   If :math:`E_1` implies that the tuple :math:`C(\bar a)` of a
   probabilistic choice :math:`C` is true, and :math:`E_2` that the tuple
   :math:`C(\bar b)` is, with :math:`\bar a \neq \bar b`, then
   :math:`E_1 \land E_2` is impossible, as a single tuple of :math:`C` is
   true.

Polynomial time
   Each operation of the plans (`Operations of the lifted plans`_) is
   computed in time polynomial in the size of its inputs: joins,
   selections, left joins for ``Difference`` and ``Subtraction``, and
   grouped products or sums for the projections. A plan is built from the
   query alone, so its size does not depend on the data, and it is
   computed in polynomial time if the construction terminates.

   The rules of `Negations and comparisons sharing relations`_ call each
   other through the queries they lift. Each call is on a query that is
   smaller for the lexicographic order on (number of quantifiers, number
   of negations, size): negation by subtraction lifts :math:`A`, a part of
   :math:`A \land \lnot N`, and :math:`A \land N`, which has one negation
   less; free-variable shattering lifts :math:`A \land \lnot\psi(x)` and
   :math:`\psi(t)`, with one quantifier less than
   :math:`A \land \lnot\exists t\, \psi(t)`; existential shattering lifts
   :math:`A` and the queries of free-variable shattering; the comparison
   rule lifts :math:`A` and :math:`\psi(t)`, parts of the query. The
   Dalvi–Suciu rules, whose recursion terminates [Dalvi2012]_, do not call
   the rules of this section. The order being well founded, the
   construction terminates.

Absorption in negations
   In every world, for every :math:`\theta`: if :math:`A\theta` holds,
   :math:`\lnot(A \land B)\theta` and :math:`\lnot B\theta` agree;
   otherwise both sides of :math:`A \land \lnot(A \land B) \equiv A \land
   \lnot B` are false. When :math:`y` is not free in :math:`A`,
   :math:`\exists y\, (A \land B) \equiv A \land \exists y\, B`, which gives
   the rule through existentials. A conjunct is removed from a negation
   only if it is one of the conjuncts asserted outside it and no variable
   quantified between the negation and it is free in it
   (``_without_asserted_conjuncts``), so that its variables are the same
   inside and outside. A negation whose conjuncts would all be removed is
   left unchanged. The rewriting is done once, before lifting.

Deterministic negations
   The atoms of :math:`N` are deterministic relations or comparisons
   (``_is_deterministic``, the extensional database of a
   ``CPLogicProgram`` excluding its probabilistic relations), so
   :math:`N`, and :math:`D = G \land \lnot N` computed by relational
   algebra, are the same in every world. :math:`G` being asserted by the
   conjunction, :math:`G \land \lnot N \equiv G \land D` in every world.
   :math:`G` binds every free variable of :math:`N` (``_can_materialise``),
   so :math:`D` is finite. :math:`G` is conjoined within the existentials
   of :math:`N`, whose variables are not free in :math:`G`:
   :math:`\lnot\exists t\, N \equiv \lnot\exists t\, (G \land N)` given
   :math:`G`, by absorption. :math:`D` is computed once, in time
   polynomial in the deterministic relations.

Independent negation
   For :math:`A \land \lnot N` where :math:`N` shares no probabilistic
   relation with :math:`A` and its free variables are among those of
   :math:`A`, which the range restriction of the rules, checked when they
   are declared (:func:`~neurolang.datalog.range_restriction.check_range_restriction`),
   ensures, Lemma 1 gives, for every :math:`\theta`,
   :math:`P[(A \land \lnot N)\theta] = P[A\theta]\,(1 - P[N\theta])`, which
   ``Difference`` computes, a tuple of :math:`A` absent from the plan of
   :math:`N` having :math:`P[N\theta] = 0`. The query is thus liftable when
   :math:`A` and :math:`N` are.

Negation by subtraction
   :math:`A\theta` is the disjoint union of :math:`(A \land N)\theta` and
   :math:`(A \land \lnot N)\theta`, so
   :math:`P[(A \land \lnot N)\theta] = P[A\theta] - P[(A \land N)\theta]`
   for any events. The free variables of :math:`N` being among those of
   :math:`A` (``_split_dependent_conjunct``), the plan of
   :math:`A \land N` has the columns of the plan of :math:`A`, and
   ``Subtraction`` computes this difference tuple by tuple. Existential
   shattering is the same identity, with :math:`\exists t\, \psi(t)` for
   :math:`N`.

Free-variable shattering
   Splitting on :math:`t = x`,
   :math:`\exists t\, \psi(t) \equiv \psi(x) \lor E` with
   :math:`E = \exists t\, (t \neq x \land \psi(t))`, so
   :math:`A \land \lnot\exists t\, \psi(t) \equiv L \land \lnot E` with
   :math:`L = A \land \lnot\psi(x)`. The rule checks that

   * :math:`t` is a separator variable of :math:`\psi`
     (``_is_separator_variable``): every probabilistic atom of
     :math:`\psi` has :math:`t`, and any two atoms of the same relation
     have :math:`t` at a common position (``find_separator_variables``).
     A tuple of :math:`\psi(a)` through one of them and a tuple of
     :math:`\psi(b)` through the other have :math:`a` and :math:`b` at
     that position, so they differ when :math:`a \neq b`. A separator
     variable cannot be a variable of a
     probabilistic choice atom (``extract_probabilistic_root_variables``),
     so :math:`\psi` has no such atom, and the relations shared with
     :math:`A` are tuple-independent (``_shares_a_choice_relation``). By
     Lemma 1, the events :math:`\psi(a)\theta` being mutually
     independent,
     :math:`P[E\theta] = 1 - \prod_{a
     \neq x\theta} (1 - P[\psi(a)\theta])`, which the independent
     projection of the plan computes, ``support(L)`` giving the value of
     :math:`x`;
   * for :math:`t \neq x`, each probabilistic atom of :math:`\psi(t)` has
     :math:`t` at a position where each atom of the same relation in
     :math:`A` and :math:`\psi(x)` has :math:`x`
     (``_are_different_tuples_when_different``): the tuples of
     :math:`E\theta` are not tuples of :math:`L\theta`, and by Lemma 1
     :math:`P[(L \land \lnot E)\theta] = P[L\theta]\,(1 - P[E\theta])`,
     which ``Difference`` computes.

Comparisons in an existential conjunct
   With :math:`E = \exists t\, (\psi(t) \land C)`, the rule checks that
   :math:`t` is a separator variable of :math:`\psi`, which as above gives
   :math:`P[E\theta] = 1 - \prod_{a : C\theta(a)} (1 - P[\psi(a)\theta])`,
   computed by selecting :math:`C` before the independent projection; and
   that each probabilistic atom of :math:`\psi(t)` has :math:`t` at a
   position where each atom of the same relation in :math:`A` has a term
   :math:`u` with :math:`t \neq u`, :math:`t < u` or :math:`t > u` in
   :math:`C` (``_terms_different_from``, ``STRICT_COMPARISONS``). When
   :math:`C\theta(a)` holds, the tuples of :math:`\psi(a)\theta` are thus
   not tuples of :math:`A\theta`, and by Lemma 1
   :math:`P[(A \land E)\theta] = P[A\theta]\,P[E\theta]` and
   :math:`P[(A \land \lnot E)\theta] = P[A\theta]\,(1 - P[E\theta])`, which
   the join and ``Difference`` compute. Olteanu and Huang
   [Olteanu2009]_ compute conjunctive queries with inequalities without
   self-joins on tuple-independent databases; this rule uses the
   inequalities instead to show that the tuples of a relation occurring
   in both :math:`A` and :math:`\psi` differ, which their result does not
   cover.

Non-repeating queries
   :func:`~neurolang.probabilistic.non_repeating_lift.non_repeating_plan`
   applies only if no probabilistic relation, fact or choice, occurs
   twice in the query (``_is_non_repeating``). Its rewrites add no atom,
   so this holds for every subformula planned, and by Lemma 1 two
   subformulas without a common atom are independent. Then:

   * *rewrites*, by ``NormaliseNonRepeatingQuery``:
     :math:`\exists v\, (v = e \land \varphi) \equiv \varphi[e/v]` when no
     variable of :math:`e` is captured, which
     ``UnifyExistentialVariableEqualitiesMixin`` checks with
     ``ExtractBoundVariables`` after the query is standardised apart;
     :math:`\lnot(A \lor B) \equiv \lnot A \land \lnot B`
     (``MoveNegationsIntoDisjunctionsMixin``);
     :math:`\exists v\, (A \lor B) \equiv \exists v\, A \lor \exists v\, B`
     (``PushExistentialsDownDisjunctionsMixin``); and, by the planner,
     :math:`\exists \bar v\, (A \land B) \equiv \exists \bar v_A\, A
     \land \exists \bar v_B\, B` when :math:`A` and :math:`B` share no
     variable of :math:`\bar v` (``_connected_groups``). These hold in
     every world;
   * *conjunction*: the join of independent positive conjuncts is the
     product of their probabilities; a comparison, deterministic, is a
     selection; for a negation whose free variables are bound by the
     positive conjuncts (checked in ``_conjunction_plan``),
     ``Difference`` computes :math:`P[A\theta]\,(1 - P[N\theta])` as for an
     independent negation above. Atoms with constants or repeated
     variables are selected on named columns (``NonRepeatingPlanner.atom``);
   * *disjunction*: its disjuncts have the same free variables (checked in
     ``NonRepeatingPlanner.disjunction``) and are independent, so
     :math:`P[(A \lor B)\theta] = 1 - (1 - P[A\theta])(1 - P[B\theta])`;
   * *projection*: the variable :math:`v` occurs in every probabilistic
     atom of the group :math:`\varphi`, including those within its
     negations and existentials (``_root_variable``); the query being
     standardised apart, these atoms depend on :math:`v`. Each relation
     occurring once, its atom has :math:`v` at some position :math:`i`, so
     for :math:`a \neq b` the tuples of :math:`\varphi(a)\theta` and
     :math:`\varphi(b)\theta` of that relation differ at :math:`i`. If no
     probabilistic choice atom has :math:`v` (``_projection``), there is
     no choice atom in :math:`\varphi`, and by Lemma 1 the events
     :math:`\varphi(a)\theta` are mutually independent:
     :math:`P[\exists v\, \varphi\theta] = 1 - \prod_a (1 -
     P[\varphi(a)\theta])`. If a conjunct of :math:`\varphi` is an atom of
     a choice :math:`C` with :math:`v`, :math:`\varphi(a)\theta` and
     :math:`\varphi(b)\theta` imply different tuples of :math:`C`, and by
     Lemma 2 they are mutually exclusive:
     :math:`P[\exists v\, \varphi\theta] = \sum_a P[\varphi(a)\theta]`.
     Otherwise no plan is given;
   * deterministic relations are the same in every world, so they neither
     count as repetitions nor need to have :math:`v`.

   Each recursive call is on a smaller formula: a part of a conjunction,
   negation or disjunction, or a group with one quantifier less. Pushing
   :math:`\exists \bar v` into a disjunction copies the quantifiers into
   every disjunct, but the disjuncts are planned one by one, and each
   :math:`\exists \bar v\, \varphi_i` is smaller than
   :math:`\exists \bar v\, (\varphi_1 \lor \dots \lor \varphi_n)`. The
   rewrites remove equalities and negations of disjunctions without adding
   atoms. The plan is thus built from the query alone, and computed in
   polynomial time in the data.

Dalvi–Suciu rules
   The rules of `Dalvi–Suciu lifting`_ are those of Dalvi and Suciu
   [Dalvi2012]_, whose proofs apply to unions of conjunctive queries on
   tuple-independent databases, and of Suciu et al. [Suciu2011]_ (section
   4.3.1) for the disjoint projection on probabilistic choices.

The tests of the rules of `Negations and comparisons sharing relations`_
(``neurolang/probabilistic/tests/test_lifted_negation.py``) also compare
their plans with an enumeration of the possible worlds of small
databases.

References
----------

.. [Chavira2008] M. Chavira and A. Darwiche. *On probabilistic inference by
   weighted model counting*. Artificial Intelligence, 172(6–7), 772–799,
   2008.

.. [Dalvi2007] N. Dalvi and D. Suciu. *Efficient query evaluation on
   probabilistic databases*. The VLDB Journal, 16(4), 523–544, 2007.

.. [Dalvi2012] N. Dalvi and D. Suciu. *The dichotomy of probabilistic
   inference for unions of conjunctive queries*. Journal of the ACM,
   59(6), 1–87, 2012.

.. [Darwiche2011] A. Darwiche. *SDD: A new canonical representation of
   propositional knowledge bases*. In Proceedings of the 22nd International
   Joint Conference on Artificial Intelligence (IJCAI), 819–826, 2011.

.. [Fink2014] R. Fink and D. Olteanu. *A dichotomy for non-repeating
   queries with negation in probabilistic databases*. In Proceedings of
   the 33rd ACM SIGMOD-SIGACT-SIGART Symposium on Principles of Database
   Systems (PODS), 144–155, 2014.

.. [Fink2016] R. Fink and D. Olteanu. *Dichotomies for queries with
   negation in probabilistic databases*. ACM Transactions on Database
   Systems, 41(1), 1–47, 2016.

.. [Olteanu2009] D. Olteanu and J. Huang. *Secondary-storage confidence
   computation for conjunctive queries with inequalities*. In Proceedings
   of the 2009 ACM SIGMOD International Conference on Management of Data,
   389–402, 2009.

.. [Suciu2011] D. Suciu, D. Olteanu, C. Ré and C. Koch. *Probabilistic
   Databases*. Synthesis Lectures on Data Management. Morgan & Claypool,
   2011.

.. [Suciu2020] D. Suciu. *Probabilistic databases for all*. In Proceedings
   of the 39th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database
   Systems (PODS), 19–31, 2020.

.. [Vennekens2009] J. Vennekens, M. Denecker and M. Bruynooghe. *CP-logic:
   A language of causal probabilistic events and its relation to logic
   programming*. Theory and Practice of Logic Programming, 9(3), 245–308,
   2009.

[Zanitti2023]_ is listed in :doc:`tutorial_squall`.
