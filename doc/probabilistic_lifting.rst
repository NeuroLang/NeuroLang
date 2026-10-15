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
2. :mod:`neurolang.probabilistic.dalvi_suciu_lift`, for unions of
   conjunctive queries with self-joins (see `Dalvi–Suciu lifting`_), extended
   to negations and comparisons which share relations with the rest of the
   query (see `Negations and comparisons sharing relations`_);
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
and existential projection only. Fink and Olteanu extend the theorem to
non-repeating queries with negation [Fink2014]_; this solver declines
queries with negations or disjunctions, which are left to the next
solver.

Dalvi–Suciu lifting
-------------------

:func:`~neurolang.probabilistic.dalvi_suciu_lift.dalvi_suciu_lift`
implements the algorithm of Dalvi and Suciu for unions of conjunctive
queries [Dalvi2012]_, extended to probabilistic choices [Suciu2011]_. Its
rules are tried in the order below, each one calling the algorithm on the
subqueries it produces. A query to which no rule applies is marked
``NonLiftable`` and declined. For unions of conjunctive queries the
algorithm is complete: a query on which it fails is #P-hard
[Dalvi2012]_.

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
   way, with ``Difference``. Fink and Olteanu show that such a query is
   liftable if and only if its positive and negated parts are
   [Fink2016]_.

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
sound, but the rules are not known to be complete: some liftable
conjunctions of this kind may still be declined and left to weighted
model counting.

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
