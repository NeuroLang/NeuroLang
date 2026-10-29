.. _inclusion-exclusion:

Inclusion–exclusion with an incomplete containment test
=======================================================

This page proves that the inclusion–exclusion rule of the lifted solver
(see :ref:`probabilistic-lifting`) computes the exact probability of a
query, although the containment test it uses to cancel terms is not
complete.

.. contents::
   :local:
   :depth: 1

Statement of the problem
------------------------

For a conjunction :math:`K_1 \land \dots \land K_n` of clauses with the same
free variables,
:func:`~neurolang.probabilistic.dalvi_suciu_lift.inclusion_exclusion_conjunction`
computes a weighted sum of the probabilities of the disjunctions
:math:`f(S) = \bigvee_{i \in S} K_i`, :math:`S` a non-empty subset of
:math:`\{1, \dots, n\}`, lifting only the disjunctions whose weight is not
0. The weights are the Möbius weights of an order :math:`\preceq` between
the disjunctions, which is built with
:func:`~neurolang.probabilistic.containment.is_contained`. This test is
sound but not complete: it looks for a homomorphism between the
disjunctions, and a negated atom :math:`\lnot R(\bar x)` is matched only by
the same negated atom, through the relation :math:`\bar R` of
``NegatedSymbol`` (``negations_as_atoms``), so that an implication that
holds because of a negation can be missed. The question is whether the
weighted sum is then still the probability of the conjunction.

The lemma
---------

**Lemma.** Let :math:`\preceq` be a partial order on the non-empty subsets
of :math:`\{1, \dots, n\}` such that

(H1) :math:`T \subseteq S` implies :math:`T \preceq S`, and

(H2) :math:`T \preceq S` implies :math:`f(T) \models f(S)`.

Let :math:`\mu` be defined by :math:`\sum_{T \preceq S} \mu(T) = 1` for every
:math:`S`, and :math:`w(S) = (-1)^{|S| + 1}`. Then for every probability
measure,

.. math::

   \sum_S \mu(S)\, P[f(S)] = \sum_S w(S)\, P[f(S)] =
   P[K_1 \land \dots \land K_n].

A partial order is reflexive and transitive: transitivity is the third
hypothesis, with (H1) and (H2).

The proof
---------

The second equality is the inclusion–exclusion principle. For the first,
call :math:`C` *closed* when :math:`T \preceq C` implies
:math:`T \subseteq C`. For closed :math:`C`,
:math:`\{T : T \preceq C\} = \{T : \emptyset \neq T \subseteq C\}` by (H1).

Closed sets are stable under non-empty intersection: if
:math:`T \preceq U \cap V`, then :math:`U \cap V \subseteq U` gives
:math:`U \cap V \preceq U` by (H1), so :math:`T \preceq U` by transitivity
and :math:`T \subseteq U`; likewise for :math:`V`. Every :math:`S` thus has
a smallest closed superset :math:`c(S)`, and for closed :math:`C`,
:math:`c(S) \subseteq C` if and only if :math:`S \subseteq C`.

:math:`c(S)` is reached by iterating
:math:`S \mapsto S \cup \bigcup \{T : T \preceq S\}`, which stays within
every closed superset of :math:`S`. Each step keeps the event: :math:`T
\preceq S` gives :math:`f(T) \models f(S)` by (H2), so
:math:`f(S \cup T) \equiv f(S)`; hence
:math:`P[f(c(S))] = P[f(S)]`.

For :math:`\varphi \in \{\mu, w\}` and closed :math:`C`, put

.. math::

   F_\varphi(C) = \sum_{S : c(S) \subseteq C} \varphi(S) =
   \sum_{\emptyset \neq S \subseteq C} \varphi(S).

For :math:`w` this is the alternating sum, :math:`F_w(C) = 1`; for
:math:`\mu` it is the recursion at :math:`C` together with the first
description of :math:`\{T : T \preceq C\}`, :math:`F_\mu(C) = 1`. Möbius
inversion on the poset of closed sets ordered by inclusion gives

.. math::

   \sum_{S : c(S) = C} \mu(S) = \sum_{S : c(S) = C} w(S)

for every closed :math:`C`. Grouping both sums of the lemma by
:math:`c(S)`, and using :math:`P[f(c(S))] = P[f(S)]`, gives the claim.
:math:`\square`

Equivalently, with a bottom element :math:`\hat 0` added,
:math:`w(S) = -\mu_B(\hat 0, S)` on the Boolean lattice of the subsets and
:math:`\mu(S) = -\mu_\preceq(\hat 0, S)` on the order :math:`\preceq`;
:math:`S \mapsto c(S)` is a closure operator on both, with the same
quotient :math:`L`, as :math:`\preceq` and :math:`\subseteq` coincide on
closed sets. Rota's closure theorem [Rota1964]_ then gives
:math:`\sum_{c(S) = C} w(S) = -\mu_L(\hat 0, C) = \sum_{c(S) = C} \mu(S)`.
Spiegel and O'Donnell [Spiegel1997]_ treat the Möbius function of
incidence algebras at length.

Completeness of the containment test is never used: a containment that is
not found only makes the closure coarser, cancelling fewer terms, and never
makes the sum wrong. A term that is not cancelled is lifted like the
others; if the algorithm cannot lift it, it is marked ``NonLiftable``, the
plan is impure, and the query is declined unless another rule lifts it.

What the implementation guarantees
----------------------------------

``_formulas_weights`` receives each non-empty set :math:`S` of indices of
clauses with the disjunctive normal form of :math:`f(S)`, which
``inclusion_exclusion_conjunction`` computes with ``convert_ucq_to_ccq``.
It builds the relation :math:`T \preceq S` when :math:`T \subseteq S`, or
when ``is_contained(f(S), f(T))`` holds, that is when the test finds that
:math:`f(T)` implies :math:`f(S)`, both directions being recorded when both
hold. It closes the relation transitively (Warshall's algorithm on the
:math:`2^n - 1` sets), and merges the sets contained in each other into
classes, each represented by one of its sets with the most clauses. The
order between the representatives is passed to ``mobius_weights``, whose
``mobius_function`` computes :math:`\mu(S) = 1 - \sum_{T \prec S} \mu(T)`.
``inclusion_exclusion_conjunction`` lifts the formula of each
representative whose weight is not 0, and ``WeightedNaturalJoin`` adds the
probabilities of the answers, multiplied by the weights.

* (H2) holds: a subset gives a disjunction of fewer clauses, which implies
  the other; ``is_contained`` is sound; and implication is transitive.
* (H1) holds: every subset is recorded, whatever ``is_contained`` finds.
* Transitivity holds by the closure.

On the sets, the relation is reflexive and transitive but need not be
antisymmetric, when two sets are in the same class. The proof does not use
antisymmetry: it uses the recursion only at closed sets, and the weights
that the code gives, the weight of a class to its representative and 0 to
its other sets, satisfy the recursion for every set, as the classes below a
set are those below its representative. Within a class the disjunctions
are equivalent by (H2), so lifting the representative alone gives the
probability of each.

For an answer :math:`\theta` of the free variables, the lemma applies to
the events :math:`f(S)\theta`. The weighted join keeps the answers in the
plans of all the terms it adds; an answer missing from the plan of a term
:math:`f(S)` has :math:`P[f(S)\theta] = 0`, hence, :math:`K_i` implying
:math:`f(S)` for :math:`i \in S`, a clause of probability 0, and
probability 0 for the conjunction.

Relation to the literature
--------------------------

Dalvi and Suciu [Dalvi2012]_ cancel the terms of inclusion–exclusion with
the Möbius function of the lattice of the disjunctions, ordered by
implication: this is the case of the lemma where :math:`\preceq` is the
full implication order, and their identity
:math:`P(Q) = -\sum_{v < \hat 1} \mu_L(v, \hat 1)\, P(d_v)` is the special
case of the lemma for it.

The "improved inclusion–exclusion identities" of Narushima [Narushima1982]_,
Dohmen [Dohmen1999]_ and Chen and Qian [Chen2018]_ also use a closure
operator on the lattice of the subsets, but they drop terms of the sum
rather than reweighting them, and are not what the implementation does.

References
----------

.. [Chen2018] Y. Chen and J. Qian. *Inclusion-exclusion by ordering-free
   cancellation*. arXiv:1801.04706, 2018.

.. [Dohmen1999] K. Dohmen. *On sums over partially ordered sets*. The
   Electronic Journal of Combinatorics, 6, R34, 1999.

.. [Narushima1982] H. Narushima. *Principle of inclusion-exclusion on
   partially ordered sets*. Discrete Mathematics, 1982.
   doi:10.1016/0012-365X(82)90221-7.

.. [Rota1964] G.-C. Rota. *On the foundations of combinatorial theory I.
   Theory of Möbius functions*. Zeitschrift für
   Wahrscheinlichkeitstheorie und verwandte Gebiete, 2, 340–368, 1964.

.. [Spiegel1997] E. Spiegel and C. J. O'Donnell. *Incidence Algebras*.
   Monographs and Textbooks in Pure and Applied Mathematics 206. Marcel
   Dekker, 1997.

[Dalvi2012]_ is listed in :ref:`probabilistic-lifting`.
