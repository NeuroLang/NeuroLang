"""
Lifted plans for non-repeating queries with negation.

A query is non-repeating when none of its probabilistic relations occurs
twice in it. Fink and Olteanu show that such a query, possibly with
negations, can be computed in polynomial time if it is hierarchical, and
is #P-hard otherwise [1]_. As the relations of the subformulas of a
non-repeating query are different, the subformulas are independent and
the plan is built with the independent operations only, once each
quantified variable equal to a term is replaced by it,
∃v (v = e ∧ φ) ≡ φ[e/v], and each negated disjunction is split,
¬(A ∨ B) ≡ ¬A ∧ ¬B:

* a conjunction is the join of its positive conjuncts, from which its
  negated conjuncts are removed with ``Difference``,
  P[A ∧ ¬B] = P[A] (1 - P[B]);
* a disjunction of formulas with the same free variables is their
  independent union;
* ``∃v (φ ∨ ψ)`` is ``∃v φ ∨ ∃v ψ``;
* ``∃v φ(v)``, after pushing the quantifier down to the conjuncts in
  which `v` is free, is an independent projection on a variable `v`
  that occurs in every probabilistic atom of φ: the events φ(a) for
  different values `a` then concern different tuples. If `v` occurs in
  a probabilistic choice conjunct of φ, the events φ(a) are mutually
  exclusive instead and the projection is disjoint. If there is no such
  variable the query is not hierarchical, and no plan is given.

.. [1] Fink, R. & Olteanu, D. A dichotomy for non-repeating queries with
   negation in probabilistic databases. In Proceedings of the 33rd ACM
   SIGMOD-SIGACT-SIGART Symposium on Principles of Database Systems
   (PODS), 144–155 (2014).
"""
from collections import Counter
from functools import reduce

from .. import relational_algebra_provenance as rap
from ..datalog.expression_processing import (
    UnifyExistentialVariableEqualitiesMixin
)
from ..datalog.translate_to_named_ra import TranslateToNamedRA
from ..expression_pattern_matching import NeuroLangPatternMatchingNoMatch
from ..expression_walker import PatternWalker, add_match, expression_iterator
from ..expressions import FunctionApplication, Symbol
from ..logic import Conjunction, Disjunction, ExistentialPredicate, Negation
from ..logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables
)
from ..logic.transformations import (
    LogicExpressionWalker,
    CollapseConjunctionsMixin,
    MoveNegationsIntoDisjunctionsMixin,
    PushExistentialsDownDisjunctionsMixin,
    RemoveTrivialOperationsMixin,
    standardise_apart
)
from ..relational_algebra import str2columnstr_constant
from .expression_processing import is_comparison
from .shattering import EQ
from .probabilistic_ra_utils import (
    is_atom_a_deterministic_relation,
    is_atom_a_probabilistic_choice_relation
)


def non_repeating_plan(formula, symbol_table):
    """Lifted plan of a non-repeating formula with negations.

    Parameters
    ----------
    formula : LogicExpression
        formula built from atoms, comparisons, conjunctions,
        disjunctions, negations and existentials.
    symbol_table : Mapping of Symbol to Expression
        symbol table for the relational symbols.

    Returns
    -------
    Optional[RelationalAlgebraOperation]
        plan whose columns are the free variables of `formula`, or None
        if `formula` has no negation, is repeating, or is not planned by
        `NonRepeatingPlanner`.
    """
    if not (
        any(isinstance(e, Negation) for _, e in expression_iterator(formula))
        and _is_non_repeating(formula, symbol_table)
    ):
        return None
    normalised = NormaliseNonRepeatingQuery().walk(standardise_apart(formula))
    try:
        return NonRepeatingPlanner(symbol_table).walk(normalised)
    except NeuroLangPatternMatchingNoMatch:
        return None


def columns_comparison(comparison):
    """The comparison of the columns named after the variables of
    `comparison`."""
    return comparison.functor(*(
        str2columnstr_constant(arg.name) if isinstance(arg, Symbol) else arg
        for arg in comparison.args
    ))


class NormaliseNonRepeatingQuery(
    UnifyExistentialVariableEqualitiesMixin,
    MoveNegationsIntoDisjunctionsMixin,
    PushExistentialsDownDisjunctionsMixin,
    CollapseConjunctionsMixin,
    RemoveTrivialOperationsMixin,
    LogicExpressionWalker
):
    """
    Equivalences applied before planning a non-repeating query, which add
    no atom: ∃v (v = e ∧ φ) ≡ φ[e/v], ¬(A ∨ B) ≡ ¬A ∧ ¬B and
    ∃v (A ∨ B) ≡ ∃v A ∨ ∃v B. The query must be standardised apart first.
    The planner pushes the existentials into conjunctions itself, as it
    groups the conjuncts.
    """


class NonRepeatingPlanner(PatternWalker):
    """
    Plan of a normalised non-repeating query, see `non_repeating_plan`.
    A formula that cannot be planned gives None, or no match.
    """
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(Conjunction)
    def conjunction(self, conjunction):
        return self._conjunction_plan(conjunction.formulas)

    @add_match(Disjunction)
    def disjunction(self, disjunction):
        free_variables = extract_logic_free_variables(disjunction)
        plans = []
        for formula in disjunction.formulas:
            if extract_logic_free_variables(formula) != free_variables:
                return None
            plan = self.walk(formula)
            if plan is None:
                return None
            plans.append(plan)
        return reduce(rap.Union, plans)

    @add_match(ExistentialPredicate)
    def existential(self, existential):
        """Plan of ∃v̄ φ: the groups of conjuncts of φ connected by the
        variables of v̄ are each projected on one of their variables
        occurring in all of their probabilistic atoms."""
        variables, body = _quantified_body(existential)
        conjuncts = _conjuncts(body)
        outside = [
            f for f in conjuncts
            if extract_logic_free_variables(f).isdisjoint(variables)
        ]
        planned = self._projected_groups(
            [f for f in conjuncts if f not in outside], variables
        )
        if planned is None:
            return None
        if len(outside) == 0 and len(planned) == 1:
            return planned[0][0]
        return self._conjunction_plan(outside, planned)

    @add_match(FunctionApplication(Symbol, ...))
    def atom(self, atom):
        """Plan of `atom`, whose constants and repeated variables are
        selected on named columns: positional selections would not skip
        the probability column of a probabilistic relation."""
        args = []
        selections = []
        variables = []
        for arg in atom.args:
            if isinstance(arg, Symbol) and arg not in variables:
                args.append(arg)
                variables.append(arg)
            else:
                fresh = Symbol.fresh()
                args.append(fresh)
                selections.append(EQ(fresh, arg))
        plan = TranslateToNamedRA().walk(atom.functor(*args))
        if len(selections) == 0:
            return plan
        for selection in selections:
            plan = rap.Selection(plan, columns_comparison(selection))
        return rap.Projection(
            plan, tuple(str2columnstr_constant(v.name) for v in variables)
        )

    def _conjunction_plan(self, formulas, planned=()):
        """Plan of the conjunction of `formulas` and of the plans
        `planned`, pairs of a plan and its columns."""
        positives, comparisons, negations = _partition_conjuncts(formulas)
        plans = self._positive_plans(positives, list(planned))
        if not plans:
            return None
        plan = reduce(rap.NaturalJoin, (plan for plan, _ in plans))
        columns = set().union(*(columns for _, columns in plans))
        plan = _selected_by_comparisons(plan, columns, comparisons)
        if plan is None:
            return None
        return self._without_negations(plan, columns, negations)

    def _positive_plans(self, positives, plans):
        """`plans` followed by the plans of the formulas `positives`, with
        their columns, or None if one cannot be planned."""
        for formula in positives:
            plan = self.walk(formula)
            if plan is None:
                return None
            plans.append((plan, extract_logic_free_variables(formula)))
        return plans

    def _without_negations(self, plan, columns, negations):
        """`plan` from which the plans of the negated formulas of
        `negations` are removed, or None if one has a variable that is not
        in `columns` or cannot be planned."""
        for negation in negations:
            negated = negation.formula
            if not extract_logic_free_variables(negated) <= columns:
                return None
            negated_plan = self.walk(negated)
            if negated_plan is None:
                return None
            plan = rap.Difference(plan, negated_plan)
        return plan

    def _projected_groups(self, formulas, variables):
        """The projected plans, with their columns, of the groups of
        `formulas` connected by `variables`, or None if one cannot be
        planned."""
        planned = []
        for group in _connected_groups(formulas, variables):
            plan = self._projected_group(group, variables)
            if plan is None:
                return None
            planned.append(plan)
        return planned

    def _projected_group(self, group, variables):
        group_variables = (
            extract_logic_free_variables(Conjunction(tuple(group))) &
            variables
        )
        root = self._root_variable(group, group_variables)
        if root is None:
            return None
        projection = self._projection(group, root)
        if projection is None:
            return None
        inner = Conjunction(tuple(group))
        for variable in sorted(
            group_variables - {root}, key=lambda v: v.name
        ):
            inner = ExistentialPredicate(variable, inner)
        plan = self.walk(inner)
        if plan is None:
            return None
        columns = extract_logic_free_variables(inner) - {root}
        return projection(
            plan,
            tuple(
                str2columnstr_constant(v.name)
                for v in sorted(columns, key=lambda v: v.name)
            )
        ), columns

    def _root_variable(self, group, variables):
        """A variable of `variables` occurring in every probabilistic
        atom of `group`, or None."""
        candidates = set(variables)
        for atom in extract_logic_atoms(Conjunction(tuple(group))):
            if self._is_probabilistic(atom):
                candidates &= set(atom.args)
        if len(candidates) == 0:
            return None
        return min(candidates, key=lambda v: v.name)

    def _projection(self, group, root):
        """The projection on `root` of the events of `group`: disjoint
        if a probabilistic choice conjunct has `root`, independent if no
        choice atom has it, None otherwise."""
        choice_atoms = [
            atom for atom in extract_logic_atoms(Conjunction(tuple(group)))
            if self._is_probabilistic(atom) and
            is_atom_a_probabilistic_choice_relation(atom, self.symbol_table)
            and root in atom.args
        ]
        if len(choice_atoms) == 0:
            return rap.IndependentProjection
        if any(formula in choice_atoms for formula in group):
            return rap.DisjointProjection
        return None

    def _is_probabilistic(self, atom):
        return (
            isinstance(atom.functor, Symbol) and
            not is_atom_a_deterministic_relation(atom, self.symbol_table)
        )


def _quantified_body(existential):
    """The variables quantified by the existentials heading
    `existential`, and the formula they quantify."""
    variables = set()
    body = existential
    while isinstance(body, ExistentialPredicate):
        variables.add(body.head)
        body = body.body
    return variables, body


def _partition_conjuncts(formulas):
    """The positive conjuncts, comparisons and negations of `formulas`."""
    negations = [f for f in formulas if isinstance(f, Negation)]
    comparisons = [f for f in formulas if is_comparison(f)]
    positives = [
        f for f in formulas
        if not (isinstance(f, Negation) or is_comparison(f))
    ]
    return positives, comparisons, negations


def _selected_by_comparisons(plan, columns, comparisons):
    """`plan` selected by `comparisons`, or None if one has a variable
    that is not in `columns`."""
    for comparison in comparisons:
        if not extract_logic_free_variables(comparison) <= columns:
            return None
        plan = rap.Selection(plan, columns_comparison(comparison))
    return plan


def _conjuncts(formula):
    if isinstance(formula, Conjunction):
        return list(formula.formulas)
    return [formula]


def _connected_groups(formulas, variables):
    """Groups of `formulas` connected by sharing variables of
    `variables`."""
    groups = []
    for formula in formulas:
        formula_variables = extract_logic_free_variables(formula) & variables
        connected = [
            group for group in groups
            if not group[1].isdisjoint(formula_variables)
        ]
        merged = [formula], set(formula_variables)
        for group in connected:
            groups.remove(group)
            merged[0][:0] = group[0]
            merged[1].update(group[1])
        groups.append(merged)
    return [group for group, _ in groups]


def _is_non_repeating(formula, symbol_table):
    counts = Counter(
        atom.functor for atom in extract_logic_atoms(formula)
        if isinstance(atom.functor, Symbol) and
        not is_atom_a_deterministic_relation(atom, symbol_table)
    )
    return all(count == 1 for count in counts.values())
