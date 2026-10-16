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
from ..datalog.translate_to_named_ra import TranslateToNamedRA
from ..expression_walker import ReplaceExpressionWalker, expression_iterator
from ..expressions import FunctionApplication, Symbol
from ..logic import Conjunction, Disjunction, ExistentialPredicate, Negation
from ..logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables
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
        if `formula` has no negation, is repeating, or is not
        hierarchical.
    """
    if not (
        _has_a_negation(formula) and
        _is_non_repeating(formula, symbol_table)
    ):
        return None
    return _NonRepeatingPlanner(symbol_table).plan(
        _eliminate_equalities(formula)
    )


def columns_comparison(comparison):
    """The comparison of the columns named after the variables of
    `comparison`."""
    return comparison.functor(*(
        str2columnstr_constant(arg.name) if isinstance(arg, Symbol) else arg
        for arg in comparison.args
    ))


class _NonRepeatingPlanner:
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    def plan(self, formula):
        if isinstance(formula, Conjunction):
            return self._conjunction(formula.formulas)
        if isinstance(formula, ExistentialPredicate):
            return self._existential(formula)
        if isinstance(formula, Disjunction):
            return self._disjunction(formula)
        if (
            isinstance(formula, FunctionApplication) and
            isinstance(formula.functor, Symbol)
        ):
            return _atom_plan(formula)
        return None

    def _conjunction(self, formulas, planned=()):
        """Plan of the conjunction of `formulas` and of the plans
        `planned`, pairs of a plan and its columns."""
        formulas = _conjuncts(formulas)
        negations = [f for f in formulas if isinstance(f, Negation)]
        comparisons = [f for f in formulas if is_comparison(f)]
        positives = [
            f for f in formulas
            if not (isinstance(f, Negation) or is_comparison(f))
        ]
        plans = list(planned)
        for formula in positives:
            plan = self.plan(formula)
            if plan is None:
                return None
            plans.append((plan, extract_logic_free_variables(formula)))
        if len(plans) == 0:
            return None
        plan = reduce(rap.NaturalJoin, (plan for plan, _ in plans))
        columns = set().union(*(columns for _, columns in plans))
        for comparison in comparisons:
            if not extract_logic_free_variables(comparison) <= columns:
                return None
            plan = rap.Selection(plan, columns_comparison(comparison))
        for negation in negations:
            negated = negation.formula
            if not extract_logic_free_variables(negated) <= columns:
                return None
            negated_plan = self.plan(negated)
            if negated_plan is None:
                return None
            plan = rap.Difference(plan, negated_plan)
        return plan

    def _disjunction(self, disjunction):
        free_variables = extract_logic_free_variables(disjunction)
        plans = []
        for formula in disjunction.formulas:
            if extract_logic_free_variables(formula) != free_variables:
                return None
            plan = self.plan(formula)
            if plan is None:
                return None
            plans.append(plan)
        return reduce(rap.Union, plans)

    def _existential(self, existential):
        """Plan of ∃v̄ φ: the quantifiers are pushed down to the groups
        of conjuncts of φ connected by the variables of v̄, each of which
        is projected on one of its variables occurring in all of its
        probabilistic atoms."""
        variables = set()
        body = existential
        while isinstance(body, ExistentialPredicate):
            variables.add(body.head)
            body = body.body
        if isinstance(body, Disjunction):
            return self.plan(Disjunction(tuple(
                _add_existentials(formula, variables)
                for formula in body.formulas
            )))
        conjuncts = _conjuncts((body,))
        outside = [
            f for f in conjuncts
            if extract_logic_free_variables(f).isdisjoint(variables)
        ]
        planned = []
        for group in _connected_groups(
            [f for f in conjuncts if f not in outside], variables
        ):
            plan = self._projected_group(group, variables)
            if plan is None:
                return None
            planned.append(plan)
        if len(outside) == 0 and len(planned) == 1:
            return planned[0][0]
        return self._conjunction(outside, planned)

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
        plan = self.plan(inner)
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
        if any(
            formula in choice_atoms for formula in _conjuncts(group)
        ):
            return rap.DisjointProjection
        return None

    def _is_probabilistic(self, atom):
        return (
            isinstance(atom.functor, Symbol) and
            not is_atom_a_deterministic_relation(atom, self.symbol_table)
        )


def _atom_plan(atom):
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


def _conjuncts(formulas):
    """The conjuncts of the conjunction of `formulas`, a negated
    disjunction being the conjunction of its negated disjuncts."""
    conjuncts = []
    for formula in formulas:
        if isinstance(formula, Conjunction):
            conjuncts.extend(_conjuncts(formula.formulas))
        elif (
            isinstance(formula, Negation) and
            isinstance(formula.formula, Disjunction)
        ):
            conjuncts.extend(_conjuncts(
                Negation(f) for f in formula.formula.formulas
            ))
        else:
            conjuncts.append(formula)
    return conjuncts


def _add_existentials(formula, variables):
    """∃v̄ `formula`, for the variables `v̄` of `variables` free in it."""
    free_variables = extract_logic_free_variables(formula)
    for variable in sorted(variables & free_variables, key=lambda v: v.name):
        formula = ExistentialPredicate(variable, formula)
    return formula


def _eliminate_equalities(formula):
    """`formula` where each existentially quantified variable `v` equal
    to a term `e` by a conjunct `v = e` is replaced by `e`, as
    ∃v (v = e ∧ φ) ≡ φ[e/v]. This is done in every subformula before
    planning, so that the variables that occur in all the atoms of a
    group are known.
    """
    if isinstance(formula, ExistentialPredicate):
        variables = set()
        while isinstance(formula, ExistentialPredicate):
            variables.add(formula.head)
            formula = formula.body
        conjuncts, variables = _substitute_equalities(
            _conjuncts((_eliminate_equalities(formula),)), variables
        )
        return _add_existentials(
            Conjunction(tuple(conjuncts)) if len(conjuncts) > 1
            else conjuncts[0],
            variables
        )
    if isinstance(formula, (Conjunction, Disjunction)):
        return type(formula)(tuple(
            _eliminate_equalities(f) for f in formula.formulas
        ))
    if isinstance(formula, Negation):
        return Negation(_eliminate_equalities(formula.formula))
    return formula


def _substitute_equalities(conjuncts, variables):
    """Replace the variables of `variables` equal to a term by a
    conjunct of `conjuncts` by that term, and remove the equalities.

    Returns
    -------
    Tuple[List[LogicExpression], Set[Symbol]]
        the conjuncts and the variables which are left.
    """
    variables = set(variables)
    conjuncts = list(conjuncts)
    while True:
        substitution = next(
            (
                (conjunct, variable, term)
                for conjunct in conjuncts
                if isinstance(conjunct, FunctionApplication) and
                conjunct.functor == EQ
                for variable, term in (
                    conjunct.args, tuple(reversed(conjunct.args))
                )
                if variable in variables and variable != term and
                not ({variable, term} & _bound_variables(conjuncts))
            ),
            None
        )
        if substitution is None or len(conjuncts) == 1:
            return conjuncts, variables
        equality, variable, term = substitution
        replace = ReplaceExpressionWalker({variable: term})
        conjuncts = [
            replace.walk(conjunct) for conjunct in conjuncts
            if conjunct is not equality
        ]
        variables.discard(variable)


def _bound_variables(formulas):
    """Variables quantified within `formulas`, into which a substitution
    could be captured."""
    return {
        expression.head
        for formula in formulas
        for _, expression in expression_iterator(formula)
        if isinstance(expression, ExistentialPredicate)
    }


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


def _has_a_negation(formula):
    if isinstance(formula, Negation):
        return True
    if isinstance(formula, (Conjunction, Disjunction)):
        return any(_has_a_negation(f) for f in formula.formulas)
    if isinstance(formula, ExistentialPredicate):
        return _has_a_negation(formula.body)
    return False


def _is_non_repeating(formula, symbol_table):
    counts = Counter(
        atom.functor for atom in extract_logic_atoms(formula)
        if isinstance(atom.functor, Symbol) and
        not is_atom_a_deterministic_relation(atom, symbol_table)
    )
    return all(count == 1 for count in counts.values())
