"""
Deterministic subqueries of a probabilistic query.

A subquery whose relations are all deterministic is the same relation in
every possible world. It can be replaced by a fresh deterministic
relation, defined by the subquery's relational algebra expression, which
a lifted solver then handles like any other deterministic relation. The
expression is not evaluated while planning: marked with `Materialise`,
it is computed as a set when the plan is evaluated.
"""
from ..datalog.expression_processing import extract_logic_atoms
from ..datalog.translate_to_named_ra import TranslateToNamedRA
from ..expression_walker import ReplaceSymbolWalker, add_match
from ..expressions import Constant, Symbol
from ..logic import Conjunction, ExistentialPredicate, Negation
from ..logic.expression_processing import extract_logic_free_variables
from ..logic.transformations import LogicExpressionWalker
from ..relational_algebra import (
    Materialise,
    NumberColumns,
    Projection,
    str2columnstr_constant
)
from ..relational_algebra.optimisers import RelationalAlgebraOptimiser


class ReplaceDeterministicNegations(LogicExpressionWalker):
    """
    Replace each negated conjunct ¬N of a conjunction, N being a
    conjunction or an existential whose atoms are all deterministic
    relations or comparisons, by a fresh deterministic relation

        D(x̄) := G(x̄) ∧ ¬N(x̄)

    x̄ being the free variables of N, and G the conjunction of the
    deterministic conjuncts whose free variables are among x̄. As G is
    asserted by the conjunction, G ∧ ¬N ≡ G ∧ D. G must be non-empty and
    bind every variable of x̄, for D to be finite, otherwise ¬N is left
    unchanged.

    The relations are in the attribute `relations`, a mapping from their
    symbols to their unevaluated relational algebra expressions, marked
    with `Materialise`, whose columns are numbered in the order of x̄
    sorted by name.

    Parameters
    ----------
    program : DatalogProgram
        program whose extensional database has the deterministic
        relations, and which has no probabilistic relation in it.
    """

    def __init__(self, program):
        self.program = program
        self.relations = {}

    @add_match(
        Conjunction,
        lambda conjunction: any(
            isinstance(formula, Negation) and
            isinstance(
                formula.formula, (Conjunction, ExistentialPredicate)
            )
            for formula in conjunction.formulas
        )
    )
    def conjunction_with_negations(self, conjunction):
        formulas = tuple(
            self._replace(formula, conjunction)
            if isinstance(formula, Negation) else self.walk(formula)
            for formula in conjunction.formulas
        )
        return Conjunction(formulas)

    def _replace(self, negation, conjunction):
        free_variables = extract_logic_free_variables(negation)
        guard = tuple(
            formula for formula in conjunction.formulas
            if not isinstance(formula, Negation) and
            extract_logic_free_variables(formula) <= free_variables and
            self._is_deterministic(formula)
        )
        if not self._can_replace(negation, guard, free_variables):
            return negation
        columns = tuple(sorted(free_variables, key=lambda v: v.name))
        symbol = Symbol.fresh()
        self.relations[symbol] = self._relation(guard, negation, columns)
        return symbol(*columns)

    def _can_replace(self, negation, guard, free_variables):
        return (
            len(guard) > 0 and
            isinstance(
                negation.formula, (Conjunction, ExistentialPredicate)
            ) and
            self._is_deterministic(negation) and
            free_variables <= extract_logic_free_variables(
                Conjunction(guard)
            )
        )

    def _is_deterministic(self, formula):
        deterministic = self.program.extensional_database()
        return all(
            atom.functor in deterministic or
            isinstance(atom.functor, Constant)
            for atom in extract_logic_atoms(formula)
        )

    def _relation(self, guard, negation, columns):
        """
        The relational algebra expression of G ∧ ¬N, with numbered
        columns, optimised, and marked to be materialised when the plan
        is evaluated. Its relations are the sets of the extensional database,
        rather than their symbols, which the lifted solver maps to
        probabilistic sets.
        """
        negated = Negation(_conjoin_within_existentials(
            guard, negation.formula
        ))
        column_names = tuple(str2columnstr_constant(v.name) for v in columns)
        plan = Projection(
            TranslateToNamedRA().walk(Conjunction(guard + (negated,))),
            column_names
        )
        plan = ReplaceSymbolWalker(
            self.program.extensional_database()
        ).walk(plan)
        plan = RelationalAlgebraOptimiser().walk(plan)
        return Materialise(NumberColumns(plan, column_names))


def _conjoin_within_existentials(formulas, formula):
    """
    G ∧ ∃y B as ∃y (G ∧ B), *formulas* being the conjuncts of G, in
    which y is not free. This makes the existential range restricted
    when B alone is not, as in ∃y (R(x, y) ∧ y ≠ z).
    """
    if isinstance(formula, ExistentialPredicate):
        return formula.apply(
            formula.head,
            _conjoin_within_existentials(formulas, formula.body)
        )
    if isinstance(formula, Conjunction):
        return Conjunction(formulas + formula.formulas)
    return Conjunction(formulas + (formula,))


def replace_deterministic_negations(formula, program):
    """
    Replace the deterministic negated subqueries of *formula* by fresh
    deterministic relations, see `ReplaceDeterministicNegations`.

    Returns
    -------
    Tuple[LogicExpression, Dict[Symbol, RelationalAlgebraOperation]]
        the new formula and the relations, as unevaluated relational
        algebra expressions.
    """
    walker = ReplaceDeterministicNegations(program)
    return walker.walk(formula), walker.relations
