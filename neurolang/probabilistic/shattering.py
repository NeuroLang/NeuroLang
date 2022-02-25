import collections
import itertools
import operator
from typing import AbstractSet

from ..expression_pattern_matching import add_match
from ..expression_walker import (
    ExpressionWalker,
    ReplaceExpressionWalker,
    expression_iterator
)
from ..expressions import Constant, FunctionApplication, Symbol
from ..logic import (
    TRUE,
    Conjunction,
    Disjunction,
    ExistentialPredicate,
    Implication
)
from ..logic.expression_processing import extract_logic_atoms
from ..logic.transformations import (
    LogicExpressionWalker,
    RemoveDuplicatedConjunctsDisjuncts,
    RemoveTrivialOperations,
    convert_to_pnf_with_dnf_matrix
)
from ..relational_algebra import Projection, Selection, int2columnint_constant
from .exceptions import NotEasilyShatterableError
from .probabilistic_ra_utils import ProbabilisticFactSet
from .transforms import convert_to_dnf_ucq

EQ = Constant(operator.eq)
NE = Constant(operator.ne)
RTO = RemoveTrivialOperations()


def query_to_tagged_set_representation(query, symbol_table):
    new_antecedent = ReplaceExpressionWalker(symbol_table).walk(
        query.antecedent
    )
    return query.apply(query.consequent, new_antecedent)


def terms_differ_by_constant_term(terms_a, terms_b):
    return any(
        isinstance(term_a, Constant)
        and isinstance(term_b, Constant)
        and term_a != term_b
        for term_a, term_b in zip(terms_a, terms_b)
    )


def all_terms_differ_by_constant_term(list_of_tuple_of_terms):
    return all(
        terms_differ_by_constant_term(terms_a, terms_b)
        for terms_a, terms_b in itertools.combinations(
            list_of_tuple_of_terms, 2
        )
    )


def constant_terms_are_constant_in_all_tuples(list_of_tuple_of_terms):
    arity = len(list_of_tuple_of_terms[0])
    for i in range(arity):
        nb_of_constant_terms = sum(
            isinstance(terms[i], Constant) for terms in list_of_tuple_of_terms
        )
        if 0 < nb_of_constant_terms < len(list_of_tuple_of_terms):
            return False
    return True


def is_easily_shatterable_self_join(list_of_tuple_of_terms):
    """
    A self-join of `m` predicates is easily shatterable if the following two
    conditions are met.

    Firstly, if the `i`th term of one of the predicates is a constant, then the
    `i`th terms of all the other self-joined predicates must also be constants.
    For example, the self-join `P(x, a), P(x, y)` does not meet this condition
    because the second term appears both as a constant (in the first predicate)
    and as a variable (in the second predicate).

    Secondly, all predicates in the self-join must differ by at least one
    constant term. In other words, for two predicates P1 and P2 in the
    self-join, there must be a constant term at some position `i` in P1 whose
    value is different than the constant `i`th term in P2 (we know it's
    constant from the first condition). For example, the self-join `P(x, a),
    P(y, a)` does not meet this condition because the only constant term is `a`
    in both predicates. However, the self-join `P(x, a, b), P(x, a, c)` does
    meet this condition because the predicates differ in their constant value
    of the third term (`b` for the first predicate and `c` for the second
    predicate).

    """
    return constant_terms_are_constant_in_all_tuples(
        list_of_tuple_of_terms
    ) and all_terms_differ_by_constant_term(list_of_tuple_of_terms)


class Shatter(FunctionApplication):
    pass


class Shatterer(ExpressionWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table
        self._cached_args = collections.defaultdict(set)
        self._cached = dict()

    @add_match(
        Shatter(ProbabilisticFactSet(..., int2columnint_constant(0)), ...)
    )
    def easy_shatter_probfact(self, shatter):
        pred_symb = shatter.functor.relation
        cache_key = (pred_symb,)
        for i, arg in enumerate(shatter.args):
            if isinstance(arg, Constant):
                cache_key += (arg,)
            else:
                cache_key += (i,)
        if cache_key in self._cached:
            new_pred_symb = self._cached[cache_key]
        else:
            new_pred_symb = Symbol.fresh()
            const_idxs = list(
                i
                for i, arg in enumerate(shatter.args)
                if isinstance(arg, Constant)
            )
            columns = [
                int2columnint_constant(c)
                for c in self.symbol_table[shatter.functor.relation]
                .value.columns
            ]
            new_relation = shatter.functor.relation
            for i in const_idxs:
                new_relation = Selection(
                    new_relation,
                    EQ(columns[i + 1], shatter.args[i])
                )
            non_prob_columns = tuple(
                c
                for c in columns
                if c != shatter.functor.probability_column
            )
            proj_cols = (shatter.functor.probability_column,) + tuple(
                non_prob_columns[i]
                for i, arg in enumerate(shatter.args)
                if not isinstance(arg, Constant)
            )
            new_relation = Projection(new_relation, proj_cols)
            self.symbol_table[new_pred_symb] = new_relation
            self._cached[cache_key] = new_pred_symb
        new_tagged = ProbabilisticFactSet(
            new_pred_symb, shatter.functor.probability_column
        )
        non_const_args = tuple(
            arg for arg in shatter.args if not isinstance(arg, Constant)
        )
        return FunctionApplication(new_tagged, non_const_args)

    @add_match(
        FunctionApplication(ProbabilisticFactSet, ...),
        lambda fa: not isinstance(fa, Shatter)
        and any(isinstance(arg, Constant) for arg in fa.args),
    )
    def shatter_probfact_predicates(self, function_application):
        self._check_can_shatter(function_application)
        self._cached_args[function_application.functor.relation].add(
            function_application.args
        )
        return self.walk(Shatter(*function_application.unapply()))

    @add_match(
        FunctionApplication(ProbabilisticFactSet, ...),
        lambda fa: not isinstance(fa, Shatter),
    )
    def cache_non_constant_args(self, function_application):
        self._check_can_shatter(function_application)
        self._cached_args[function_application.functor.relation].add(
            function_application.args
        )
        return function_application

    def _check_can_shatter(self, function_application):
        pred_symb = function_application.functor.relation
        args = function_application.args
        list_of_tuple_of_terms = list(
            self._cached_args.get(pred_symb, set()).union({args})
        )
        if not is_easily_shatterable_self_join(list_of_tuple_of_terms):
            raise NotEasilyShatterableError(
                f"Cannot easily shatter {pred_symb}-predicates"
            )


def shatter_easy_probfacts(query, symbol_table):
    """
    Remove constants occurring in a given query, possibly removing self-joins.

    A query containing self-joins can be "easily" shattered whenever the
    predicates in the self-joins do not have more than one variable occurring
    in the same term in multiple predicates (e.g. `P(x), P(y)`, both `x` and
    `y` occurr in the same term in both predicates) or the same constant
    occurring in the same term in multiple predicates (e.g. `P(a, x), P(a, b)`,
    `a` occurrs in the same term in both predicates).

    If there is a self-join, the self-joined relation is split into multiple
    relations. These relations are added in-place to the symbol table. The
    returned equivalent query makes use of these relations.

    Parameters
    ----------
    query : Implication
        A conjunctive query (the body can be a single predicate).
    symbol_table : mapping, mutable
        Symbol table containing relations associated with the query's
        relational symbols. This `symbol_table` can be modified in-place by
        this function to add newly generated relational symbols and their
        associated relations.

    Returns
    -------
    Implication
        An equivalent conjunctive query without constants.

    """
    dnf_query_antecedent = convert_to_dnf_ucq(query.antecedent)
    dnf_query_antecedent = RemoveDuplicatedConjunctsDisjuncts().walk(
        dnf_query_antecedent
    )
    dnf_query = Implication(query.consequent, dnf_query_antecedent)
    tagged_query = query_to_tagged_set_representation(dnf_query, symbol_table)
    shatterer = Shatterer(symbol_table)
    shattered = shatterer.walk(tagged_query)
    shattered = RemoveTrivialOperations().walk(shattered)
    return shattered


def atom_to_constant_to_RA_conditions(atom: FunctionApplication) -> set:
    conditions = set()
    for i, arg in enumerate(atom.args):
        if isinstance(arg, Constant):
            conditions.add(EQ(int2columnint_constant(i), arg))

    return conditions


class NormalizeNotEquals(LogicExpressionWalker):
    @add_match(FunctionApplication(NE, (Constant, Symbol)))
    def flip_ne_arguments(self, expression):
        return expression.apply(
            NE,
            expression.args[::-1]
        )


def conditions_per_symbol(ucq_query):
    ucq_query_dnf = convert_to_pnf_with_dnf_matrix(ucq_query)
    ucq_query_dnf = NormalizeNotEquals().walk(ucq_query_dnf)

    conditions_per_symbol = collections.defaultdict(lambda: set())
    number_of_args = dict()
    for atom in extract_logic_atoms(ucq_query_dnf):
        if not isinstance(atom.functor, Symbol):
            continue
        condition = atom_to_constant_to_RA_conditions(atom)
        conditions_per_symbol[atom.functor] |= condition
        number_of_args[atom.functor] = len(atom.args)

    conjunctions = (
        expression for _, expression in expression_iterator(ucq_query_dnf)
        if isinstance(expression, Conjunction) and
        any(
            _formula_is_ne(formula)
            for formula in expression.formulas
        )
    )

    for conjunction in conjunctions:
        nes = (
            formula for formula in conjunction.formulas
            if _formula_is_ne(formula)
        )
        for ne in nes:
            ne_symbol = ne.args[0]
            ne_constant = ne.args[1]
            for atom in extract_logic_atoms(conjunction):
                if not isinstance(atom.functor, Symbol):
                    continue
                ne_conditions = {
                    NE(int2columnint_constant(i), ne_constant)
                    for i, s in enumerate(atom.args)
                    if s == ne_symbol
                }
                if ne_conditions:
                    conditions_per_symbol[atom.functor] |= ne_conditions
                    number_of_args[atom.functor] = len(atom.args)

    return conditions_per_symbol, number_of_args


def _formula_is_ne(formula):
    return (
        isinstance(formula, FunctionApplication) and
        formula.functor == NE and
        isinstance(formula.args[0], Symbol)
    )


def sets_per_symbol(ucq_query):
    cps, number_of_args = conditions_per_symbol(ucq_query)

    complementary = {EQ: NE, NE: EQ}
    symbol_sets = dict()

    for symbol, conditions in cps.items():
        if len(conditions) == 0:
            continue
        column_conditions = collections.defaultdict(lambda: set())
        for condition in conditions:
            if condition.functor == NE:
                condition = condition.apply(EQ, condition.unapply()[1])
            column_conditions[condition.args[0]].add((condition,))

        for column, conditions in column_conditions.items():
            column_conditions[column].add(
                tuple(
                    complementary[c[0].functor](*c[0].args) for c in conditions

                )
            )

        set_formulas = list(
            Conjunction(tuple(
                sorted(
                    itertools.chain(*conditions),
                    key=lambda exp: (exp.args[0].value, exp.args[1].value)
                )
            ))
            for conditions in
            itertools.product(*column_conditions.values())
        )
        conditions_columns = [
            set(
                condition_item.args[0].value
                for condition_item in condition.formulas
                if condition_item.functor == EQ
            )
            for condition in set_formulas
        ]
        conditions_projected_args = [
            tuple(
                int2columnint_constant(i)
                for i in range(number_of_args[symbol])
                # if i not in condition_columns
            )
            for condition_columns in conditions_columns
        ]

        symbol_sets[symbol] = set(
            Projection(Selection(symbol, RTO.walk(condition)), projected_args)
            for condition, projected_args
            in zip(set_formulas, conditions_projected_args)
        )

    return symbol_sets


class ShatterEqualities(ExpressionWalker):
    def __init__(self, symbol_sets, symbol_table):
        self.symbol_sets = symbol_sets
        self.symbol_table = symbol_table
        self.shatter_symbols = dict()

    @add_match(FunctionApplication(Symbol, ...))
    def shatter_symbol(self, expression):
        if expression.functor not in self.symbol_sets:
            return expression

        expression = self._shatter_atom(expression)

        return expression

    def _shatter_atom(self, expression, nes=None):
        sets = self.symbol_sets[expression.functor]
        sets_to_keep = self._identify_sets_to_keep(expression, sets, nes=nes)

        expression, e_vars = self._compute_expression_arguments(
            expression, sets_to_keep
        )

        if len(expression) == 1:
            expression = expression[0]
        else:
            expression = Disjunction(expression)

        for e_var in e_vars:
            expression = ExistentialPredicate(e_var, expression)
        return expression

    @add_match(
        Conjunction,
        lambda expression: (
            any(
                _formula_is_ne(formula) and
                isinstance(formula.args[1], Constant)
                for formula in expression.formulas
            )
        )
    )
    def shatter_inequalities(self, expression):
        nes = set()
        formulas_to_keep = []
        for formula in expression.formulas:
            if _formula_is_ne(formula):
                nes.add(formula)
            else:
                formulas_to_keep.append(formula)

        resulting_formulas = tuple()
        for formula in formulas_to_keep:
            if formula.functor not in self.symbol_sets:
                resulting_formulas += (formula,)
            else:
                resulting_formulas += (self._shatter_atom(formula, nes=nes),)

        return Conjunction(resulting_formulas)

    def _compute_expression_arguments(self, expression, sets_to_keep):
        args = []
        for i, arg in enumerate(expression.args):
            if isinstance(arg, Symbol):
                args.append((i, arg, False))
            elif isinstance(arg, Constant):
                args.append((i, Symbol.fresh(), True))
        formulas = tuple()
        e_vars = set()
        for set_ in sets_to_keep:
            if set_ not in self.shatter_symbols:
                sym = Symbol.fresh()
                self.shatter_symbols[set_] = sym
                self.symbol_table[sym] = set_
            formula_args = []
            for i, arg, is_e_var in args:
                if int2columnint_constant(i) in set_.columns():
                    formula_args.append(arg)
                    if is_e_var:
                        e_vars.add(arg)
            formulas += (self.shatter_symbols[set_](*formula_args),)
        return formulas, e_vars

    def _identify_sets_to_keep(self, expression, sets, nes=None):
        conditions = atom_to_constant_to_RA_conditions(expression)
        negative_conditions = set()
        if nes is not None:
            args = list(expression.args)
            for ne in nes:
                if ne.args[0] in expression.args:
                    negative_conditions.add(
                        EQ(
                            int2columnint_constant(args.index(ne.args[0])),
                            ne.args[1]
                        )
                    )
        sets_to_keep = []
        for set_ in sets:
            set_conditions = self._selection_formulas(set_)
            if (
                conditions <= set_conditions and
                set_conditions.isdisjoint(negative_conditions)
            ):
                sets_to_keep.append(set_)
        return sets_to_keep

    def _selection_formulas(self, set_):
        formula = set_.relation.formula
        if isinstance(formula, Conjunction):
            set_conditions = set(formula.formulas)
        else:
            set_conditions = {formula}
        return set_conditions


def eliminate_non_equals(query):
    relational_atom_arguments = {
        arg
        for atom in extract_logic_atoms(query)
        for arg in atom.args
        if (
            isinstance(atom.functor, Symbol) and
            isinstance(arg, Symbol)
        )
    }

    ne_to_replace = {
        atom: TRUE
        for atom in extract_logic_atoms(query)
        if (
            atom.functor is NE and
            atom.args[0] in relational_atom_arguments and
            isinstance(atom.args[1], Constant)
        )
    }

    query = ReplaceExpressionWalker(ne_to_replace).walk(query)
    query = RTO.walk(query)


def shatter_constants(query, symbol_table):
    symbol_sets = sets_per_symbol(query)
    se = ShatterEqualities(symbol_sets, symbol_table)
    query = se.walk(query)
    return query
