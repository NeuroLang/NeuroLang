"""
Ranking of queries, Dalvi & Suciu [1]_, section 4.1.

A query is ranked when there is an order on its bound variables such that
the variables of every atom follow it: its free variables, the answer
variables, are constants, as in Dalvi & Suciu, and need no order. A query
that is not ranked is rewritten into an equivalent ranked one, so that the
lifting rules, which need it, can apply.

.. [1] Dalvi, N. & Suciu, D. The dichotomy of probabilistic inference for
   unions of conjunctive queries. J. ACM 59, 1–87 (2012).
"""
import numbers
import operator
import typing
from collections import defaultdict
from itertools import product

from nl_expr.exceptions import NotRankedException
from nl_expr.expression_walker import (
    ReplaceExpressionWalker,
    add_match,
    expression_iterator,
)
from nl_expr.expressions import Constant, FunctionApplication, Symbol
from nl_expr.logic import Disjunction, Negation
from nl_expr.logic.expression_processing import (
    extract_logic_atoms,
    extract_logic_free_variables,
)
from nl_expr.logic.transformations import (
    LogicExpressionWalker,
    MakeExistentialsImplicit,
    convert_to_pnf_with_dnf_matrix,
)
from .expression_processing import is_comparison
from .probabilistic_ra_utils import (
    DeterministicFactSet,
    ProbabilisticChoiceSet,
    ProbabilisticFactSet
)
from .transforms import add_existentials_except
from ..relational_algebra import (
    Projection,
    Selection,
    eq_,
    int2columnint_constant
)

# The number of orders of n variables grows as 1, 3, 13, 75, 541...
MAX_RANKED_VARIABLES = 2


def verify_that_the_query_is_ranked(query):
    argument_precedence = defaultdict(set)
    free_variables = extract_logic_free_variables(query)
    query = convert_to_pnf_with_dnf_matrix(query)

    for atom in extract_logic_atoms(query):
        if is_comparison(atom):
            continue
        if not isinstance(atom.functor, Symbol):
            return False
        for i, arg in enumerate(atom.args[1:], 1):
            argument_precedence[arg].update(
                set(atom.args[:i]) - free_variables
            )

    for arg1, arg2 in product(argument_precedence, argument_precedence):
        if (
            arg1 in argument_precedence[arg2] and
            arg2 in argument_precedence[arg1]
        ):
            return False
    return True


def partially_rank_query(query, symbol_table):
    atoms_to_rank, atom_argument_quantity = \
        _process_atoms_with_repeatd_arguments(query)

    for symbol, instances_to_rank in atoms_to_rank.items():
        if len(instances_to_rank) != 1:
            continue
        arguments_to_rank = instances_to_rank.pop()
        if all(len(repeats) == 0 for _, repeats in arguments_to_rank):
            continue
        expression, arguments_to_eliminate = _build_ra_expression(
            atom_argument_quantity, symbol, arguments_to_rank
        )
        new_symbol = Symbol.fresh()
        symbol_table[new_symbol] = expression
        replacements = {
            atom: new_symbol(*(
                arg for i, arg in enumerate(atom.args)
                if i not in arguments_to_eliminate
            )
            )
            for atom in extract_logic_atoms(query)
            if atom.functor == symbol
        }
        query = ReplaceExpressionWalker(replacements).walk(query)
    return query


def _build_ra_expression(atom_argument_quantity, symbol, arguments_to_rank):
    expression = symbol
    arguments_to_eliminate = set()
    for arg, repeats in arguments_to_rank:
        arg = int2columnint_constant(arg)
        for repeat in repeats:
            expression = Selection(
                    expression,
                    eq_(arg, int2columnint_constant(repeat))
                )
        arguments_to_eliminate.update(repeats)
    attributes = tuple(
            int2columnint_constant(i)
            for i in range(atom_argument_quantity[symbol])
            if i not in arguments_to_eliminate
        )
    expression = Projection(expression, attributes)
    return expression, arguments_to_eliminate


def _process_atoms_with_repeatd_arguments(query):
    atoms_to_rank = defaultdict(set)
    atom_argument_quantity = dict()
    for atom in extract_logic_atoms(query):
        if not isinstance(atom.functor, Symbol):
            continue
        atom_argument_quantity[atom.functor] = len(atom.args)
        arg_counts = defaultdict(list)
        for i, arg in enumerate(atom.args):
            arg_counts[arg].append(i)
        arg_counts_tuple = tuple(
            (v[0], tuple(v[1:])) for v in arg_counts.values()
        )
        atoms_to_rank[atom.functor].add(arg_counts_tuple)
    return atoms_to_rank, atom_argument_quantity


def rank_query(query, symbol_table):
    """
    Rewrite `query`, a union of conjunctive queries, into an equivalent
    ranked query, following Dalvi & Suciu, Proposition 4.2.

    The bound variables `S` in both orders in atoms, or repeated
    in an atom, are ranked, together with the variables occurring, in an
    atom of a relation, at a position where another atom of the relation
    has a variable of `S`: every atom of a relation has variables of `S`
    at the same positions, its *ranked positions*. For each weak order of
    `S`, each atom `R(x̄)` becomes an atom of the relation of the tuples
    of `R` whose values at its ranked positions are in that order, with
    the variables of these positions, merged when equal, in increasing
    order. The query is the disjunction of these queries, one per order.
    Each tuple of `R` is in a single one of these relations, so a
    tuple-independent relation gives tuple-independent relations.

    Parameters
    ----------
    query : LogicExpression
        union of conjunctive queries, whose free variables are the
        answer variables.
    symbol_table : Mapping of Symbol to Expression
        symbol table of the relations, to which those of the ranked
        atoms are added.

    Returns
    -------
    LogicExpression
        the ranked query.

    Raises
    ------
    NotRankedException
        if the query cannot be ranked so: a relation to split is a
        probabilistic choice, which is a single block of exclusive tuples,
        or occurs within a negation; a ranked position has a constant or a
        comparison has a ranked variable; an answer variable is to rank;
        the ranked columns are not all of numbers or all of strings,
        the types with a total order and an equality, or have NaN; or more
        than `MAX_RANKED_VARIABLES` variables are to rank.
    """
    free_variables = extract_logic_free_variables(query)
    matrix = MakeExistentialsImplicit().walk(
        convert_to_pnf_with_dnf_matrix(query)
    )
    atoms = [
        atom for atom in extract_logic_atoms(matrix)
        if isinstance(atom.functor, Symbol)
    ]
    variables, positions = _ranked_variables_and_positions(
        atoms, free_variables
    )
    _check_rankable(
        matrix, variables, positions, free_variables, symbol_table
    )
    relations = {}
    disjuncts = []
    # As in Dalvi & Suciu, the variables of rank r in every order are the
    # same variable, so that the disjuncts follow a single order.
    canonical = [Symbol.fresh() for _ in variables]
    for order in _weak_orders(sorted(variables, key=lambda v: v.name)):
        substitution, ranks = _order_substitution(order, canonical)
        disjuncts.append(
            RankAtoms(positions, ranks, relations, symbol_table).walk(
                ReplaceExpressionWalker(substitution).walk(matrix)
            )
        )
    return add_existentials_except(
        Disjunction(tuple(disjuncts)), free_variables
    )


class RankAtoms(LogicExpressionWalker):
    """
    Replace each atom `R(x̄)` of a relation with ranked positions by an
    atom of the relation of the tuples of `R` whose values at these
    positions are ordered as their variables are by `ranks`, see
    `rank_query`.
    """
    def __init__(self, positions, ranks, relations, symbol_table):
        self.positions = positions
        self.ranks = ranks
        self.relations = relations
        self.symbol_table = symbol_table

    @add_match(FunctionApplication(Symbol, ...))
    def ranked_atom(self, atom):
        if atom.functor not in self.positions:
            return atom
        ranked = sorted(self.positions[atom.functor])
        distinct = sorted({self.ranks[atom.args[i]] for i in ranked})
        pattern = tuple(
            distinct.index(self.ranks[atom.args[i]]) for i in ranked
        )
        key = (atom.functor, pattern)
        if key not in self.relations:
            symbol = Symbol.fresh()
            self.symbol_table[symbol] = _ranked_relation(
                atom.functor, len(atom.args), ranked, pattern,
                self.symbol_table
            )
            self.relations[key] = symbol
        others = tuple(
            arg for i, arg in enumerate(atom.args) if i not in ranked
        )
        increasing = tuple(
            atom.args[ranked[pattern.index(rank)]]
            for rank in range(len(distinct))
        )
        return self.relations[key](*(others + increasing))


def _ranked_relation(symbol, arity, ranked, pattern, symbol_table):
    """The relation of the tuples of `symbol` whose values at the
    positions `ranked` are in the order `pattern`, with the other
    positions first and then one position per rank, in increasing order.

    It is computed on the data, keeping the probability of each tuple,
    and wrapped as the relation of `symbol` is."""
    entry = symbol_table[symbol]
    offset = int(isinstance(entry, ProbabilisticFactSet))
    relation = entry
    if isinstance(entry, (ProbabilisticFactSet, DeterministicFactSet)):
        relation = symbol_table[entry.relation]
    conditions = []
    for (i, rank_i), (j, rank_j) in product(
        zip(ranked, pattern), repeat=2
    ):
        if i < j:
            conditions.append((
                i + offset, j + offset,
                operator.eq if rank_i == rank_j else
                operator.lt if rank_i < rank_j else operator.gt
            ))
    columns = tuple(range(offset)) + tuple(
        i + offset for i in range(arity) if i not in ranked
    ) + tuple(
        ranked[pattern.index(rank)] + offset
        for rank in range(len(set(pattern)))
    )
    value = relation.value.selection(
        lambda t: all(op(t[i], t[j]) for i, j, op in conditions)
    ).projection(*columns)
    types = typing.get_args(typing.get_args(relation.type)[0])
    ranked_relation = Constant[
        typing.AbstractSet[typing.Tuple[tuple(types[c] for c in columns)]]
    ](value, verify_type=False)
    if relation is entry:
        return ranked_relation
    fresh = Symbol.fresh()
    symbol_table[fresh] = ranked_relation
    if isinstance(entry, ProbabilisticFactSet):
        return ProbabilisticFactSet(fresh, entry.probability_column)
    return DeterministicFactSet(fresh)


def _ranked_variables_and_positions(atoms, free_variables):
    """The variables to rank, and the ranked positions of each relation,
    closed so that every atom of a relation has variables to rank at its
    ranked positions."""
    precedence = defaultdict(set)
    for atom in atoms:
        for i, arg in enumerate(atom.args):
            precedence[arg].update(set(atom.args[:i]) - free_variables)
    variables = {
        arg for arg, previous in precedence.items()
        if isinstance(arg, Symbol) and (
            arg in previous or
            any(arg in precedence[other] for other in previous)
        )
    }
    while True:
        positions = defaultdict(set)
        for atom in atoms:
            for i, arg in enumerate(atom.args):
                if arg in variables:
                    positions[atom.functor].add(i)
        ranked_args = {
            atom.args[i] for atom in atoms
            for i in positions.get(atom.functor, ())
        }
        if any(not isinstance(arg, Symbol) for arg in ranked_args):
            raise NotRankedException(
                "A constant is at a position to rank"
            )
        if ranked_args <= variables:
            return variables, dict(positions)
        variables |= ranked_args


def _check_rankable(
    matrix, variables, positions, free_variables, symbol_table
):
    if len(variables) > MAX_RANKED_VARIABLES:
        raise NotRankedException(
            f"More than {MAX_RANKED_VARIABLES} variables to rank"
        )
    if variables & free_variables:
        # Dalvi & Suciu rank Boolean queries; the answer variables are
        # constants, which this ranking does not shatter.
        raise NotRankedException("An answer variable to rank")
    for atom in extract_logic_atoms(matrix):
        if (
            not isinstance(atom.functor, Symbol) and
            extract_logic_free_variables(atom) & variables
        ):
            raise NotRankedException(
                "A comparison or a function of a variable to rank"
            )
    negated = {
        atom.functor
        for _, negation in expression_iterator(matrix)
        if isinstance(negation, Negation)
        for atom in extract_logic_atoms(negation)
    }
    if negated & set(positions):
        raise NotRankedException("A relation to rank is negated")
    column_types = []
    for symbol, ranked in positions.items():
        types, relation = _column_types(symbol, symbol_table)
        column_types.extend(types[i] for i in ranked)
        if any(issubclass(types[i], float) for i in ranked):
            _check_no_nan(relation, symbol, ranked, symbol_table)
    if not _is_totally_ordered(column_types):
        raise NotRankedException(
            "The columns to rank are not all numbers or all strings"
        )


def _column_types(symbol, symbol_table):
    """The types of the columns of the relation `symbol`, without its
    probability column, and the relation."""
    entry = symbol_table.get(symbol)
    offset = 0
    if isinstance(entry, ProbabilisticChoiceSet):
        raise NotRankedException(
            "A probabilistic choice cannot be split into ranked relations"
        )
    if isinstance(entry, ProbabilisticFactSet):
        offset = 1
    if isinstance(entry, (ProbabilisticFactSet, DeterministicFactSet)):
        entry = symbol_table.get(entry.relation)
    if not isinstance(entry, Constant):
        raise NotRankedException(f"The relation {symbol} has no known type")
    tuple_types = typing.get_args(entry.type)
    if len(tuple_types) != 1:
        raise NotRankedException(f"The relation {symbol} has no known type")
    types = typing.get_args(tuple_types[0])
    if not all(isinstance(t, type) for t in types):
        raise NotRankedException(f"The relation {symbol} has no known type")
    return types[offset:], entry


def _check_no_nan(relation, symbol, ranked, symbol_table):
    offset = int(
        isinstance(symbol_table.get(symbol), ProbabilisticFactSet)
    )
    data = relation.value.as_pandas_dataframe()
    if data.iloc[:, [i + offset for i in ranked]].isna().any(axis=None):
        raise NotRankedException(
            f"The relation {symbol} has NaN in a column to rank"
        )


def _is_totally_ordered(types):
    """Whether the values of `types` have a total order and an equality
    together: all real numbers, or all strings. Sets, for instance, are
    partially ordered by inclusion."""
    return (
        all(
            issubclass(t, numbers.Real) and not issubclass(t, complex)
            for t in types
        ) or
        all(issubclass(t, str) for t in types)
    )


def _weak_orders(variables):
    """The weak orders of `variables`, as lists of sets of equal
    variables in increasing order."""
    if len(variables) == 0:
        yield []
        return
    first, rest = variables[0], variables[1:]
    for order in _weak_orders(rest):
        for i in range(len(order)):
            yield order[:i] + [order[i] | {first}] + order[i + 1:]
        for i in range(len(order) + 1):
            yield order[:i] + [{first}] + order[i:]


def _order_substitution(order, canonical):
    """The substitution of each variable by the variable `canonical` of
    the rank of its set of equal variables, and the rank of each of
    these."""
    substitution = {}
    ranks = {}
    for rank, equal in enumerate(order):
        ranks[canonical[rank]] = rank
        for variable in equal:
            substitution[variable] = canonical[rank]
    return substitution, ranks
