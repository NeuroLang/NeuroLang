from collections import defaultdict
from typing import AbstractSet, Any, Mapping

from ..datalog.wrapped_collections import (
    WrappedNamedRelationalAlgebraFrozenSet
)
from ..expression_walker import PatternWalker, add_match
from ..expressions import Constant, Expression, Symbol
from ..relational_algebra import (
    ColumnInt,
    RelationalAlgebraOperation,
    UnaryRelationalAlgebraOperation
)


class ProbabilisticFactSet(RelationalAlgebraOperation):
    def __init__(self, relation, probability_column):
        self.relation = relation
        self.probability_column = probability_column


class ProbabilisticChoiceSet(RelationalAlgebraOperation):
    def __init__(self, relation, probability_column):
        self.relation = relation
        self.probability_column = probability_column


class DeterministicFactSet(RelationalAlgebraOperation):
    def __init__(self, relation):
        self.relation = relation


class NonLiftable(RelationalAlgebraOperation):
    """Represents a logic-based probabilistic query
    which could not be lifted into a safe plan.
    """
    def __init__(self, non_liftable_query):
        self.non_liftable_query = non_liftable_query

    def __repr__(self):
        return (
            "NonLiftable"
            f"({self.non_liftable_query})"
        )


def generate_probabilistic_symbol_table_for_query(
    cpl_program, query_predicate
):
    """Generate a symbol table adding a fresh symbol for
    each probabilistic/deterministic table and wrapping
    it into one of the classes
    DeterministicFactSet, ProbabilisticFactSet, ProbabilisticChoiceSet

    Parameters
    ----------
    cpl_program : neurolang.probabilistic.cplogic.CPLogicProgram
        CPLogic program representation indexing the query's symbols.
    query_predicate : neurolang.logic.LogicOperator
        Logic expression representing a query from where
        to extract the predicates which are represented as Symbols.

    Returns
    -------
    dict or Mapping
        symbol table with relationship symbols of query_predicate.
        The tables has neurolang.expressions.Symbol as keys and
        relational algebra representations as values
    """
    symbol_table = defaultdict(
        lambda: DeterministicFactSet(
            Constant[AbstractSet](
                WrappedNamedRelationalAlgebraFrozenSet(
                    columns=tuple()
                )
            )
        )
    )
    classify_and_wrap_symbols(
        cpl_program.probabilistic_facts(), query_predicate,
        symbol_table, ProbabilisticFactSet
    )

    classify_and_wrap_symbols(
        cpl_program.probabilistic_choices(), query_predicate,
        symbol_table, ProbabilisticChoiceSet
    )

    classify_and_wrap_symbols(
        cpl_program.extensional_database(), query_predicate,
        symbol_table, lambda s, _: DeterministicFactSet(s)
    )

    return symbol_table


def classify_and_wrap_symbols(
    ra_set_dict, query_predicate, symbol_table, wrapper
):
    for predicate_symbol, facts in ra_set_dict.items():
        if predicate_symbol not in query_predicate._symbols:
            continue

        fresh_symbol = Symbol.fresh()
        symbol_table[predicate_symbol] = wrapper(
            fresh_symbol,
            Constant[ColumnInt](ColumnInt(0))
        )
        symbol_table[fresh_symbol] = facts


class GetProbabilisticSetAtom(PatternWalker):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    @add_match(
        UnaryRelationalAlgebraOperation,
        lambda expression: not isinstance(
            expression,
            (
                DeterministicFactSet,
                ProbabilisticFactSet,
                ProbabilisticChoiceSet
            )
        )
    )
    def projection(self, expression):
        return self.walk(expression.relation)

    @add_match(Symbol)
    def resolve_symbol(self, expression):
        if expression in self.symbol_table:
            return self.walk(self.symbol_table[expression])
        else:
            return expression

    @add_match(...)
    def default(self, expression):
        return expression


def is_atom_a_deterministic_relation(
    atom: Expression, symbol_table: Mapping[Any, Expression]
) -> bool:
    """Returns if a particular expression is a deterministic relation

    Parameters
    ----------
    atom : Expression
        neurolang Expression
    symbol_table : Mapping[Any, Expression]
        mapping maching symbols to expressions.

    Returns
    -------
    bool
        True if the Expression is a relational algebra expression
        containing a single deterministic relation.
    """
    gpsa = GetProbabilisticSetAtom(symbol_table)
    relation = gpsa.walk(atom.functor)
    return (
        isinstance(relation, DeterministicFactSet)
    )


def is_atom_a_probabilistic_fact_relation(
    atom: Expression, symbol_table: Mapping[Any, Expression]
) -> bool:
    """Returns if a particular expression is a tuple independent
    fact set relation

    Parameters
    ----------
    atom : Expression
        neurolang Expression
    symbol_table : Mapping[Any, Expression]
        mapping maching symbols to expressions.

    Returns
    -------
    bool
        True if the Expression is a relational algebra expression
        containing a single independent fact set relation.
    """
    gpsa = GetProbabilisticSetAtom(symbol_table)
    relation = gpsa.walk(atom.functor)
    return (
        isinstance(relation, ProbabilisticFactSet)
    )


def is_atom_a_probabilistic_choice_relation(
    atom: Expression, symbol_table: Mapping[Any, Expression]
) -> bool:
    """Returns if a particular expression is a choice
    relation

    Parameters
    ----------
    atom : Expression
        neurolang Expression
    symbol_table : Mapping[Any, Expression]
        mapping maching symbols to expressions.

    Returns
    -------
    bool
        True if the Expression is a relational algebra expression
        containing a single choice relation.
    """
    gpsa = GetProbabilisticSetAtom(symbol_table)
    relation = gpsa.walk(atom.functor)
    return (
        isinstance(relation, ProbabilisticChoiceSet)
    )
