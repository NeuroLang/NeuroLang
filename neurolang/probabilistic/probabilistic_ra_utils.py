from neurolang.probabilistic.expression_processing import is_probabilistic_fact
from ..expressions import Constant, Symbol
from ..relational_algebra import RelationalAlgebraOperation, ColumnInt


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
    symbol_table = dict()
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
