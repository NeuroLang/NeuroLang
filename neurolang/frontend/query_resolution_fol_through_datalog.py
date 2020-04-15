from typing import AbstractSet, Tuple, Callable

from .. import expressions as exp
from .. import logic
from ..datalog.aggregation import Chase as Chase_
from ..datalog.aggregation import DatalogWithAggregationMixin
from ..datalog.chase.negation import NegativeFactConstraints
from ..datalog.negation import DatalogProgramNegation

from ..logic.horn_clauses import fol_query_to_datalog_program
from ..type_system import Unknown, is_leq_informative
from ..datalog.expressions import TranslateToLogic
from ..expression_walker import (
    ExpressionWalker,
    add_match,
    ExpressionBasicEvaluator,
)
from ..logic.transformations import RemoveUniversalPredicates
from ..region_solver import RegionSolver
from .query_resolution import QueryBuilderBase, RegionMixin, NeuroSynthMixin
from .query_resolution_expressions import (
    All,
    Exists,
    Expression,
    Query,
    Symbol,
)
from uuid import uuid1


__all__ = ["QueryBuilderFirstOrderThroughDatalog"]


class Chase(NegativeFactConstraints, Chase_):
    pass


class RegionFrontendFolThroughDatalogSolver(
    RegionSolver,
    DatalogWithAggregationMixin,
    DatalogProgramNegation,
    ExpressionBasicEvaluator,
):
    pass


class QueryBuilderFirstOrderThroughDatalog(
    RegionMixin, NeuroSynthMixin, QueryBuilderBase
):
    def __init__(self, solver, chase_class=Chase):
        if solver is None:
            solver = RegionFrontendFolThroughDatalogSolver()
        super().__init__(solver, logic_programming=True)
        self.type_predicate_symbols = dict()
        self.chase_class = chase_class

    def execute_expression(self, expression, name=None):
        if name is None:
            name = str(uuid1())

        if not isinstance(expression, exp.Query):
            raise NotImplementedError(
                f"{self.__class__.__name__} can only evaluate Query "
                f"expressions, {expression.__class__.__name__} given"
            )
        program = self._get_program_from_query(expression, name)
        self.solver.walk(program)
        self._populate_type_predicates()

        solution = self.chase_class(self.solver).build_chase_solution()
        solution_set = solution.get(name, exp.Constant(set()))

        self.symbol_table[exp.Symbol[solution_set.type](name)] = solution_set
        return Symbol(self, name)

    def _populate_type_predicates(self):
        for type_, pred_symbol in self.type_predicate_symbols.items():
            symbols = self.symbol_table.symbols_by_type(type_).values()
            self.add_tuple_set(symbols, type_, pred_symbol.name)

    def _get_program_from_query(self, query, name):
        query = RestrictVariablesByType(self).walk(query)
        args = _get_head_symbols(query.head)
        type_ = self._head_type(args)
        head_symbol = exp.Symbol[type_](name)
        head = head_symbol(*args)
        return fol_query_to_datalog_program(head, query.body)

    def _head_type(self, args):
        if len(args) == 1:
            type_ = args[0].type
        else:
            type_ = Tuple[tuple(map(lambda s: s.type, args))]
        return AbstractSet[type_]

    def add_tuple_set(self, iterable, type_=Unknown, name=None):
        if name is None:
            name = str(uuid1())

        items = list(map(self._enforceTuple, map(self._getValue, iterable)))

        if isinstance(type_, tuple):
            type_ = Tuple[type_]
        elif not is_leq_informative(type_, Tuple):
            type_ = Tuple[type_]

        # This may not be the best way of doing this but doing
        # so I ensure that they are returned by symbols_by_type
        self._add_individual_items_to_symbol_table(items, type_)

        symbol = exp.Symbol[AbstractSet[type_]](name)
        self.solver.add_extensional_predicate_from_tuples(
            symbol, items, type_=type_
        )

        return Symbol(self, name)

    def _add_individual_items_to_symbol_table(self, items, type_):
        for row in items:
            self._add_row_to_symbol_table(row, type_)

    def _add_row_to_symbol_table(self, row, type_):
        for e, t in zip(row, type_.__args__):
            if not isinstance(e, Symbol):
                s, c = self._create_symbol_and_get_constant(e, t)
                self.symbol_table[s] = c

    def _getValue(self, x):
        if isinstance(x, exp.Constant):
            x = x.value
        return x

    def _enforceTuple(self, x):
        if not hasattr(x, "__iter__"):
            x = (x,)
        return tuple(x)

    def type_predicate_for(self, var):
        type_ = var.type
        if type_ not in self.type_predicate_symbols:
            s = exp.Symbol("type_of(" + str(type_) + ")")
            s = s.cast(Callable[[type_], bool])
            s.is_type_symbol_for = type_
            self.type_predicate_symbols[type_] = s
        return self.type_predicate_symbols[type_](var)

    def query(self, head, predicate):
        if isinstance(head, tuple):
            symbols = ()
            for e in head:
                symbols += (e.expression,)
            head = exp.Constant(symbols)
        else:
            head = exp.Constant(head.expression)
        return Query(
            self,
            exp.Query[AbstractSet[head.type]](head, predicate.expression),
            head,
            predicate,
        )

    def exists(self, symbol, predicate):
        return Exists(
            self,
            logic.ExistentialPredicate[bool](
                symbol.expression, predicate.expression
            ),
            symbol,
            predicate,
        )

    def all(self, symbol, predicate):
        return All(
            self,
            logic.UniversalPredicate[bool](
                symbol.expression, predicate.expression
            ),
            symbol,
            predicate,
        )


class RestrictVariablesByType(TranslateToLogic, RemoveUniversalPredicates):
    def __init__(self, query_builder):
        self.query_builder = query_builder

    @add_match(
        exp.Query, lambda q: not _contains_type_restrictions(q.body),
    )
    def query(self, query):
        body = self.walk(query.body)
        for var in _get_head_symbols(query.head):
            body = self._type_restrict(body, var)
        return exp.Query[AbstractSet[query.head.type]](query.head, body)

    @add_match(
        logic.ExistentialPredicate,
        lambda ep: not _contains_type_restrictions(ep.body),
    )
    def existential(self, ep):
        return self.walk(
            logic.ExistentialPredicate(
                ep.head, self._type_restrict(ep.body, ep.head)
            )
        )

    def _type_restrict(self, body, var):
        pred = self.query_builder.type_predicate_for(var)
        return logic.Conjunction((pred, body))


def _contains_type_restrictions(body):
    if isinstance(body, logic.Conjunction):
        for pred in body.formulas:
            if isinstance(pred, exp.FunctionApplication) and hasattr(
                pred.functor, "is_type_symbol_for"
            ):
                return True
    return False


def _get_head_symbols(head):
    v = head.value
    if isinstance(v, tuple):
        symbols = v
    elif isinstance(v, exp.Symbol):
        symbols = (v,)
    else:
        raise Exception("Unknown value in query head")
    return symbols
