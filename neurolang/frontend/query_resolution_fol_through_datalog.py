from typing import AbstractSet, Tuple

from .. import expressions as exp
from .. import logic
from ..datalog.chase import Chase as Chase_
from ..datalog.chase.negation import NegativeFactConstraints
from ..logic.horn_clauses import fol_query_to_datalog_program
from ..type_system import Unknown
from ..datalog.expressions import TranslateToLogic
from ..expression_walker import ExpressionWalker, add_match
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


class QueryBuilderFirstOrderThroughDatalog(
    RegionMixin, NeuroSynthMixin, QueryBuilderBase
):
    def __init__(self, solver, chase_class=Chase):
        super().__init__(solver, logic_programming=True)
        self.type_predicate_symbols = dict()
        self.chase_class = chase_class

    def execute_expression(self, expression, name=None):
        if name is None:
            name = str(uuid1())

        __import__("pdb").set_trace()

        program = TranslateQueryExpressionToDatalog(self, name).walk(
            expression
        )
        self.solver.walk(program)
        solution = self.chase_class(self.solver).build_chase_solution()
        solution_set = solution.get(name, exp.Constant(set()))

        self.symbol_table[exp.Symbol[solution.type](name)] = solution_set
        return Symbol(self, name)

    def add_tuple_set(self, iterable, type_=Unknown, name=None):
        if name is None:
            name = str(uuid1())

        if isinstance(type_, tuple):
            type_ = Tuple[type_]
        symbol = exp.Symbol[AbstractSet[type_]](name)
        self.solver.add_extensional_predicate_from_tuples(
            symbol, iterable, type_=type_
        )

        return Symbol(self, name)

    def type_predicate_for(self, var):
        type_ = var.type
        if type_ not in self.type_predicate_symbols:
            self.type_predicate_symbols[type_] = exp.Symbol.fresh()
            # maybe ... = exp.Symbol[bool].fresh()
        return self.type_predicate_symbols[type_](var)

    def query(self, head, predicate):

        if isinstance(head, tuple):
            symbols = ()
            for e in head:
                symbols += (e.expression,)
            head = exp.Constant(symbols)
        else:
            head = head.expression
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


class TranslateQueryExpressionToDatalog(TranslateToLogic, ExpressionWalker):
    def __init__(self, query_builder, name):
        self.name = name
        self.query_builder = query_builder

    @add_match(exp.Query)
    def query(self, query):
        head_symbol = exp.Symbol(self.name)
        args = self._get_symbols(query.head)
        head = head_symbol(*args)
        body = self.walk(query.body)
        for var in args:
            body = self._type_restrict(body, var)
        return fol_query_to_datalog_program(head, body)

    def _get_symbols(self, head):
        v = head.value
        if isinstance(v, tuple):
            symbols = v
        elif isinstance(v, Symbol):
            symbols = (v,)
        else:
            __import__("pdb").set_trace()
            raise Exception("Unknown value in query head")
        return symbols

    def _type_restrict(self, body, var):
        pred = self.query_builder.type_predicate_for(var)
        return logic.Conjunction((pred, body))
