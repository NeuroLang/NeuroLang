from typing import AbstractSet, Tuple
from collections import defaultdict

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
    add_match,
    ExpressionBasicEvaluator,
)
from ..logic.transformations import RemoveUniversalPredicates
from ..region_solver import RegionSolver
from .query_resolution import QueryBuilderBase, RegionMixin, NeuroSynthMixin
from .query_resolution_expressions import (
    All,
    Exists,
    Query,
    Symbol,
)


__all__ = ["QueryBuilderFirstOrderThroughDatalog"]


class Chase(NegativeFactConstraints, Chase_):
    pass


class RegionFrontendFolThroughDatalogSolver(
    RegionSolver,
    DatalogWithAggregationMixin,
    DatalogProgramNegation,
    ExpressionBasicEvaluator,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def isin(x, y):
            return exp.Constant(x) in y

        self.symbol_table[exp.Symbol("isin")] = exp.Constant(isin)


class QueryBuilderFirstOrderThroughDatalog(
    RegionMixin, NeuroSynthMixin, QueryBuilderBase
):
    def __init__(self, solver, chase_class=Chase):
        if solver is None:
            solver = RegionFrontendFolThroughDatalogSolver()
        super().__init__(solver, logic_programming=True)
        self.type_predicates = defaultdict(set)
        self.chase_class = chase_class

    # @profile
    def execute_expression(self, expression, name=None):
        if not isinstance(expression, exp.Query):
            raise NotImplementedError(
                f"{self.__class__.__name__} can only evaluate Query "
                f"expressions, {expression.__class__.__name__} given"
            )

        symbol, program, type_predicate_symbols = self._get_program_from_query(
            expression, name
        )
        self.solver.walk(program)
        self._populate_type_predicates(type_predicate_symbols)

        rules = logic.Union(program.expressions)
        solution = self.chase_class(
            self.solver, rules=rules
        ).build_chase_solution()
        solution_set = solution.get(symbol, exp.Constant(set()))

        if symbol in self.symbol_table:
            self.del_symbol(symbol.name)
        self.symbol_table[symbol] = solution_set

        return Symbol(self, symbol.name)

    def _populate_type_predicates(self, type_predicate_symbols):
        for s in type_predicate_symbols:
            type_ = s.restricted_type
            values = self._get_values_for_type(type_)
            self.add_tuple_set(values, type_, s.name)

    def _get_values_for_type(self, type_):
        values = set()
        for t, v in self.type_predicates.items():
            if issubclass(t, type_):
                values |= v
        return values

    def _get_program_from_query(self, query, name=None):
        walker = RestrictVariablesByType(self)
        query = walker.walk(query)
        type_predicate_symbols = walker.type_predicate_symbols
        args = _get_head_symbols(query.head)
        type_ = self._head_type(args)

        st = exp.Symbol[type_]
        head_symbol = st(name) if name else st.fresh()
        head = head_symbol(*args)
        return (
            head_symbol,
            fol_query_to_datalog_program(head, query.body),
            type_predicate_symbols,
        )

    def _head_type(self, args):
        if len(args) == 1:
            type_ = args[0].type
        else:
            type_ = Tuple[tuple(map(lambda s: s.type, args))]
        return AbstractSet[type_]

    def add_symbol(self, value, name=None):
        type_ = value.__class__
        if not isinstance(value, Symbol):
            self.type_predicates[type_].add(value)
        return super().add_symbol(value, name)

    def add_tuple_set(self, iterable, type_=Unknown, name=None):
        items = list(map(self._enforceTuple, map(self._getValue, iterable)))

        if isinstance(type_, tuple):
            type_ = Tuple[type_]
        elif not is_leq_informative(type_, Tuple):
            type_ = Tuple[type_]

        self._register_values_for_type(items, type_)

        st = exp.Symbol[type_]
        symbol = st(name) if name else st.fresh()
        self.solver.add_extensional_predicate_from_tuples(
            symbol, items, type_=type_
        )

        return Symbol(self, symbol.name)

    def _register_values_for_type(self, items, type_):
        for row in items:
            self._add_row_to_type_predicates(row, type_)

    def _add_row_to_type_predicates(self, row, type_):
        for e, t in zip(row, type_.__args__):
            if not isinstance(e, Symbol):
                self.type_predicates[t].add(e)

    def _getValue(self, x):
        if isinstance(x, Symbol):
            x = x.value
        if isinstance(x, exp.Constant):
            x = x.value
        return x

    def _enforceTuple(self, x):
        try:
            iter(x)
        except TypeError:
            x = (x,)
        return tuple(x)

    def type_predicate_symbol(self, type_):
        s = exp.Symbol("type_of(" + str(type_) + ")")
        s = s.cast(AbstractSet[type_])
        s.restricted_type = type_
        return s

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
        self.type_predicate_symbols = set()

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
        pred = self.query_builder.type_predicate_symbol(var.type)
        self.type_predicate_symbols.add(pred)
        return logic.Conjunction((pred(var), body))


def _contains_type_restrictions(body):
    return isinstance(body, logic.Conjunction) and any(
        _is_type_restriction(pred) for pred in body.formulas
    )


def _is_type_restriction(pred):
    return isinstance(pred, exp.FunctionApplication) and hasattr(
        pred.functor, "restricted_type"
    )


def _get_head_symbols(head):
    v = head.value
    if isinstance(v, tuple):
        symbols = v
    elif isinstance(v, exp.Symbol):
        symbols = (v,)
    else:
        raise Exception("Unknown value in query head")
    return symbols
