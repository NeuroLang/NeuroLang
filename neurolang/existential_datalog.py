from typing import AbstractSet, Any, Tuple

from .expressions import NeuroLangException as NLE
from .expressions import (
    Definition, Statement, ExistentialPredicate, Constant, Symbol,
    ExpressionBlock, Query, FunctionApplication
)
from .solver_datalog_naive import NaiveDatalog
from .expression_pattern_matching import add_match


def _get_query_head(query):
    if isinstance(query.head, Symbol):
        return query.head
    elif (
        isinstance(query.head, Constant) and
        is_subtype(query.head.type, Tuple) and
        all(isinstance(s, Symbol) for s in query.head)
    ):
        return query.head.value
    else:
        raise NLE('Query head must be symbol or tuple of symbols')


class ExistentialDatalog(NaiveDatalog):
    @add_match(
        Statement(ExistentialPredicate, ...),
        # ensure the predicate is a simple function application on symbols
        lambda expression: all(
            isinstance(arg, Symbol) and
            not isinstance(arg, FunctionApplication)
            for arg in expression.lhs.body.args
        )
    )
    def existential_predicate_in_head(self, expression):
        '''
        Add statement with a lhs existential predicate to symbol table
        '''
        eq_variable = expression.lhs.head
        if not isinstance(eq_variable, Symbol):
            raise NLE('\u2203-quantified variable must be a symbol')
        if eq_variable in expression.rhs._symbols:
            raise NLE('\u2203-quantified variable cannot occur in rhs')
        if eq_variable not in expression.lhs.body._symbols:
            raise NLE('\u2203-quantified variable must occur in lhs body')
        predicate_name = expression.lhs.body.functor.name
        if predicate_name in self.symbol_table:
            expressions = self.symbol_table[predicate_name].expressions
        else:
            expressions = tuple()
        expressions += (expression, )
        self.symbol_table[predicate_name] = ExpressionBlock(expressions)
        return expression


class SolverExistentialDatalog(ExistentialDatalog):

    @add_match(
        Query(Symbol, FunctionApplication),
        lambda expression: all(
            isinstance(arg, Symbol)
            for arg in expression.body.args
        ) and (
            len(set(expression.body.args) - {expression.head}) > 0
        )
    )
    def query_introduce_existential(self, expression):
        eq_variables = set(expression.body.args) - {expression.head}
        new_body = expression.body
        for eq_variable in eq_variables:
            new_body = ExistentialPredicate(eq_variable, new_body)
        return self.walk(Query(expression.head, new_body))

    @add_match(Query(..., ExistentialPredicate(..., FunctionApplication)))
    def query_resolution(self, query):
        q_head = _get_query_head(query)
        q_predicate = query.body.body
        # index of e-quantified variable in query
        q_eq_idx = q_predicate.args.index(query.body.head)
        predicate_name = q_predicate.functor.name
        if (
            predicate_name in self.symbol_table and
            isinstance(self.symbol_table[predicate_name], ExpressionBlock)
        ):
            eq_expressions = (
                e for e in self.symbol_table[predicate_name].expressions if (
                    isinstance(e, Statement) and
                    isinstance(e.lhs, ExistentialPredicate)
                )
            )
            found_matching_existential_statement = False
            result = set()
            for e in eq_expressions:
                # index of e-quantified variable
                # in existential intensional rule
                eq_idx = e.lhs.body.args.index(e.lhs.head)
                if q_eq_idx == eq_idx:
                    map_q_arg = {
                        q_predicate.args[i]: e.lhs.body.args[i]
                        for i in range(len(q_predicate.args))
                    }
                    if isinstance(q_head, Symbol):
                        new_q_head = map_q_arg[q_head]
                    else:
                        new_q_head = (map_q_arg[s] for s in q_head)
                    result |= self.walk(Query(new_q_head, e.rhs)).value
                    found_matching_existential_statement = True
            if not found_matching_existential_statement:
                return query
            return Constant[AbstractSet[Any]](result)
        else:
            return super().walk(query)
