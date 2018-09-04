from typing import AbstractSet, Any, Tuple

from .expressions import (
    NeuroLangException, Definition, ExistentialPredicate, Constant, Symbol,
    ExpressionBlock, Query, FunctionApplication
)
from .solver_datalog_naive import NaiveDatalog, DatalogBasic
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
        raise NeuroLangException(
            'Query head must be symbol or tuple of symbols'
        )


def _get_eq_variables(expression):
    if not isinstance(expression, ExistentialPredicate):
        return set()
    else:
        return {expression.head} | _get_eq_variables(expression.body)


class Implication(Definition):
    def __init__(self, consequent, antecedent):
        self.consequent = consequent
        self.antecedent = antecedent

    def __repr__(self):
        return 'Implication{{{} \u2190 {}}}'.format(
            repr(self.consequent), repr(self.antecedent)
        )


def _get_existential_implication_fa(expression):
    if not isinstance(expression, Implication):
        raise NeuroLangException('Must be called on an Implication')
    elif not isinstance(expression.consequent, ExistentialPredicate):
        raise NeuroLangException(
            'Implication consequent must contain existential predicate'
        )
    else:
        result = expression.consequent
        while isinstance(result, ExistentialPredicate):
            result = result.body
        if (
            not isinstance(result, FunctionApplication) or
            not all(isinstance(arg, Symbol) for arg in result.args)
        ):
            raise Exception(
                'Implication e-quantified consequent must be a '
                'function application on symbols'
            )
        return result


class ExistentialDatalog(DatalogBasic):
    @add_match(
        Implication(ExistentialPredicate(..., FunctionApplication), ...),
        # ensure the predicate is a simple function application on symbols
        lambda expression: all(
            isinstance(arg, Symbol)
            for arg in expression.consequent.body.args
        )
    )
    def existential_predicate_in_head(self, expression):
        '''
        Add implication with a \u2203-quantified function application
        consequent
        '''
        eq_variable = expression.consequent.head
        if not isinstance(eq_variable, Symbol):
            raise NeuroLangException(
                '\u2203-quantified variable must be a symbol'
            )
        if eq_variable in expression.antecedent._symbols:
            raise NeuroLangException(
                '\u2203-quantified variable cannot occur in antecedent'
            )
        if eq_variable not in expression.consequent.body._symbols:
            raise NeuroLangException(
                "\u2203-quantified variable must occur in consequent's body"
            )
        predicate_name = expression.consequent.body.functor.name
        if predicate_name in self.symbol_table:
            expressions = self.symbol_table[predicate_name].expressions
        else:
            expressions = tuple()
        expressions += (expression, )
        self.symbol_table[predicate_name] = ExpressionBlock(expressions)
        return expression


class SolverExistentialDatalog(NaiveDatalog, ExistentialDatalog):

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
        if len(eq_variables) > 1:
            raise NotImplementedError(
                'Multiple \u2203-quantified variable currently unsupported'
            )
        new_body = expression.body
        for eq_variable in eq_variables:
            new_body = ExistentialPredicate(eq_variable, new_body)
        return self.walk(Query(expression.head, new_body))

    @add_match(FunctionApplication(Implication(ExistentialPredicate, ...)))
    def existential_consequent_resolution(self, expression):
        eq_variables = _get_eq_variables(expression.functor.consequent)
        fa = _get_existential_implication_fa(expression.functor)
        if not isinstance(fa.functor, Symbol):
            raise NeuroLangException(
                'E-quantified consequent must be a '
                'function application of a symbol'
            )
        fa_name = fa.functor.name
        eq_expressions = (
            e for e in self.symbol_table[fa_name].expressions if (
                isinstance(e, Implication) and
                isinstance(e.consequent, ExistentialPredicate)
            )
        )
        result = False
        for eq_exp in eq_expressions:
            result |= self.walk(FunctionApplication(eq_exp, expression.args))
        return result
