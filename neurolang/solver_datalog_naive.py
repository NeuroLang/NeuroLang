'''
Naive implementation of non-typed datalog. There's no optimizations and will be
surely very slow
'''
from itertools import product
from operator import and_
from typing import AbstractSet, Any, Callable, Tuple
from warnings import warn

from .datalog import NULL, UNDEFINED
from .datalog import DatalogProgram as DatalogBasic
from .datalog import (Union, Fact, Implication, NullConstant, Undefined,
                      WrappedRelationalAlgebraSet,
                      extract_logic_free_variables,
                      is_conjunctive_expression,
                      is_conjunctive_expression_with_nested_predicates)
from .expression_walker import TypedSymbolTableEvaluator, add_match
from .expressions import (Constant, Expression, FunctionApplication, Lambda,
                          NeuroLangException, Query, Symbol,
                          is_leq_informative)
from .logic import ExistentialPredicate
from .type_system import Unknown

warn("This module is being deprecated please use the datalog module")


__all__ = [
    "Implication", "Fact",
    "Undefined", "NullConstant", "Unknown",
    "UNDEFINED", "NULL",
    "WrappedRelationalAlgebraSet",
    "DatalogBasic",
    "SolverNonRecursiveDatalogNaive",
    "is_conjunctive_expression",
    "extract_logic_free_variables",
    "is_conjunctive_expression_with_nested_predicates"
]


class SolverNonRecursiveDatalogNaive(
    TypedSymbolTableEvaluator,
    DatalogBasic
):
    '''
    Naive resolution system of Datalog.
    '''
    constant_set_name = '__dl_constants__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protected_keywords.add(self.constant_set_name)
        self.symbol_table[self.constant_set_name] =\
            Constant[AbstractSet[Any]](set())

    @add_match(Fact(FunctionApplication[bool](Symbol, ...)))
    def fact(self, expression):
        # Not sure of using inheritance here (i.e. super), it generates
        # confusing coding patterns between Pattern Matching + Mixins
        # and inheritance-based code.
        expression = super().fact(expression)

        fact = expression.fact
        if all(isinstance(a, Constant) for a in fact.args):
            self.symbol_table[self.constant_set_name].value.update(fact.args)

        return expression

    @add_match(
        Implication(FunctionApplication, ...),
        lambda e: len(
            extract_logic_free_variables(e.antecedent) -
            extract_logic_free_variables(e.consequent)
        ) > 0
    )
    def implication_add_existential(self, expression):
        consequent = expression.consequent
        antecedent = expression.antecedent

        fv_consequent = extract_logic_free_variables(consequent)
        fv_antecedent = (
            extract_logic_free_variables(antecedent) -
            fv_consequent
        )

        new_antecedent = antecedent
        for v in fv_antecedent:
            new_antecedent = ExistentialPredicate[bool](v, new_antecedent)

        out_type = expression.type
        return self.walk(
            Implication[out_type](consequent, new_antecedent)
        )

    @add_match(Implication(
        FunctionApplication[bool](Symbol, ...),
        Expression
    ))
    def statement_intensional(self, expression):
        expression = super().statement_intensional(expression)
        consequent = expression.consequent
        new_constants = {a for a in consequent.args if isinstance(a, Constant)}
        self.symbol_table[self.constant_set_name].value.update(new_constants)
        return expression

    @add_match(
        FunctionApplication(Constant[AbstractSet], ...),
        lambda e: any_arg_is_null(e.args)
    )
    def fa_on_null_constant(self, expression):
        return Constant[bool](False)

    @add_match(
        FunctionApplication(Constant[AbstractSet], (Constant,)),
        lambda exp: not is_leq_informative(exp.args[0].type, Tuple)
    )
    def function_application_edb_notuple(self, expression):
        return self.walk(FunctionApplication(
            expression.functor,
            (Constant(expression.args),)
        ))

    @add_match(
        FunctionApplication[bool](Implication, ...),
        lambda e: all(isinstance(a, Constant) for a in e.args)
    )
    def function_application_idb(self, expression):
        new_lambda_args = []
        new_args = []
        arg_types = []
        for la, a in zip(
            expression.functor.consequent.args,
            expression.args
        ):
            if isinstance(la, Constant):
                if la != a:
                    return Constant[bool](False)
            else:
                new_lambda_args.append(la)
                new_args.append(a)
                arg_types.append(a.type)

        return self.walk(
            FunctionApplication[bool](
                Lambda[Callable[arg_types, bool]](
                    tuple(new_lambda_args),
                    expression.functor.antecedent
                ),
                tuple(new_args)
            )
        )

    @add_match(
        FunctionApplication(Union, ...),
        lambda e: all(
            isinstance(a, Constant) for a in e.args
        )
    )
    def evaluate_datalog_union(self, expression):
        for formula in expression.functor.formulas:
            fa = FunctionApplication[bool](formula, expression.args)
            res = self.walk(fa)
            if isinstance(res, Constant) and res.value is True:
                break
        else:
            return Constant[bool](False)

        return Constant[bool](True)

    @add_match(ExistentialPredicate)
    def existential_predicate_nrndl(self, expression):
        if isinstance(expression.head, Symbol):
            head = (expression.head,)
        elif (
            isinstance(expression.head, Constant) and
            is_leq_informative(expression.head.type, Tuple)
        ):
            head = expression.head.value

        loop = product(
            *((self.symbol_table[self.constant_set_name].value,) * len(head))
        )

        body = Lambda(head, expression.body)
        for args in loop:
            fa = FunctionApplication[bool](body, args)
            res = self.walk(fa)
            if isinstance(res, Constant) and res.value is True:
                break
        else:
            return Constant(False)
        return Constant(True)

    @add_match(
        Query,
        lambda e: (
            extract_logic_free_variables(e.body) >
            get_head_free_variables(e.head)
        )
    )
    def query_introduce_existential(self, expression):
        return self.walk(Query(
            expression.head,
            query_introduce_existential_aux(
                expression.body, get_head_free_variables(expression.head)
            )
        ))

    @add_match(Query)
    def query_resolution(self, expression):
        if isinstance(expression.head, Symbol):
            head = (expression.head,)
        elif isinstance(expression.head, tuple):
            head = expression.head
        elif (
            isinstance(expression.head, Constant) and
            is_leq_informative(expression.head.type, Tuple)
        ):
            head = expression.head.value
        else:
            raise NeuroLangException(
                'Head needs to be a tuple of symbols or a symbol'
            )

        head_type = [arg.type for arg in head]
        constant_set = self.symbol_table[self.constant_set_name].value
        constant_set = constant_set.union({NULL})
        loop = product(*((constant_set, ) * len(head)))
        body = Lambda[Callable[head_type, bool]](head, expression.body)
        result = set()
        for args in loop:
            fa = FunctionApplication[bool](body, args)
            res = self.walk(fa)
            if isinstance(res, Constant) and res.value is True:
                if any_arg_is_null(args):
                    break
                if len(head) == 1:
                    result.add(args[0])
                else:
                    result.add(Constant(args))
        else:
            return Constant[AbstractSet[Any]](result)

        return UNDEFINED


def get_head_free_variables(expression_head):
    if isinstance(expression_head, Symbol):
        head_variables = {expression_head}
    elif isinstance(expression_head, tuple):
        head_variables = set(e for e in expression_head)
    elif (
        isinstance(expression_head, Constant) and
        is_leq_informative(expression_head.type, Tuple)
    ):
        head_variables = set(e for e in expression_head.value)
    else:
        raise NeuroLangException(
            'Head needs to be a tuple of symbols or a symbol'
        )
    return head_variables


def query_introduce_existential_aux(body, head_variables):
    new_body = body
    if isinstance(body, FunctionApplication):
        if (
            isinstance(body.functor, Constant) and
            body.functor.value is and_
        ):
            new_body = FunctionApplication(
                body.functor,
                tuple(
                    query_introduce_existential_aux(arg, head_variables)
                    for arg in body.args
                )
            )
        else:
            fa_free_variables = extract_logic_free_variables(body)
            eq_variables = fa_free_variables - head_variables
            for eq_variable in eq_variables:
                new_body = ExistentialPredicate(
                    eq_variable, new_body
                )
    return new_body


def any_arg_is_null(args):
    return any(
        arg is NULL or (
            isinstance(arg, Constant) and
            is_leq_informative(arg.type, Tuple) and
            any(x is NULL for x in arg.value)
        )
        for arg in args
    )
