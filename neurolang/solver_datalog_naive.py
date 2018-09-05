'''
Naive implementation of non-typed datalog. There's no optimizations and will be
surely very slow
'''
from typing import AbstractSet, Any, Tuple
from itertools import product
from operator import and_

from .expressions import (
    FunctionApplication, Constant, NeuroLangException, is_subtype,
    Statement, Symbol, Lambda, ExpressionBlock, Expression,
    Query, ExistentialPredicate, UniversalPredicate, Quantifier
)
from .expression_walker import (
    add_match, PatternWalker, expression_iterator,
)


class Fact(Statement):
    def __init__(self, lhs):
        super().__init__(lhs, Constant(True))

    @property
    def fact(self):
        return self.lhs

    def __repr__(self):
        return f'{self.fact}: {self.type} :- True'


class DatalogBasic(PatternWalker):
    '''
    Implementation of Datalog grammar in terms of
    Intermediate Representations. No query resolution implemented.
    '''
    constant_set_name = '__dl_constants__'

    def function_equals(self, a: Any, b: Any) -> bool:
        return a == b

    @add_match(Fact(FunctionApplication[bool](Symbol, ...)))
    def fact(self, expression):
        fact = expression.fact

        if any(
            not isinstance(a, Constant)
            for _, a, level in expression_iterator(
                fact.args, include_level=True
            )
            if level > 0
        ):
            raise NeuroLangException(
                'Facts can only have constants as arguments'
            )

        if fact.functor.name == self.constant_set_name:
            raise NeuroLangException(
                f'symbol {self.constant_set_name} is protected'
            )

        if fact.functor.name not in self.symbol_table:
            self.symbol_table[fact.functor.name] = \
                Constant[AbstractSet[Any]](set())
        fact_set = self.symbol_table[fact.functor.name]

        if isinstance(fact_set, ExpressionBlock):
            raise NeuroLangException(
                f'{fact.functor.name} has been previously '
                'define as intensional predicate.'
            )

        if all(isinstance(a, Constant) for a in fact.args):
            if self.constant_set_name not in self.symbol_table:
                self.symbol_table[self.constant_set_name] = \
                        Constant[AbstractSet[Any]](set())
            self.symbol_table[self.constant_set_name].value.update(fact.args)
            fact_set.value.add(Constant(fact.args))

        return expression

    @add_match(Statement(
        FunctionApplication[bool](Symbol, ...),
        Constant[bool](True)
    ))
    def statement_extensional(self, expression):
        return self.fact(Fact[expression.type](expression.lhs))

    @add_match(Statement(
        FunctionApplication[bool](Symbol, ...),
        Expression
    ))
    def statement_intensional(self, expression):
        lhs = expression.lhs
        if lhs.functor.name == self.constant_set_name:
            raise NeuroLangException(
                f'symbol {self.constant_set_name} is protected'
            )

        rhs = expression.rhs
        if not is_conjunctive_expression(rhs):
            raise NeuroLangException(
                f'Expression {rhs} is not conjunctive'
            )

        lhs_symbols = lhs._symbols - lhs.functor._symbols

        if not lhs_symbols.issubset(rhs._symbols):
            raise NeuroLangException(
                "All variables on the left need to be on the right"
            )

        if lhs.functor.name in self.symbol_table:
            value = self.symbol_table[lhs.functor.name]
            if (
                isinstance(value, Constant) and
                is_subtype(value.type, AbstractSet)
            ):
                raise NeuroLangException(
                    'f{lhs.functor.name} has been previously '
                    'defined as Fact or extensional database.'
                )
            eb = self.symbol_table[lhs.functor.name].expressions
        else:
            eb = tuple()

        lambda_ = Lambda(lhs.args, rhs)
        eb = eb + (lambda_,)

        self.symbol_table[lhs.functor.name] = ExpressionBlock(eb)

        return expression

    def intensional_database(self):
        return self.symbol_table.symbols_by_type(ExpressionBlock)

    def extensional_database(self):
        ret = self.symbol_table.symbols_by_type(AbstractSet)
        del ret[self.constant_set_name]
        return ret


class NaiveDatalog(DatalogBasic):
    '''
    Naive resolution system of Datalog.
    '''
    @add_match(Statement(
        FunctionApplication[bool](Symbol, ...),
        Expression
     ),
        lambda e: len(
            extract_datalog_free_variables(e.rhs) -
            extract_datalog_free_variables(e.lhs)
        ) > 0
    )
    def statement_intensional_add_existential(self, expression):
        lhs = expression.lhs
        rhs = expression.rhs
        fv = (
            extract_datalog_free_variables(rhs) -
            extract_datalog_free_variables(lhs)
        )

        if len(fv) > 0:
            for v in fv:
                rhs = ExistentialPredicate[bool](v, rhs)
        return self.walk(Statement[expression.type](lhs, rhs))

    @add_match(
        FunctionApplication(ExpressionBlock, ...),
        lambda e: all(
            isinstance(a, Constant) for a in e.args
        )
    )
    def evaluate_datalog_expression(self, expression):
        for exp in expression.functor.expressions:
            if (
                isinstance(exp, Lambda) and
                len(exp.args) != len(expression.args)
            ):
                continue

            fa = FunctionApplication(exp, expression.args)

            res = self.walk(fa)
            if isinstance(res, Constant) and res.value is True:
                break
        else:
            return Constant(False)

        return Constant(True)

    @add_match(UniversalPredicate)
    def universal_predicate_ndl(self, expression):
        if isinstance(expression.head, Symbol):
            head = (expression.head,)
        elif (
            isinstance(expression.head, Constant) and
            is_subtype(expression.head.type, Tuple)
        ):
            head = expression.head.value

        loop = product(
            *((self.symbol_table[self.constant_set_name].value,) * len(head))
        )

        body = Lambda(head, expression.body)
        for args in loop:
            fa = FunctionApplication(body, args)
            res = self.walk(fa)
            if isinstance(res, Constant) and res.value is False:
                break
        else:
            return Constant(True)
        return Constant(False)

    @add_match(ExistentialPredicate)
    def existential_predicate_ndl(self, expression):
        if isinstance(expression.head, Symbol):
            head = (expression.head,)
        elif (
            isinstance(expression.head, Constant) and
            is_subtype(expression.head.type, Tuple)
        ):
            head = expression.head.value

        loop = product(
            *((self.symbol_table[self.constant_set_name].value,) * len(head))
        )

        body = Lambda(head, expression.body)
        for args in loop:
            fa = FunctionApplication(body, args)
            res = self.walk(fa)
            if isinstance(res, Constant) and res.value is True:
                break
        else:
            return Constant(False)
        return Constant(True)

    @add_match(Query)
    def query_resolution(self, expression):
        if isinstance(expression.head, Symbol):
            head = (expression.head,)
        elif isinstance(expression.head, tuple):
            head = expression.head
        elif (
            isinstance(expression.head, Constant) and
            is_subtype(expression.head.type, Tuple)
        ):
            head = expression.head.value
        else:
            raise NeuroLangException(
                'Head needs to be a tuple of symbols or a symbol'
            )

        loop = product(
            *((self.symbol_table[self.constant_set_name].value,) * len(head))
        )

        body = Lambda(head, expression.body)

        result = set()

        for args in loop:
            if len(head) == 1:
                args = (Constant(args),)
            fa = FunctionApplication(body, args)
            res = self.walk(fa)
            if isinstance(res, Constant) and res.value is True:
                if len(head) > 1:
                    result.add(Constant(args))
                else:
                    result.add(args[0].value[0])

        return Constant[AbstractSet[Any]](result)


def is_conjunctive_expression(expression):
    return all(
        not isinstance(exp, FunctionApplication) or
        (
            isinstance(exp, FunctionApplication) and
            (
                (
                    isinstance(exp.functor, Constant) and
                    exp.functor.value is and_
                ) or all(
                    not isinstance(arg, FunctionApplication)
                    for arg in exp.args
                )
            )
        )
        for _, exp in expression_iterator(expression)
    )


class ExtractDatalogFreeVariablesWalker(PatternWalker):
    @add_match(FunctionApplication)
    def extract_variables_fa(self, expression):
        functor = expression.functor
        args = expression.args
        if isinstance(functor, Constant) and functor.value == and_:
            return set().union(*(self.walk(a) for a in args))

        variables = set()
        for a in expression.args:
            if isinstance(a, Symbol):
                variables.add(a)
            elif isinstance(a, Constant):
                pass
            else:
                raise NeuroLangException('Not a Datalog function application')

        return variables

    @add_match(Quantifier)
    def extract_variables_q(self, expression):
        variables = self.walk(expression.body)
        variables -= expression.head._symbols

        return variables

    @add_match(Statement)
    def extract_variables_s(self, expression):
        return self.walk(expression.rhs) - self.walk(expression.lhs)

    @add_match(...)
    def _(self, expression):
        return set()


def extract_datalog_free_variables(expression):
    '''extract variables from expression knowing that it's in Datalog format'''
    efvw = ExtractDatalogFreeVariablesWalker()
    return efvw.walk(expression)


def extract_datalog_predicates(expression):
    """
    extract predicates from expression
    knowing that it's in Datalog format
    """
    res = set()
    for _, exp in expression_iterator(expression):
        if (
            isinstance(exp, FunctionApplication) and
            exp.functor.value is not and_
        ):
            res.add(exp)
    return res
