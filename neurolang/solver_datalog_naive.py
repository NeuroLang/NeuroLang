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
    Query, ExistentialPredicate, UniversalPredicate
)
from .expression_walker import (
    add_match, PatternWalker, expression_iterator
)


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


class NaiveDatalog(PatternWalker):
    constant_set_name = '__dl_constants__'

    def function_equals(self, a: Any, b: Any) -> bool:
        return a == b

    @add_match(Statement(
        FunctionApplication[bool](Symbol, ...),
        None
    ))
    def statement_extensional(self, expression):
        lhs = expression.lhs

        if any(
            not isinstance(a, Constant)
            for _, a, level in expression_iterator(
                lhs.args, include_level=True
            )
            if level > 0
        ):
            raise NeuroLangException(
                'Facts can only have constants as arguments'
            )

        if lhs.functor.name == self.constant_set_name:
            raise NeuroLangException(
                f'symbol {self.constant_set_name} is protected'
            )

        if lhs.functor.name in self.symbol_table:
            eb = self.symbol_table[lhs.functor.name].expressions
        else:
            eb = tuple()

        if all(isinstance(a, Constant) for a in lhs.args):
            if self.constant_set_name not in self.symbol_table:
                self.symbol_table[self.constant_set_name] = \
                        Constant[AbstractSet[Any]](set())
            self.symbol_table[self.constant_set_name].value.update(lhs.args)

            value = {Constant(lhs.args)}

            for i, block in enumerate(eb):
                if isinstance(block, Constant):
                    fact_set = block
                    break
            else:
                fact_set = Constant[AbstractSet[Any]](set())
                eb = eb + (fact_set,)
                i = len(eb) - 1

            fact_set.value |= value

        elif all(isinstance(a, Symbol) for a in lhs.args):
            equalities = []
            parameters = [
                Symbol[a.type](f'a{i}') for i, a in enumerate(lhs.args)
            ]
            for i, a in enumerate(lhs.args[:-1]):
                sa = parameters[i]
                for j, b in enumerate(lhs.args[i + 1:]):
                    if a == b:
                        sb = parameters[j + i + 1]
                        equalities.append(Symbol('equals')(sa, sb))

            if len(equalities) == 0:
                functor = Constant(True)
            else:
                functor = equalities[0]
                for eq in equalities[1:]:
                    functor = functor & eq
            lambda_ = Lambda(tuple(parameters), functor)
            eb = eb + (lambda_,)
        else:
            raise NeuroLangException(
                "Can't mix contants and symbols on the left hand side "
                "of a definition"
            )

        self.symbol_table[lhs.functor.name] = ExpressionBlock(eb)

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
            eb = self.symbol_table[lhs.functor.name].expressions
        else:
            eb = tuple()

        lambda_ = Lambda(lhs.args, expression.rhs)
        eb = eb + (lambda_,)

        self.symbol_table[lhs.functor.name] = ExpressionBlock(eb)

    @add_match(FunctionApplication(ExpressionBlock, ...))
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
            fa = FunctionApplication(body, args)
            res = self.walk(fa)
            if isinstance(res, Constant) and res.value is True:
                if len(head) > 1:
                    result.add(Constant(args))
                else:
                    result.add(args[0])

        return Constant[AbstractSet[Any]](result)
