import typing
from operator import (
    invert, and_, or_,
)

from ..expressions import (
    Constant,
    FunctionApplication, ExistentialPredicate, Query,
    unify_types, is_leq_informative, Unknown
)
from ..expression_walker import (
    add_match,
    ReplaceSymbolWalker,
)
from ..solver import GenericSolver


T = typing.TypeVar('T')


class Predicate(FunctionApplication):
    def __repr__(self):
        r = 'P{{{}: {}}}'.format(self.functor, self.__type_repr__)
        if self.args is ...:
            r += '(...)'
        elif self.args is not None:
            r += (
                '(' +
                ', '.join(repr(arg) for arg in self.args)
            )
        if hasattr(self, 'kwargs') and self.kwargs is not None:
            r += ', '.join(
                repr(k) + '=' + repr(v)
                for k, v in self.kwargs.items()
            )
        r += ')'
        return r


class FiniteDomain(object):
    pass


class FiniteDomainSet(frozenset):
    pass


class SetBasedSolver(GenericSolver[T]):
    """
    A predicate `in <set>` which results in the `<set>` given as parameter
    `and` and `or` operations between sets which are disjunction and
    conjunction.
    """

    def function_in(
        self, argument: typing.AbstractSet[T]
    )->typing.AbstractSet[T]:
        return argument

    @add_match(
        FunctionApplication(Constant(invert), (Constant[typing.AbstractSet],)),
        lambda expression: isinstance(
            expression.args[0].value,
            FiniteDomainSet
        )
    )
    def rewrite_finite_domain_inversion(self, expression):
        set_constant = expression.args[0]
        set_type = set_constant.type
        set_value = set_constant.value
        result = FiniteDomainSet(
            (
                v.value for v in
                self.symbol_table.symbols_by_type(
                    set_type.__args__[0]
                ).values()
                if v not in set_value
            ),
            type_=set_type,
        )
        return self.walk(Constant[set_type](result))

    @add_match(
        FunctionApplication(
            Constant(...),
            (Constant[typing.AbstractSet], Constant[typing.AbstractSet])
        ),
        lambda expression: expression.functor.value in (or_, and_)
    )
    def rewrite_and_or(self, expression):
        f = expression.functor.value
        a_type = expression.args[0].type
        b_type = expression.args[1].type
        if isinstance(expression.args[0], Constant):
            a = expression.args[0].value
        else:
            a = expression.args[0]
        if isinstance(expression.args[1], Constant):
            b = expression.args[1].value
        else:
            b = expression.args[1]

        e = Constant[unify_types(a_type, b_type)](
            f(a, b)
        )
        return e

    @add_match(
        ExistentialPredicate,
        lambda expression: expression.head._symbols == expression.body._symbols
    )
    def existential_predicate_process(self, expression):
        free_variable_symbol = expression.head
        body = expression.body
        results = frozenset()

        for elem_set in self.symbol_table.symbols_by_type(
            free_variable_symbol.type
        ).values():
            for elem in elem_set.value:
                elem = Constant[free_variable_symbol.type](frozenset([elem]))
                rsw = ReplaceSymbolWalker({free_variable_symbol: elem})
                rsw_walk = rsw.walk(body)
                pred = self.walk(rsw_walk)
                if pred.value != frozenset():
                    results = results.union(elem.value)
        return Constant[free_variable_symbol.type](results)

    @add_match(ExistentialPredicate)
    def existential_predicate_no_process(self, expression):
        body = self.walk(expression.body)
        if body.type is not Unknown:
            return_type = unify_types(expression.type, body.type)
        else:
            return_type = expression.type

        if (
            isinstance(body, Constant) and
            is_leq_informative(body.type, typing.AbstractSet)
        ):
            body = body.cast(return_type)
            self.symbol_table[expression.head] = body
            return body
        elif (
            body is expression.body and
            return_type is expression.type
        ):
            return expression
        else:
            return self.walk(
                ExistentialPredicate[return_type](expression.head, body)
            )

    @add_match(Query)
    def query(self, expression):
        body = self.walk(expression.body)
        return_type = unify_types(expression.type, body.type)
        body.change_type(return_type)
        expression.head.change_type(return_type)
        if body is expression.body:
            if isinstance(body, Constant):
                self.symbol_table[expression.head] = body
            else:
                self.symbol_table[expression.head] = expression
            return expression
        else:
            return self.walk(
                Query[expression.type](expression.head, body)
            )
