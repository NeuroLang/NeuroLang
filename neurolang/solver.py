from collections import OrderedDict
import itertools
from operator import (
    invert, and_, or_,
    add, sub, mul, truediv, pos, neg
)
import typing

from .exceptions import NeuroLangException
from .expressions import (
    Expression, NonConstant, ExistentialPredicate, UniversalPredicate,
    Symbol, Constant, Predicate,
    FunctionApplication, Query, Definition, get_type_and_value, is_subtype,
    unify_types, ToBeInferred
)
from .expression_walker import (
    add_match, ExpressionBasicEvaluator, ReplaceSymbolWalker,
    PatternWalker
)

T = typing.TypeVar('T')


class NeuroLangPredicateException(NeuroLangException):
    pass


class FiniteDomain(object):
    pass


class FiniteDomainSet(frozenset):
    pass


class GenericSolver(ExpressionBasicEvaluator):
    @property
    def plural_type_name(self):
        return self.type_name + 's'

    def set_symbol_table(self, symbol_table):
        self.symbol_table = symbol_table


class SetBasedSolver(GenericSolver[T]):
    """
    A predicate `in <set>` which results in the `<set>` given as parameter
    `and` and `or` operations between sets which are disjunction and
    conjunction.
    """

    def predicate_in(
        self, argument: typing.AbstractSet[T]
    )->typing.AbstractSet[T]:
        return argument

    @add_match(
        FunctionApplication(Constant(invert), (Constant[typing.AbstractSet],)),
        lambda expression: isinstance(
            get_type_and_value(expression.args[0])[1],
            FiniteDomainSet
        )
    )
    def rewrite_finite_domain_inversion(self, expression):
        set_constant = expression.args[0]
        set_type, set_value = get_type_and_value(set_constant)
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
        a_type, a = get_type_and_value(expression.args[0])
        b_type, b = get_type_and_value(expression.args[1])
        e = Constant[a_type](
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
                rsw = ReplaceSymbolWalker(free_variable_symbol, elem)
                rsw_walk = rsw.walk(body)
                pred = self.walk(rsw_walk)
                if pred.value != frozenset():
                    results = results.union(elem.value)
        return Constant[free_variable_symbol.type](results)

    @add_match(ExistentialPredicate)
    def existential_predicate_no_process(self, expression):
        body = self.walk(expression.body)
        if body.type is not ToBeInferred:
            return_type = unify_types(expression.type, body.type)
        else:
            return_type = expression.type

        if (
            isinstance(body, Constant) and
            is_subtype(body.type, typing.AbstractSet)
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


class BooleanRewriteSolver(PatternWalker):
    @add_match(
       FunctionApplication(Constant, (Expression[bool],) * 2),
       lambda expression: (
           expression.functor.value in (or_, and_) and
           expression.type is not bool
        )
    )
    def cast_binary(self, expression):
        functor, args = expression.functor, expression.args
        new_functor = functor.cast(typing.Callable[[bool, bool], bool])
        new_application = FunctionApplication[bool](new_functor, args)
        return self.walk(new_application)

    @add_match(
       FunctionApplication(Constant(invert), (Expression[bool],)),
       lambda expression: (
           expression.type is not bool
        )
    )
    def cast_unary(self, expression):
        functor, args = expression.functor, expression.args
        new_functor = functor.cast(typing.Callable[[bool, bool], bool])
        new_application = FunctionApplication[bool](new_functor, args)
        return self.walk(new_application)

    @add_match(
        FunctionApplication[bool](
            Constant(invert), (
                FunctionApplication[bool](
                    Constant(invert), (
                        Expression[bool],
                    )
                ),
            )
        )
    )
    def simplify_double_inversion(self, expression):
        return self.walk(expression.args[0].args[0])

    @add_match(
        FunctionApplication[bool](
            Constant(...),
            (NonConstant[bool], Constant[bool])
        ),
        lambda expression: (
            expression.functor.value in (or_, and_)
        )
    )
    def dual_operator(self, expression):
        return self.walk(
            FunctionApplication[bool](
                expression.functor,
                expression.args[::-1]
            )
        )

    @add_match(
        FunctionApplication[bool](Constant, (
            NonConstant[bool],
            FunctionApplication[bool](
                Constant,
                (Constant[bool], Expression[bool])
            )
        )),
        lambda expression: (
            expression.functor.value in (or_, and_)
            and expression.args[1].functor.value is expression.functor.value
        )
    )
    def bring_constants_up_left(self, expression):
        outer = expression
        inner = expression.args[1]
        new_inner = FunctionApplication[bool](
            inner.functor, (outer.args[0], inner.args[1])
        )
        new_outer = FunctionApplication[bool](
            outer.functor, (inner.args[0], new_inner)
        )
        return self.walk(new_outer)

    @add_match(
        FunctionApplication[bool](
            Constant(invert),
            (FunctionApplication[bool](
                Constant(or_),
                (Expression[bool], Expression[bool])
            ),)
        )
    )
    def neg_disj_to_conj(self, expression):
        return self.walk(
            FunctionApplication[bool](
                Constant(and_), (
                    (~expression.args[0].args[0]).cast(bool),
                    (~expression.args[0].args[1]).cast(bool)
                )
            )
        )

    @add_match(
        FunctionApplication[bool](
            Constant(...), (FunctionApplication[bool], Expression[bool])
        ),
        lambda expression: expression.functor.value in (or_, and_) and
        any(
            isinstance(arg, Definition) for arg in expression.args[0].args
        ) and (
            not isinstance(expression.args[1], Definition) or (
                all(
                    not isinstance(arg, Definition)
                    for arg in expression.args[1].args
                )
            )
        )
    )
    def conjunction_composition_dual(self, expression):
        return self.walk(
            FunctionApplication[bool](
                Constant(expression.functor.value),
                (expression.args[1], expression.args[0])
            )
        )

    @add_match(
        FunctionApplication[bool](
            Constant(...), (Definition, Expression[bool])
        ),
        lambda expression: expression.functor.value in (or_, and_) and
        not isinstance(expression.args[1], Definition)
    )
    def conjunction_definition_dual(self, expression):
        return self.walk(
            FunctionApplication[bool](
                Constant(expression.functor.value),
                (expression.args[1], expression.args[0])
            )
        )

    @add_match(
        FunctionApplication(
            Constant(and_),
            (
                FunctionApplication(Constant(and_), ...),
                ...
            )
        )
    )
    def conjunction_distribution(self, expression):
        return self.walk(FunctionApplication[expression.type](
            expression.functor,
            (
                expression.args[0].args[0],
                FunctionApplication[expression.args[0].type](
                    expression.args[0].functor,
                    (expression.args[0].args[1], expression.args[1])
                )
            )
        ))

    @add_match(
        FunctionApplication(Constant(...), (NonConstant, NonConstant)),
        lambda expression: expression.functor.value in (or_, and_)
    )
    def partial_binary_evaluation(self, expression):
        '''Partially evaluate a binary AND or OR operator with
        non-constant arguments by giving priority to the evaluation of
        the first argument.

        Parameters
        ----------
        expression : FunctionApplication
            A OR or AND binary operator application.

        Returns
        -------
        FunctionApplication
            Same expression passed as parameter where the first
            argument was walked on and the second argument was
            potentially walked on if the walk on the first argument
            did not modify its expression.

        '''
        first_arg, second_arg = expression.args
        # we walk on the first argument
        walk_first_result = self.walk(first_arg)
        # and replace that argument with the result of the walk
        new_args = (walk_first_result, expression.args[1])
        # if the walk on the first argument did not change anything
        # we walk on the second argument and replace it with the result
        if walk_first_result is first_arg:
            new_args = (first_arg, self.walk(expression.args[1]))

        # if the expression arguments did not change, we stop walking here
        if new_args == expression.args:
            return expression
        # otherwise, we keep walking
        else:
            return self.walk(
                FunctionApplication[bool](expression.functor, new_args)
            )


class BooleanOperationsSolver(PatternWalker):
    @add_match(FunctionApplication(Constant(invert), (Constant[bool],)))
    def rewrite_boolean_inversion(self, expression):
        return Constant(not expression.args[0].value)

    @add_match(
        FunctionApplication(Constant(and_), (Constant[bool], Constant[bool]))
    )
    def rewrite_boolean_and(self, expression):
        return Constant(expression.args[0].value and expression.args[1].value)

    @add_match(
        FunctionApplication(Constant(or_), (Constant[bool], Constant[bool]))
    )
    def rewrite_boolean_or(self, expression):
        return Constant(expression.args[0].value or expression.args[1].value)

    @add_match(
        FunctionApplication(Constant(or_), (True, Expression[bool]))
    )
    def rewrite_boolean_or_l(self, expression):
        return Constant(True)

    @add_match(
        FunctionApplication(Constant(or_), (Expression[bool], True))
    )
    def rewrite_boolean_or_r(self, expression):
        return Constant(True)

    @add_match(
        FunctionApplication(Constant(and_), (False, Expression[bool]))
    )
    def rewrite_boolean_and_l(self, expression):
        return Constant(False)

    @add_match(
        FunctionApplication(Constant(and_), (Expression[bool], False))
    )
    def rewrite_boolean_and_r(self, expression):
        return Constant(False)


class NumericOperationsSolver(PatternWalker[T]):
    @add_match(
        FunctionApplication[ToBeInferred](Constant, (Expression[T],) * 2),
        lambda expression: expression.functor.value in (add, sub, mul, truediv)
    )
    def cast_binary(self, expression):
        type = expression.args[0].type
        functor = expression.functor.cast(typing.Callable[[type, type], type])
        if functor is not expression.functor:
            new_expression = FunctionApplication[type](
                functor, expression.args
            )
        else:
            new_expression = expression.cast(type)
        return self.walk(new_expression)

    @add_match(
        FunctionApplication(Constant, (Expression[T],)),
        lambda expression: expression.functor.value in (pos, neg)
    )
    def cast_unary(self, expression):
        return self.walk(expression.cast(expression.args[0].type))


class DatalogSolver(
        BooleanRewriteSolver,
        BooleanOperationsSolver,
        NumericOperationsSolver[int],
        NumericOperationsSolver[float],
        GenericSolver
):
    '''
    WIP Solver with queries having the semantics of Datalog.
    For now predicates work only on constants on the symbols table
    '''

    @add_match(
        Query,
        guard=lambda expression: (
            expression.head._symbols == expression.body._symbols
        )
    )
    def query_resolution(self, expression):
        out_query_type = expression.type
        if out_query_type is ToBeInferred:
            out_query_type = typing.AbstractSet[expression.head.type]

        result = []

        symbols_domains = self.quantifier_head_symbols_and_adom(
            expression.head
        )
        for symbol_values in itertools.product(*symbols_domains.values()):
            body = expression.body
            for i, s in enumerate(symbols_domains.keys()):

                if s in body._symbols:
                    rsw = ReplaceSymbolWalker(s, symbol_values[i][1])
                    body = rsw.walk(body)

            res = self.walk(body)
            if res.value:
                if isinstance(expression.head, Symbol):
                    result.append(symbol_values[0][0])
                else:
                    result.append(tuple(zip(*symbol_values))[0])

        return Constant[out_query_type](
            frozenset(result)
        )

    @add_match(
        ExistentialPredicate,
        lambda expression: expression.head._symbols == expression.body._symbols
    )
    def existential_predicate(self, expression):
        symbols_domains = self.quantifier_head_symbols_and_adom(
            expression.head
        )

        for symbol_values in itertools.product(*symbols_domains.values()):
            body = expression.body
            for i, s in enumerate(symbols_domains.keys()):
                if s in body._symbols:
                    rsw = ReplaceSymbolWalker(s, symbol_values[i][1])
                    body = rsw.walk(body)

            res = self.walk(body)
            if res.value:
                return Constant(True)

        return Constant(False)

    @add_match(
        UniversalPredicate,
        lambda expression: expression.head._symbols == expression.body._symbols
    )
    def universal_predicate(self, expression):
        symbols_domains = self.quantifier_head_symbols_and_adom(
            expression.head
        )
        for symbol_values in itertools.product(*symbols_domains.values()):
            body = expression.body
            for i, s in enumerate(symbols_domains.keys()):
                if s in body._symbols:
                    rsw = ReplaceSymbolWalker(s, symbol_values[i][1])
                    body = rsw.walk(body)

            res = self.walk(body)
            if not res.value:
                return Constant(False)

        return Constant(True)

    def quantifier_head_symbols_and_adom(self, head):
        '''
        Returns an ordered dictionary with the symbols of the quantifier head
        as keys and the active domain for each symbol as value.
        '''
        if (
            isinstance(head, Constant) and
            is_subtype(head.type, typing.Tuple) and
            all(isinstance(a, Symbol) for a in head.value)
        ):
            symbols_in_head = head.value
        else:
            symbols_in_head = (head,)

        constants = tuple((
            (
                (k, v)
                for k, v in self.symbol_table.symbols_by_type(
                    sym.type
                ).items()
                if isinstance(v, Constant)
            )
            for sym in symbols_in_head
        ))
        return OrderedDict(zip(symbols_in_head, constants))
