'''
Naive implementation of non-typed datalog. There's no optimizations and will be
surely very slow
'''
from typing import AbstractSet, Any, Tuple, Callable
from itertools import product
from operator import and_

from .utils import OrderedSet

from .expressions import (
    FunctionApplication, Constant, NeuroLangException, is_leq_informative,
    Statement, Symbol, Lambda, ExpressionBlock, Expression,
    Query, ExistentialPredicate, UniversalPredicate, Quantifier,
    Unknown
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
    In the symbol table the value for a symbol `S` is implemented as:

    * If `S` is part of the extensional database, then the value of the symbol
    is a set of tuples `a` representing `S(*a)` as facts

    * If `S` is part of the intensional database then its value is an
    `ExpressionBlock` such that each expression is a case of the symbol
    each expression is a `Lambda` instance where the `function_expression` is
    the query and the `args` are the needed projection. For instance
    `Q(x) :- R(x, x)` and `Q(x) :- T(x)` is represented as a symbol `Q`
     with value `ExpressionBlock((Lambda(R(x, x), (x,)), Lambda(T(x), (x,))))`
    '''

    protected_keywords = set()

    def function_equals(self, a: Any, b: Any) -> bool:
        return a == b

    @add_match(Fact(FunctionApplication[bool](Symbol, ...)))
    def fact(self, expression):
        fact = expression.fact
        if fact.functor.name in self.protected_keywords:
            raise NeuroLangException(
                f'symbol {self.constant_set_name} is protected'
            )

        if any(
            not isinstance(a, Constant)
            for a in fact.args
        ):
            raise NeuroLangException(
                'Facts can only have constants as arguments'
            )

        if fact.functor.name not in self.symbol_table:
            if fact.functor.type is Unknown:
                c = Constant(fact.args)
                set_type = c.type
            elif isinstance(fact.functor.type, Callable):
                set_type = Tuple[fact.functor.type.__args__[:-1]]
            else:
                raise NeuroLangException('Fact functor type incorrect')

            self.symbol_table[fact.functor.name] = \
                Constant[AbstractSet[set_type]](set())

        fact_set = self.symbol_table[fact.functor.name]

        if isinstance(fact_set, ExpressionBlock):
            raise NeuroLangException(
                f'{fact.functor.name} has been previously '
                'define as intensional predicate.'
            )

        fact_set.value.add(Constant(fact.args))

        return expression

    @add_match(Statement(
        FunctionApplication[bool](Symbol, ...),
        Constant[bool](True)
    ))
    def statement_extensional(self, expression):
        return self.walk(Fact[expression.type](expression.lhs))

    @add_match(Statement(
        FunctionApplication[bool](Symbol, ...),
        Expression
    ))
    def statement_intensional(self, expression):
        lhs = expression.lhs
        rhs = expression.rhs

        if lhs.functor.name in self.protected_keywords:
            raise NeuroLangException(
                f'symbol {self.constant_set_name} is protected'
            )

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
                is_leq_informative(value.type, AbstractSet)
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
        return {
            k: v for k, v in self.symbol_table.items()
            if (
                k not in self.protected_keywords and
                isinstance(v, ExpressionBlock)
            )
        }

    def extensional_database(self):
        ret = self.symbol_table.symbols_by_type(AbstractSet)
        for keyword in self.protected_keywords:
            del ret[keyword]
        return ret


class NaiveDatalog(DatalogBasic):
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
        FunctionApplication(Constant[AbstractSet], (Constant,)),
        lambda exp: not is_subtype(exp.args[0].type, Tuple)
    )
    def function_application_edb_notuple(self, expression):
        return self.walk(FunctionApplication(
            expression.functor,
            (Constant(expression.args),)
        ))

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
            is_leq_informative(expression.head.type, Tuple)
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
            is_leq_informative(expression.head.type, Tuple)
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
            is_leq_informative(expression.head.type, Tuple)
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
    @add_match(FunctionApplication(Constant(and_), ...))
    def conjunction(self, expression):
        fvs = OrderedSet()
        for arg in expression.args:
            fvs |= self.walk(arg)
        return fvs

    @add_match(FunctionApplication)
    def extract_variables_fa(self, expression):
        args = expression.args

        variables = OrderedSet()
        for a in args:
            if isinstance(a, Symbol):
                variables.add(a)
            elif isinstance(a, Constant):
                pass
            else:
                raise NeuroLangException('Not a Datalog function application')
        return variables

    @add_match(Quantifier)
    def extract_variables_q(self, expression):
        return self.walk(expression.body) - expression.head._symbols

    @add_match(Statement)
    def extract_variables_s(self, expression):
        return self.walk(expression.rhs) - self.walk(expression.lhs)

    @add_match(...)
    def _(self, expression):
        return OrderedSet()


def extract_datalog_free_variables(expression):
    '''extract variables from expression knowing that it's in Datalog format'''
    efvw = ExtractDatalogFreeVariablesWalker()
    return efvw.walk(expression)


class ExtractDatalogPredicates(PatternWalker):
    @add_match(FunctionApplication(Constant(and_), ...))
    def conjunction(self, expression):
        res = OrderedSet()
        for arg in expression.args:
            res |= self.walk(arg)
        return res

    @add_match(FunctionApplication)
    def extract_predicates_fa(self, expression):
        return OrderedSet([expression])

    @add_match(ExpressionBlock)
    def expression_block(self, expression):
        res = OrderedSet()
        for exp in expression.expressions:
            res |= self.walk(exp)
        return res

    @add_match(Lambda)
    def lambda_(self, expression):
        return self.walk(expression.function_expression)


def extract_datalog_predicates(expression):
    """
    extract predicates from expression
    knowing that it's in Datalog format
    """

    edp = ExtractDatalogPredicates()
    return edp.walk(expression)
