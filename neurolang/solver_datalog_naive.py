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
    Symbol, Lambda, ExpressionBlock, Expression, Definition,
    Query, ExistentialPredicate, Quantifier,
)

from .type_system import Unknown
from .expression_walker import (
    add_match, PatternWalker, expression_iterator,
)


class Implication(Definition):
    """Expression of the form `P(x) \u2190 Q(x)`"""

    def __init__(self, consequent, antecedent):
        self.consequent = consequent
        self.antecedent = antecedent
        self._symbols = consequent._symbols | antecedent._symbols

    def __repr__(self):
        return 'Implication{{{} \u2190 {}}}'.format(
            repr(self.consequent), repr(self.antecedent)
        )


class Fact(Implication):
    def __init__(self, consequent):
        super().__init__(consequent, Constant(True))

    @property
    def fact(self):
        return self.consequent

    def __repr__(self):
        return 'Fact{{{} \u2190 {}}}'.format(
            repr(self.consequent), True
        )


class Undefined(Constant):
    def __repr__(self):
        return 'UNDEFINED'


class NullConstant(Constant):
    def __repr__(self):
        return 'NULL'


UNDEFINED = Undefined(None)
NULL = NullConstant[Any](None)


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

        self._initialize_fact_set_if_needed(fact)
        fact_set = self.symbol_table[fact.functor]

        if isinstance(fact_set, ExpressionBlock):
            raise NeuroLangException(
                f'{fact.functor} has been previously '
                'define as intensional predicate.'
            )

        fact_set.value.add(Constant(fact.args))

        return expression

    def _initialize_fact_set_if_needed(self, fact):
        if fact.functor not in self.symbol_table:
            if fact.functor.type is Unknown:
                c = Constant(fact.args)
                set_type = c.type
            elif isinstance(fact.functor.type, Callable):
                set_type = Tuple[fact.functor.type.__args__[:-1]]
            else:
                raise NeuroLangException('Fact functor type incorrect')

            self.symbol_table[fact.functor] = \
                Constant[AbstractSet[set_type]](set())

    @add_match(Implication(
        FunctionApplication[bool](Symbol, ...),
        Constant[bool](True)
    ))
    def statement_extensional(self, expression):
        return self.walk(Fact(expression.consequent))

    @add_match(Implication(
        FunctionApplication[bool](Symbol, ...),
        Expression
    ))
    def statement_intensional(self, expression):
        consequent = expression.consequent
        antecedent = expression.antecedent

        self._validate_implication_syntax(consequent, antecedent)

        if consequent.functor in self.symbol_table:
            value = self.symbol_table[consequent.functor]
            if (
                isinstance(value, Constant) and
                is_leq_informative(value.type, AbstractSet)
            ):
                raise NeuroLangException(
                    'f{consequent.functor} has been previously '
                    'defined as Fact or extensional database.'
                )
            eb = self.symbol_table[consequent.functor].expressions

            if (
                not isinstance(eb[0].consequent, FunctionApplication) or
                len(extract_datalog_free_variables(eb[0].consequent.args)) !=
                len(expression.consequent.args)
            ):
                raise NeuroLangException(
                    f"{eb[0].consequent} is already in the IDB "
                    f"with different signature."
                )
        else:
            eb = tuple()

        eb = eb + (expression,)

        self.symbol_table[consequent.functor] = ExpressionBlock(eb)

        return expression

    def _validate_implication_syntax(self, consequent, antecedent):
        if consequent.functor.name in self.protected_keywords:
            raise NeuroLangException(
                f'symbol {self.constant_set_name} is protected'
            )

        if any(
            not isinstance(arg, (Constant, Symbol))
            for arg in consequent.args
        ):
            raise NeuroLangException(
                f'The consequent {consequent} can only be '
                'constants or symbols'
            )

        consequent_symbols = consequent._symbols - consequent.functor._symbols

        if not consequent_symbols.issubset(antecedent._symbols):
            raise NeuroLangException(
                "All variables on the consequent need to be on the antecedent"
            )

        if not is_conjunctive_expression(antecedent):
            raise NeuroLangException(
                f'Expression {antecedent} is not conjunctive'
            )

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

    def builtins(self):
        return self.symbol_table.symbols_by_type(Callable)


class SolverNonRecursiveDatalogNaive(DatalogBasic):
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
            extract_datalog_free_variables(e.antecedent) -
            extract_datalog_free_variables(e.consequent)
        ) > 0
    )
    def implication_add_existential(self, expression):
        consequent = expression.consequent
        antecedent = expression.antecedent

        fv_consequent = extract_datalog_free_variables(consequent)
        fv_antecedent = (
            extract_datalog_free_variables(antecedent) -
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
        FunctionApplication(ExpressionBlock, ...),
        lambda e: all(
            isinstance(a, Constant) for a in e.args
        )
    )
    def evaluate_datalog_conjunction(self, expression):
        for exp in expression.functor.expressions:
            fa = FunctionApplication[bool](exp, expression.args)
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
            extract_datalog_free_variables(e.body) >
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

    @add_match(Implication)
    def extract_variables_s(self, expression):
        return (
            self.walk(expression.antecedent) -
            self.walk(expression.consequent)
        )

    @add_match(ExpressionBlock)
    def extract_variables_eb(self, expression):
        res = set()
        for exp in expression.expressions:
            res |= self.walk(exp)

        return res

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
            fa_free_variables = extract_datalog_free_variables(body)
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
