from typing import AbstractSet, Any, Tuple, Callable
from operator import and_, invert

from .type_system import Unknown
from .expression_walker import add_match, expression_iterator

from .solver_datalog_naive import (
    DatalogBasic, Implication,
    extract_datalog_free_variables,
)

from .expressions import (
    Symbol, NonConstant, FunctionApplication,
    NeuroLangException, is_leq_informative, ExpressionBlock,
    Constant
)


class NegativeFact(Implication):
    def __init__(self, antecedent):
        super().__init__(Constant(False), invert(antecedent))

    @property
    def fact(self):
        return self.antecedent

    def __repr__(self):
        return 'NegativeFact{{{} \u2190 {}}}'.format(
            repr(self.antecedent), True
        )

class DatalogBasicNegation(DatalogBasic):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.negated_symbols = {}

    @add_match(Implication(
        FunctionApplication[bool](Symbol, ...),
        NonConstant
    ))
    def statement_intensional(self, expression):
        consequent = expression.consequent
        antecedent = expression.antecedent

        if consequent.functor.name in self.protected_keywords:
            raise NeuroLangException(
                f'symbol {self.constant_set_name} is protected'
            )

        if not is_conjunctive_negation(antecedent):
            raise NeuroLangException(
                f'Expression {antecedent} is not conjunctive'
            )

        consequent_symbols = consequent._symbols - consequent.functor._symbols

        if not consequent_symbols.issubset(antecedent._symbols):
            raise NeuroLangException(
                "All variables on the consequent need to be on the antecedent"
            )

        if consequent.functor.name in self.symbol_table:
            value = self.symbol_table[consequent.functor.name]
            if (
                isinstance(value, Constant) and
                is_leq_informative(value.type, AbstractSet)
            ):
                raise NeuroLangException(
                    'f{consequent.functor.name} has been previously '
                    'defined as Fact or extensional database.'
                )
            eb = self.symbol_table[consequent.functor.name].expressions

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

        self.symbol_table[consequent.functor.name] = ExpressionBlock(eb)

        return expression

    @add_match(NegativeFact)
    def negative_fact(self, expression):
        #negated_fact = expression.fact
        fact = expression.fact.args[0]
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

        if fact.functor.name not in self.negated_symbols:
            if fact.functor.type is Unknown:
                c = Constant(fact.args)
                set_type = c.type
            elif isinstance(fact.functor.type, Callable):
                set_type = Tuple[fact.functor.type.__args__[:-1]]
            else:
                raise NeuroLangException('Fact functor type incorrect')

            self.negated_symbols[fact.functor.name] = \
                Constant[AbstractSet[set_type]](set())

        fact_set = self.negated_symbols[fact.functor.name]

        if isinstance(fact_set, ExpressionBlock):
            raise NeuroLangException(
                f'{fact.functor.name} has been previously '
                'define as intensional predicate.'
            )

        fact_set.value.add(Constant(fact.args))

        return expression

def is_conjunctive_negation(expression):
    return all(
        not isinstance(exp, FunctionApplication) or
        (
            isinstance(exp, FunctionApplication) and
            (
                (
                    isinstance(exp.functor, Constant) and
                    (
                        exp.functor.value is and_ or
                        exp.functor.value is invert
                    )
                ) or all(
                    not isinstance(arg, FunctionApplication)
                    for arg in exp.args
                )
            )
        )
        for _, exp in expression_iterator(expression)
    )

