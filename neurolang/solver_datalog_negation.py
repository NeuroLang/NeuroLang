from typing import AbstractSet

from .expressions import (
    Symbol, NonConstant, FunctionApplication,
    NeuroLangException, is_leq_informative, ExpressionBlock,
    Constant
)

from operator import and_, invert

from .expression_walker import add_match, expression_iterator

from .solver_datalog_naive import (
    DatalogBasic, Implication,
    extract_datalog_free_variables,
)


class DatalogBasicNegation(DatalogBasic):
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

