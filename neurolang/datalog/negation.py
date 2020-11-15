from typing import AbstractSet, Callable, Tuple

from ..expression_walker import add_match, PatternWalker
from ..expressions import (Constant, FunctionApplication, NeuroLangException,
                           NonConstant, Symbol, is_leq_informative)
from ..type_system import Unknown
from .basic_representation import DatalogProgram, UnionOfConjunctiveQueries
from .expression_processing import extract_logic_free_variables
from ..logic import Conjunction, Union, Implication, Negation, Quantifier


class NegativeFact(Implication):
    '''This class defines negative facts. They are composed of an inverted
    antecedent and False in the consequent. It is not necessary that the
    initialization parameter is inverted.'''

    def __init__(self, antecedent):
        super().__init__(Constant(False), Negation(antecedent))

    @property
    def fact(self):
        return self.antecedent

    def __repr__(self):
        return 'NegativeFact{{{} \u2190 {}}}'.format(
            repr(self.antecedent), True
        )


class DatalogProgramNegationMixin(PatternWalker):
    '''Datalog solver that implements negation. Adds the possibility of
    inverted terms when checking that expressions are in conjunctive
    normal form.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.negated_symbols = {}

    @add_match(Negation(Constant[bool]))
    def negation_constant(self, expression):
        if expression.formula.value:
            return Constant[bool](False)
        else:
            return Constant[bool](True)

    @add_match(
        Implication(FunctionApplication[bool](Symbol, ...), NonConstant)
    )
    def statement_intensional(self, expression):
        consequent = expression.consequent
        antecedent = expression.antecedent

        self._check_implication(consequent, antecedent)

        consequent_symbols = consequent._symbols - consequent.functor._symbols

        if not consequent_symbols.issubset(antecedent._symbols):
            raise NeuroLangException(
                "All variables on the consequent need to be on the antecedent"
            )

        if consequent.functor.name in self.symbol_table:
            value = self.symbol_table[consequent.functor.name]
            self._is_previously_defined(value)
            disj = self.symbol_table[consequent.functor.name].formulas
            self._is_in_idb(expression, disj)

        else:
            disj = tuple()

        if expression not in disj:
            disj += (expression,)

        symbol = consequent.functor.cast(UnionOfConjunctiveQueries)
        self.symbol_table[symbol] = Union(disj)

        return expression

    def _check_implication(self, consequent, antecedent):
        if consequent.functor.name in self.protected_keywords:
            raise NeuroLangException(
                f'symbol {self.constant_set_name} is protected'
            )

        if not is_conjunctive_negation(antecedent):
            raise NeuroLangException(
                f'Expression {antecedent} is not conjunctive'
            )

    def _is_previously_defined(self, value):
        if (
            isinstance(value, Constant) and
            is_leq_informative(value.type, AbstractSet)
        ):
            raise NeuroLangException(
                'f{consequent.functor.name} has been previously '
                'defined as Fact or extensional database.'
            )

    def _is_in_idb(self, expression, eb):
        if (
            not isinstance(eb[0].consequent, FunctionApplication) or
            len(extract_logic_free_variables(eb[0].consequent.args)
                ) != len(expression.consequent.args)
        ):
            raise NeuroLangException(
                f"{eb[0].consequent} is already in the IDB "
                f"with different signature."
            )

    @add_match(NegativeFact)
    def negative_fact(self, expression):
        fact = expression.fact.formula
        if fact.functor.name in self.protected_keywords:
            raise NeuroLangException(
                f'symbol {self.constant_set_name} is protected'
            )

        if any(not isinstance(a, Constant) for a in fact.args):
            raise NeuroLangException(
                'Facts can only have constants as arguments'
            )

        self._not_in_negated_symbol(fact)

        fact_set = self.negated_symbols[fact.functor.name]

        if isinstance(fact_set, Union):
            raise NeuroLangException(
                f'{fact.functor.name} has been previously '
                'define as intensional predicate.'
            )

        fact_set.value.add(Constant(fact.args))

        return expression

    def _not_in_negated_symbol(self, fact):
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


def is_conjunctive_negation(expression):
    stack = [expression]
    while stack:
        exp = stack.pop()
        if exp == Constant(True) or exp == Constant(False):
            pass
        elif isinstance(exp, FunctionApplication):
            stack += [
                arg for arg in exp.args
                if isinstance(arg, FunctionApplication)
            ]
        elif isinstance(exp, Conjunction):
            stack += exp.formulas
        elif isinstance(exp, Negation):
            stack.append(exp.formula)
        else:
            return False

    return True


class DatalogProgramNegation(DatalogProgramNegationMixin, DatalogProgram):
    pass
