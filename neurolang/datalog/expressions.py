"""
Expressions for the intermediate representation of a
Datalog program.
"""

from operator import and_, invert, or_
from typing import Any

from ..expression_walker import PatternWalker, add_match
from ..expressions import (Constant, Definition, ExpressionBlock,
                           FunctionApplication)


class LogicOperator(Definition):
    pass


class UnaryLogicOperator(LogicOperator):
    pass


class Implication(LogicOperator):
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


class Conjunction(LogicOperator):
    def __init__(self, formulas):
        self.formulas = tuple(formulas)

        self._symbols = set()
        for formula in self.formulas:
            self._symbols |= formula._symbols

    def __repr__(self):
        return '\u22C0(' + ', '.join(
            repr(e) for e in self.formulas
        ) + ')'


class Disjunction(LogicOperator):
    def __init__(self, formulas):
        self.formulas = tuple(formulas)

        self._symbols = set()
        for formula in self.formulas:
            self._symbols |= formula._symbols

    def __repr__(self):
        repr_formulas = []
        chars = 0
        for formula in self.formulas:
            repr_formulas.append(repr(formula))
            chars += len(repr_formulas[-1])

        if chars < 30:
            join_text = ', '
        else:
            join_text = ',\n'

        return '\u22C1(' + join_text.join(
            repr(e) for e in self.formulas
        ) + ')'


class Negation(UnaryLogicOperator):
    def __init__(self, formula):
        self.formula = formula
        self._symbols |= formula._symbols

    def __repr__(self):
        return f'\u00AC{self.formula}'


class Undefined(Constant):
    def __repr__(self):
        return 'UNDEFINED'


class NullConstant(Constant):
    def __repr__(self):
        return 'NULL'


UNDEFINED = Undefined(None)
NULL = NullConstant[Any](None)


class TranslateToLogic(PatternWalker):
    @add_match(FunctionApplication(Constant(and_), ...))
    def build_conjunction(self, conjunction):
        args = tuple()
        for arg in conjunction.args:
            new_arg = self.walk(arg)
            if isinstance(new_arg, Conjunction):
                args += new_arg.formulas
            else:
                args += (new_arg,)

        return self.walk(Conjunction(args))

    @add_match(FunctionApplication(Constant(or_), ...))
    def build_disjunction(self, disjunction):
        args = tuple()
        for arg in disjunction.args:
            new_arg = self.walk(arg)
            if isinstance(new_arg, Disjunction):
                args += new_arg.formulas
            else:
                args += (new_arg,)

        return self.walk(Disjunction(args))

    @add_match(FunctionApplication(Constant(invert), ...))
    def build_negation(self, inversion):
        arg = self.walk(inversion.args[0])
        return self.walk(Negation(arg))

    @add_match(ExpressionBlock)
    def build_conjunction_from_expression_block(self, expression_block):
        formulas = tuple()
        for expression in expression_block.expressions:
            new_exp = self.walk(expression)
            formulas += (new_exp,)
        return self.walk(Disjunction(formulas))

    @add_match(
        Implication(..., Constant(True)),
        lambda x: not isinstance(x, Fact)
    )
    def translate_true_implication(self, implication):
        return self.walk(Fact(implication.consequent))

    @add_match(
        Implication(..., FunctionApplication(Constant, ...)),
        lambda implication: (
            implication.antecedent.functor.value is and_ or
            implication.antecedent.functor.value is or_ or
            implication.antecedent.functor.value is invert
        )
    )
    def translate_implication(self, implication):
        new_consequent = self.walk(implication.consequent)
        new_antecedent = self.walk(implication.antecedent)

        if (
            new_consequent is not implication.consequent or
            new_antecedent is not implication.antecedent
        ):
            implication = self.walk(
                Implication(new_consequent, new_antecedent)
            )

        return implication
