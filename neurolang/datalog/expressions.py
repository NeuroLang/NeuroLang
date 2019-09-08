"""
Expressions for the intermediate representation of a
Datalog program.
"""

from operator import and_, invert, or_
from typing import Any

from ..expression_walker import PatternWalker, add_match
from ..expressions import (Constant, Definition, ExpressionBlock,
                           FunctionApplication)


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


class Conjunction(Definition):
    def __init__(self, literals):
        self.literals = tuple(literals)

        self._symbols = set()
        for literal in self.literals:
            self._symbols |= literal._symbols

    def __repr__(self):
        return '(' + ' \u2227 '.join(
            repr(e) for e in self.literals
        ) + ')'


class Disjunction(Definition):
    def __init__(self, literals):
        self.literals = tuple(literals)

        self._symbols = set()
        for literal in self.literals:
            self._symbols |= literal._symbols

    def __repr__(self):
        repr_literals = []
        chars = 0
        for literal in self.literals:
            repr_literals.append(repr(literal))
            chars += len(repr_literals[-1])

        if chars < 30:
            join_text = ' \u2228 '
        else:
            join_text = ' \u2228\n'

        return '(' + join_text.join(
            repr(e) for e in self.literals
        ) + ')'


class Negation(Definition):
    def __init__(self, literal):
        self.literal = literal
        self._symbols |= literal._symbols

    def __repr__(self):
        return f'\u00AC{self.literal}'


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
        left = self.walk(conjunction.args[0])
        right = self.walk(conjunction.args[1])

        if isinstance(left, Conjunction):
            conj = left.literals
        else:
            conj = (left,)

        if isinstance(right, Conjunction):
            conj += right.literals
        else:
            conj += (right,)

        return self.walk(Conjunction(conj))

    @add_match(FunctionApplication(Constant(or_), ...))
    def build_disjunction(self, disjunction):
        left = self.walk(disjunction.args[0])
        right = self.walk(disjunction.args[1])

        if isinstance(left, Disjunction):
            disj = left.literals
        else:
            disj = (left,)

        if isinstance(right, Disjunction):
            disj += right.literals
        else:
            disj += (right,)

        return self.walk(Disjunction(disj))

    @add_match(FunctionApplication(Constant(invert), ...))
    def build_negation(self, inversion):
        arg = self.walk(inversion.args[0])
        return self.walk(Negation(arg))

    @add_match(ExpressionBlock)
    def build_conjunction_from_expression_block(self, expression_block):
        return Conjunction(
            self.walk(expression)
            for expression in expression_block.expressions
        )

    @add_match(Implication)
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
