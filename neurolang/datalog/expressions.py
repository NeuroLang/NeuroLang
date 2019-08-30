from typing import Any

from ..expressions import (
    Definition, Constant
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
