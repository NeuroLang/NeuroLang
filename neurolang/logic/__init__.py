from ..exceptions import NeuroLangException
from ..expressions import Constant, Definition, FunctionApplication, Symbol


class LogicOperator(Definition):
    pass


class UnaryLogicOperator(LogicOperator):
    pass


class BinaryLogicOperator(LogicOperator):
    pass


class NaryLogicOperator(LogicOperator):
    pass


class Conjunction(NaryLogicOperator):
    def __init__(self, formulas):
        self.formulas = tuple(formulas)

        self._symbols = set()
        for formula in self.formulas:
            self._symbols |= formula._symbols

    def __repr__(self):
        return '\u22C0(' + ', '.join(
            repr(e) for e in self.formulas
        ) + ')'


class Disjunction(NaryLogicOperator):
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


class Union(NaryLogicOperator):
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

        return '\u222A(' + join_text.join(
            repr(e) for e in self.formulas
        ) + ')'


class Negation(UnaryLogicOperator):
    def __init__(self, formula):
        self.formula = formula
        self._symbols |= formula._symbols

    def __repr__(self):
        return f'\u00AC{self.formula}'


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


class Quantifier(LogicOperator):
    pass


class ExistentialPredicate(Quantifier):
    def __init__(self, head, body):

        if not isinstance(head, Symbol):
            raise NeuroLangException(
                'A symbol should be provided for the '
                'existential quantifier expression'
            )
        if not isinstance(body, Definition):
            raise NeuroLangException(
                'A function application over '
                'predicates should be associated to the quantifier'
            )

        if head not in body._symbols:
            raise NeuroLangException(
                'Symbol should be a free '
                'variable on the predicate'
            )
        self.head = head
        self.body = body
        self._symbols = body._symbols - {head}

    def __repr__(self):
        r = (
            u'\u2203{{{}: {} st {}}}'
            .format(self.head, self.__type_repr__, self.body)
        )
        return r


class UniversalPredicate(Quantifier):
    def __init__(self, head, body):

        if not isinstance(head, Symbol):
            raise NeuroLangException(
                'A symbol should be provided for the '
                'universal quantifier expression'
            )
        if not isinstance(body, Definition):
            raise NeuroLangException(
                'A function application over '
                'predicates should be associated to the quantifier'
            )

        if head not in body._symbols:
            raise NeuroLangException(
                'Symbol should be a free '
                'variable on the predicate'
            )
        self.head = head
        self.body = body
        self._symbols = body._symbols - {head}

    def __repr__(self):
        r = (
            u'\u2200{{{}: {} st {}}}'
            .format(self.head, self.__type_repr__, self.body)
        )
        return r


class Predicate(FunctionApplication, LogicOperator):
    pass


TRUE = Constant[bool](True)
FALSE = Constant[bool](False)
