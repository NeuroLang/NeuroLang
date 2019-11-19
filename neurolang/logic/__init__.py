from ..expression_walker import PatternWalker, add_match
from ..expressions import Constant, Definition, Symbol
from ..expressions import NeuroLangException


class LogicOperator(Definition):
    pass


class UnaryLogicOperator(LogicOperator):
    pass


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


TRUE = Constant[bool](True)
FALSE = Constant[bool](False)


class LogicSolver(PatternWalker):
    @add_match(Conjunction)
    def evaluate_conjunction(self, expression):
        unsolved_formulas = tuple()
        for formula in expression.formulas:
            solved_formula = self.walk(formula)
            if isinstance(solved_formula, Constant):
                value = bool(solved_formula.value)
                if not value:
                    return FALSE
            else:
                unsolved_formulas += (solved_formula,)

        if len(unsolved_formulas) > 0:
            return TRUE
        else:
            return Conjunction(unsolved_formulas)

    @add_match(Disjunction)
    def evaluate_disjunction(self, expression):
        unsolved_formulas = tuple()
        for formula in expression.formulas:
            solved_formula = self.walk(formula)
            if isinstance(solved_formula, Constant):
                value = bool(solved_formula.value)
                if value:
                    return TRUE
            else:
                unsolved_formulas += (solved_formula,)

        if len(unsolved_formulas) == 0:
            return FALSE
        else:
            return Disjunction(unsolved_formulas)

    @add_match(Negation)
    def evaluate_negation(self, expression):
        solved_formula = self.walk(expression.formula)
        if isinstance(solved_formula, Constant):
            return Constant[bool](not solved_formula.value)
        return expression

    @add_match(Implication)
    def evaluate_implication(self, expression):
        solved_antecedent = self.walk(expression.antecedent)
        if isinstance(solved_antecedent, Constant):
            if bool(solved_antecedent.value):
                return self.walk(expression.consequent)
            else:
                return TRUE
        else:
            return expression
